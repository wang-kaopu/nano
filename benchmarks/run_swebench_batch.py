#!/usr/bin/env python3
"""可恢复的 SWE-bench Lite 批量运行器。

每个实例串联：拉取镜像 → 创建 worktree → 运行 Agent → 生成 patch → 官方评分。
支持 --resume 断点续跑和 --dry-run 预演。
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = "/root/datasets/swebench-lite/test.json"
DEFAULT_CACHE_ROOT = "/root/swebench-cache/repos"
DEFAULT_OUTPUT_ROOT = "/root/swebench-runs"
DEFAULT_PERMISSIONS_PATH = "/root/pico/benchmarks/swebench_permissions.json"
DEFAULT_ENV_FILE = "/root/pico/.env"

FIVE_INSTANCE_IDS = [
    "astropy__astropy-12907",
    "astropy__astropy-14182",
    "astropy__astropy-14365",
    "astropy__astropy-14995",
    "astropy__astropy-6938",
]

ALLOWED_TOOLS = ("read_file", "list_files", "search", "grep_search", "patch_file", "run_shell")

# 状态机
STATES = [
    "pending",
    "preparing",
    "image_ready",
    "worktree_ready",
    "agent_running",
    "patch_generated",
    "evaluating",
    "resolved",
    "unresolved",
    "agent_error",
    "infra_error",
    "interrupted",
]

TERMINAL_STATES = {"resolved", "unresolved", "agent_error", "infra_error"}
RESUMABLE_SKIP_STATES = {"resolved", "unresolved"}
RESUMABLE_CONTINUE_STATES = {"patch_generated", "evaluating"}
RETRYABLE_STATES = {"infra_error"}

# 确保 score 不回退到宿主机环境
SCORER_SAFE_ENV = {
    "HOME": os.environ.get("HOME", "/root"),
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "LOGNAME": os.environ.get("LOGNAME", "root"),
    "USER": os.environ.get("USER", "root"),
    "LANG": os.environ.get("LANG", "en_US.UTF-8"),
}


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: Any) -> None:
    """原子写入 JSON，避免并发读取到半截文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """带统一超时和编码的执行器。"""
    defaults: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 120}
    defaults.update(kwargs)
    return subprocess.run(args, **defaults)


def timed_log(msg: str) -> None:
    """带时间戳的进度输出。"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 文件锁
# ---------------------------------------------------------------------------


@dataclass
class InstanceLock:
    """基于 fcntl 的实例文件锁，防止同实例重复运行。"""

    path: Path
    _fd: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.path, "w")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fd.write(str(os.getpid()))
            self._fd.flush()
            return True
        except (IOError, OSError):
            self._fd.close()
            self._fd = None
            return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            self._fd.close()
            self._fd = None


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class BatchConfig:
    dataset_path: str = DEFAULT_DATASET_PATH
    cache_root: str = DEFAULT_CACHE_ROOT
    output_root: str = DEFAULT_OUTPUT_ROOT
    permissions_path: str = DEFAULT_PERMISSIONS_PATH
    env_file: str = DEFAULT_ENV_FILE
    provider: str = "deepseek"
    model: str | None = None
    temperature: float = 0.0
    max_steps: int = 40
    agent_timeout: int = 2700
    evaluation_timeout: int = 1800
    max_parallel: int = 1
    namespace: str = "swebench"
    dry_run: bool = False
    resume: bool = False
    instance_ids: list[str] = field(default_factory=lambda: list(FIVE_INSTANCE_IDS))
    swebench_path: str = "/root/SWE-bench"
    swebench_python: str = "/root/SWE-bench/.venv/bin/python"
    gold_smoke: str | None = None


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------


def load_instances(config: BatchConfig) -> list[dict[str, Any]]:
    """从本地 JSON 数据集只读取公开字段。"""
    data = read_json(Path(config.dataset_path))
    requested = set(config.instance_ids)
    rows: list[dict[str, Any]] = []
    for item in data:
        iid = str(item.get("instance_id", ""))
        if iid in requested:
            rows.append(
                {
                    "instance_id": iid,
                    "repo": str(item["repo"]),
                    "base_commit": str(item["base_commit"]),
                    "version": str(item.get("version", "")),
                    "problem_statement": str(item.get("problem_statement", "")),
                }
            )
    missing = requested - {r["instance_id"] for r in rows}
    if missing:
        raise ValueError(f"数据集中缺少实例: {', '.join(sorted(missing))}")
    return rows


# ---------------------------------------------------------------------------
# 镜像名推导
# ---------------------------------------------------------------------------


def instance_image_name(instance: dict[str, Any], namespace: str = "swebench") -> str:
    """通过 SWE-bench TestSpec 动态计算官方 instance image key。"""
    sys.path.insert(0, "/root/SWE-bench")
    from swebench.harness.test_spec.test_spec import make_test_spec

    # 构造足够让 make_test_spec 通过的 instance dict
    spec_instance = {
        "instance_id": instance["instance_id"],
        "repo": instance.get("repo", ""),
        "base_commit": instance.get("base_commit", ""),
        "version": instance.get("version", ""),
        "test_patch": "",
        "PASS_TO_PASS": "[]",
        "FAIL_TO_PASS": "[]",
    }
    ns = namespace if namespace and namespace.lower() != "none" else None
    ts = make_test_spec(spec_instance, namespace=ns)
    return ts.instance_image_key


def test_spec_from_instance(instance: dict[str, Any], namespace: str | None = None) -> Any:
    """通过官方 make_test_spec 获取实例 TestSpec。"""
    sys.path.insert(0, "/root/SWE-bench")
    from swebench.harness.test_spec.test_spec import make_test_spec
    return make_test_spec(instance, namespace=namespace)


# ---------------------------------------------------------------------------
# Docker / 镜像
# ---------------------------------------------------------------------------


def pull_image(image: str) -> bool:
    """拉取 Docker 镜像，失败返回 False。"""
    timed_log(f"  拉取镜像: {image}")
    result = run_cmd(["docker", "pull", image], timeout=600, check=False)
    if result.returncode != 0:
        timed_log(f"  镜像拉取失败: {result.stderr.strip()[-300:]}")
        return False
    return True


def inspect_image(image: str) -> dict[str, Any] | None:
    """检查镜像信息。"""
    result = run_cmd(["docker", "image", "inspect", image], timeout=30, check=False)
    if result.returncode != 0:
        return None
    try:
        info = json.loads(result.stdout)[0]
        return {
            "image": image,
            "id": info.get("Id", ""),
            "size": info.get("Size", 0),
            "arch": info.get("Architecture", ""),
        }
    except (json.JSONDecodeError, IndexError, KeyError):
        return None


def image_exists(image: str) -> bool:
    result = run_cmd(["docker", "image", "inspect", image], timeout=10, check=False)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Worktree 管理
# ---------------------------------------------------------------------------


def repo_cache_path(config: BatchConfig, repo: str) -> Path:
    return Path(config.cache_root) / repo.split("/")[-1]


def create_worktree(config: BatchConfig, repo: str, base_commit: str, worktree: Path) -> bool:
    """从缓存仓库创建 detached worktree。"""
    cache = repo_cache_path(config, repo)
    if not cache.exists():
        timed_log(f"  缓存仓库不存在: {cache}")
        return False

    # 确保 base_commit 存在
    result = run_cmd(["git", "-C", str(cache), "cat-file", "-t", base_commit], check=False)
    if result.returncode != 0:
        timed_log(f"  base_commit 不在缓存中: {base_commit}")
        return False

    worktree.parent.mkdir(parents=True, exist_ok=True)
    result = run_cmd(
        ["git", "-C", str(cache), "worktree", "add", "--detach", str(worktree), base_commit],
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        timed_log(f"  worktree 创建失败: {result.stderr.strip()[-200:]}")
        return False
    return True


def remove_worktree(config: BatchConfig, repo: str, worktree: Path) -> None:
    """清理 worktree 并在缓存仓库中 prune。"""
    cache = repo_cache_path(config, repo)
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
    run_cmd(["git", "-C", str(cache), "worktree", "prune"], timeout=30, check=False)


def git_diff_worktree(worktree: Path) -> str:
    """生成二进制安全的 git diff（包含未跟踪文件，排除运行时产物目录）。"""
    # 将 .nano 移到工作区外的临时目录，避免 git add -N . 将其纳入 diff
    nano_dir = worktree / ".nano"
    nano_stash: Path | None = None
    if nano_dir.exists():
        nano_stash = Path(tempfile.mkdtemp(prefix="nano-stash-")) / ".nano"
        shutil.move(str(nano_dir), str(nano_stash))
    try:
        run_cmd(["git", "-C", str(worktree), "add", "-N", "."], check=False)
        result = run_cmd([
            "git", "-C", str(worktree), "diff", "HEAD", "--binary", "--no-ext-diff",
            "--", ".", ":(exclude)*.so", ":(exclude)*.pyc", ":(exclude)build/",
        ], check=False)
        run_cmd(["git", "-C", str(worktree), "reset"], check=False)
    finally:
        if nano_stash is not None and nano_stash.exists():
            shutil.move(str(nano_stash), str(nano_dir))
            # 清理临时目录
            stash_parent = nano_stash.parent
            if stash_parent.exists():
                shutil.rmtree(stash_parent, ignore_errors=True)
    return result.stdout


def worktree_is_dirty(worktree: Path) -> bool:
    result = run_cmd(["git", "-C", str(worktree), "status", "--porcelain"], check=False)
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Worktree 预热 — 从 Docker 镜像提取编译产物，使 Python import 可用
# ---------------------------------------------------------------------------


def _ensure_gitignore_patterns(worktree: Path) -> None:
    """确保 worktree 的 .gitignore 包含编译产物的排除模式。"""
    gitignore = worktree / ".gitignore"
    patterns = ["*.so", "*.pyc", "__pycache__/", "build/"]
    existing = set()
    if gitignore.exists():
        existing = {line.strip() for line in gitignore.read_text().splitlines()}
    missing = [p for p in patterns if p not in existing]
    if missing:
        with open(gitignore, "a") as f:
            f.write("\n# pico: warmup artifacts (auto-added)\n")
            for p in missing:
                f.write(p + "\n")


def warmup_worktree(image: str, worktree: Path, instance_id: str) -> bool:
    """从 Docker 镜像的 /testbed 复制预编译 C 扩展到 worktree。

    镜像里的 /testbed 有编译好的 .so 文件，但 worktree 是从 git 缓存创建
    的干净 checkout（只有源代码）。本函数在 worktree 被挂载覆盖 /testbed 之
    前，先把镜像里的编译产物复制到 worktree，使 ``import astropy`` 和
    ``pytest`` 在 Agent 容器内能正常工作。
    """
    temp_name = f"pico-warmup-{uuid.uuid4().hex[:8]}"
    timed_log(f"  [{instance_id}] 从镜像提取编译产物...")

    # 启动临时容器：worktree 挂载到 /worktree，镜像的 /testbed 保持不变
    result = run_cmd([
        "docker", "run", "--detach", "--name", temp_name,
        "--network", "none",
        "--volume", f"{worktree}:/worktree:rw",
        "--entrypoint", "sleep", image, "3600",
    ], timeout=30, check=False)

    if result.returncode != 0:
        timed_log(f"  [{instance_id}] 预热容器启动失败: {result.stderr.strip()[-200:]}")
        return False

    try:
        # 策略 1：从镜像 /testbed 复制所有 .so 文件到 worktree（秒级完成）
        timed_log(f"  [{instance_id}] 复制 .so 文件...")
        copy_cmd = (
            "cd /testbed && find . -name '*.so' -exec sh -c '"
            "  mkdir -p /worktree/$(dirname \"$1\") && cp \"$1\" \"/worktree/$1\""
            "' _ {} \\; && echo OK"
        )
        copy_result = run_cmd([
            "docker", "exec", temp_name, "bash", "-lc", copy_cmd,
        ], timeout=120, check=False)

        if copy_result.returncode != 0 or "OK" not in copy_result.stdout:
            timed_log(f"  [{instance_id}] .so 复制失败: {copy_result.stderr[:200]}")

            # 策略 2：尝试 build_ext --inplace 编译（不需要网络，但需要 gcc）
            timed_log(f"  [{instance_id}] 尝试编译 C 扩展...")
            build_cmd = (
                "cd /worktree && python setup.py build_ext --inplace 2>&1"
            )
            build_result = run_cmd([
                "docker", "exec", "--workdir", "/worktree", temp_name,
                "bash", "-lc", build_cmd,
            ], timeout=300, check=False)

            if build_result.returncode != 0:
                timed_log(f"  [{instance_id}] 编译失败: {build_result.stderr[:200]}")
                return False

        # 计数并验证
        count_result = run_cmd([
            "docker", "exec", temp_name, "bash", "-lc",
            "find /worktree -name '*.so' | wc -l",
        ], timeout=30, check=False)
        so_count = count_result.stdout.strip()
        timed_log(f"  [{instance_id}] 已就位 {so_count} 个 .so 文件")

        # 验证 import astropy
        verify = run_cmd([
            "docker", "exec", "--workdir", "/worktree", temp_name, "bash", "-lc",
            "python -c 'import astropy; print(astropy.__version__)' 2>&1",
        ], timeout=30, check=False)

        if verify.returncode == 0:
            timed_log(f"  [{instance_id}] ✓ warmup 成功 (astropy {verify.stdout.strip()})")
            return True
        else:
            # .so 文件已就位但 import 仍失败，可能是路径问题
            timed_log(f"  [{instance_id}] import 验证失败: {verify.stderr[:200]}")
            timed_log(f"  [{instance_id}] Agent 将使用纯代码阅读模式")
            return False

    finally:
        run_cmd(["docker", "rm", "-f", temp_name], timeout=10, check=False)


# ---------------------------------------------------------------------------
# 容器管理（Agent 工具执行用）
# ---------------------------------------------------------------------------


class AgentContainer:
    """管理单个实例的 Docker 容器生命周期。"""

    def __init__(self, instance_id: str, image: str, worktree: Path, batch_id: str) -> None:
        safe_id = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
        suffix = uuid.uuid4().hex[:8]
        self.name = f"pico-{batch_id}-{safe_id}-{suffix}"
        self.image = image
        self.worktree = worktree
        self._started = False

    def start(self) -> bool:
        """启动无网络容器，挂载 worktree 到 /testbed。"""
        if self._started:
            return True
        result = run_cmd(
            [
                "docker", "run", "--detach",
                "--name", self.name,
                "--network", "none",
                "--volume", f"{self.worktree}:/testbed:rw",
                "--workdir", "/testbed",
                "--entrypoint", "/bin/bash",
                self.image,
                "-lc", "while true; do sleep 3600; done",
            ],
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            timed_log(f"  容器启动失败: {result.stderr.strip()[-200:]}")
            return False
        self._started = True
        return True

    def exec_shell(self, command: str, timeout: int = 120) -> dict[str, Any]:
        """在容器内执行 shell 命令并返回结构化结果。"""
        if not self._started:
            return {"exit_code": -1, "stdout": "", "stderr": "容器未运行", "timed_out": False}
        started = time.monotonic()
        try:
            result = run_cmd(
                ["docker", "exec", "--workdir", "/testbed", self.name, "/bin/bash", "-lc", command],
                timeout=timeout,
                check=False,
            )
            elapsed = time.monotonic() - started
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed": round(elapsed, 3),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
            return {
                "exit_code": 124,
                "stdout": stdout,
                "stderr": stderr or "命令超时",
                "elapsed": round(elapsed, 3),
                "timed_out": True,
            }

    def stop(self) -> None:
        run_cmd(["docker", "rm", "-f", self.name], timeout=30, check=False)
        self._started = False


# ---------------------------------------------------------------------------
# Agent 构建与执行
# ---------------------------------------------------------------------------


def build_model_client(config: BatchConfig) -> Any:
    """复用项目 _build_model_client。"""
    sys.path.insert(0, str(PROJECT_ROOT))
    from nano.cli import _build_model_client

    ns = argparse.Namespace(
        provider=config.provider,
        model=config.model,
        base_url=None,
        temperature=config.temperature,
        openai_timeout=config.agent_timeout,
        secret_env_names=[],
    )
    return _build_model_client(ns)


def build_agent_prompt(instance: dict[str, Any]) -> str:
    """构建仅含公开信息的 prompt，引导 Agent 完成完整的 fix→test 循环。"""
    return textwrap.dedent(f"""\
        Fix {instance['instance_id']}.

        {instance['problem_statement']}

        # Workflow — follow these steps IN ORDER

        1. **Explore**: use read_file / list_files / grep_search to locate the
           relevant source files and understand the root cause.
        2. **Fix**: use patch_file to apply the minimal correct change. Never stop
           at "I found the bug" — you MUST edit the file and leave a non-empty patch.
        3. **Trace**: use grep_search to find every downstream consumer and comparison
           point of the changed value or API. Check each parser, serializer, caller,
           and comparison that may need the same correction.
        4. **Verify**: run targeted pytest tests after the patch. If you changed
           parsing or serialization, explicitly verify a write→read round trip and
           inspect the parsed result.

        ## Rules

        - Always call patch_file BEFORE run_shell verification.
        - patch_file uses path + old_text + new_text, where old_text is the exact
          existing text occurring once; it does not accept a unified diff. Example:
          path="astropy/io/ascii/qdp.py", old_text="old code", new_text="new code".
        - Use the smallest diff possible — one focused change per file.
        - Use repository-relative paths in every file tool and shell command. The
          shell starts at the repository root; never use a host path or `/testbed`.
        - For parser or serializer changes, the final verification command MUST
          construct the complete transformed input (for example, the entire input
          lowercased), read it through the changed public API, and assert the
          expected parsed values. A partial smoke test is not completion evidence.
          Do not claim the task is complete until this end-to-end check runs.
        - If an import or test fails because the benchmark environment lacks a
          dependency, record it as an environment limitation. Do not install
          dependencies and do not repeat the same failed environment check.
        - Do not access network resources.
        - Do not use SWE-bench gold patches or hidden test data.
        - Do not stop after explaining the solution; implement it in the workspace.
    """)


def _make_shell_executor(container: AgentContainer, log_file: Path):
    """创建带日志的 shell executor。"""
    def executor(command: str, timeout: int) -> str:
        import textwrap as _tw

        cmd_start = time.monotonic()
        timed_log(f"  [工具] 开始: {command[:80]}...")
        result = container.exec_shell(command, timeout=timeout)
        elapsed = time.monotonic() - cmd_start
        timed_log(f"  [工具] 结束: exit={result['exit_code']} ({elapsed:.1f}s)")

        # 写入日志
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"开始: {datetime.now().isoformat()}\n")
            f.write(f"命令: {command}\n")
            f.write(f"超时: {timeout}s  实际: {elapsed:.1f}s\n")
            f.write(f"退出码: {result['exit_code']}\n")
            f.write(f"stdout:\n{result['stdout']}\n")
            f.write(f"stderr:\n{result['stderr']}\n")

        return _tw.dedent(f"""\
            exit_code: {result['exit_code']}
            stdout:
            {result['stdout'].strip() or '(no output)'}
            stderr:
            {result['stderr'].strip() or '(no output)'}
        """).strip()
    return executor


async def run_agent(
    config: BatchConfig,
    instance: dict[str, Any],
    worktree: Path,
    container: AgentContainer,
    artifact_dir: Path,
) -> dict[str, Any]:
    """在隔离 worktree 中运行 Agent，返回执行结果。"""
    sys.path.insert(0, str(PROJECT_ROOT))

    from nano.runtime.runtime import AgentRuntime
    from nano.storage.run_store import RunStore
    from nano.storage.session_store import SessionStore
    from nano.workspace.context import WorkspaceContext

    log_file = artifact_dir / "agent.log"
    agent_started = time.monotonic()

    model_client = build_model_client(config)
    shell_executor = _make_shell_executor(container, log_file)
    prompt = build_agent_prompt(instance)

    # 保存 prompt
    (artifact_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    nano_dir = worktree / ".nano"
    nano_dir.mkdir(parents=True, exist_ok=True)

    runtime = AgentRuntime(
        model_client=model_client,
        workspace=WorkspaceContext.build(worktree, repo_root_override=worktree),
        session_store=SessionStore(nano_dir / "sessions"),
        run_store=RunStore(nano_dir / "runs"),
        approval_policy="auto",
        max_steps=config.max_steps,
        allowed_tools=list(ALLOWED_TOOLS),
        permissions_path=config.permissions_path,
        shell_executor=shell_executor,
        feature_flags={"memory": False, "relevant_memory": False},
    )

    error = ""
    tool_steps = 0
    final_answer = ""

    try:
        timed_log(f"  Agent 启动 (max_steps={config.max_steps}, timeout={config.agent_timeout}s)")
        await asyncio.wait_for(runtime.ask_async(prompt), timeout=config.agent_timeout)
    except asyncio.TimeoutError:
        error = f"Agent 超时 ({config.agent_timeout}s)"
        timed_log(f"  {error}")
    except Exception as exc:
        error = f"Agent 异常: {exc}"
        timed_log(f"  {error}")

    if runtime.current_task_state is not None:
        tool_steps = runtime.current_task_state.tool_steps
        final_answer = runtime.current_task_state.final_answer or ""

    agent_duration = time.monotonic() - agent_started

    # 保存最终回答
    (artifact_dir / "final-answer.md").write_text(final_answer or "(空)", encoding="utf-8")

    # 保存 trajectory
    if runtime.current_run_dir and runtime.current_run_dir.exists():
        try:
            shutil.copytree(runtime.current_run_dir, artifact_dir / "trajectory", dirs_exist_ok=True)
        except Exception:
            pass

    return {
        "error": error,
        "tool_steps": tool_steps,
        "agent_duration": round(agent_duration, 3),
        "final_answer_length": len(final_answer),
    }


# ---------------------------------------------------------------------------
# 官方评分
# ---------------------------------------------------------------------------


def generate_patch(worktree: Path, artifact_dir: Path) -> tuple[str, int]:
    """生成 patch 并返回 (patch_content, byte_size)。"""
    patch = git_diff_worktree(worktree)
    (artifact_dir / "model.patch").write_text(patch, encoding="utf-8")
    return patch, len(patch.encode("utf-8"))


def write_prediction_jsonl(artifact_dir: Path, instance_id: str, model_name: str, patch: str) -> Path:
    """写入官方格式 prediction.jsonl。"""
    path = artifact_dir / "prediction.jsonl"
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }
    path.write_text(json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_official_scorer(
    config: BatchConfig,
    instance_id: str,
    prediction_path: Path,
    run_id: str,
    artifact_dir: Path,
) -> tuple[bool | None, dict[str, Any]]:
    """调用官方 SWE-bench harness 评分。"""
    swebench_python = config.swebench_python
    if not Path(swebench_python).exists():
        swebench_python = sys.executable

    namespace_arg = config.namespace if config.namespace.lower() != "none" else ""

    cmd = [
        swebench_python, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", config.dataset_path,
        "--split", "test",
        "--predictions_path", str(prediction_path),
        "--max_workers", "1",
        "--instance_ids", instance_id,
        "--namespace", namespace_arg,
        "--run_id", run_id,
        "--timeout", str(config.evaluation_timeout),
        "--cache_level", "env",
    ]

    timed_log(f"  评分命令: {' '.join(cmd)}")
    harness_start = time.monotonic()

    try:
        result = run_cmd(
            cmd,
            cwd=str(artifact_dir),
            timeout=config.evaluation_timeout + 300,
            check=False,
            env={**os.environ, **SCORER_SAFE_ENV},
        )
    except subprocess.TimeoutExpired:
        return None, {"error": "harness 超时", "duration": config.evaluation_timeout + 300}

    harness_duration = time.monotonic() - harness_start

    (artifact_dir / "harness.stdout.log").write_text(result.stdout, encoding="utf-8")
    (artifact_dir / "harness.stderr.log").write_text(result.stderr, encoding="utf-8")

    # 解析报告（SWE-bench v4+ 使用 resolved_ids / unresolved_ids 格式）
    reports = sorted(artifact_dir.glob(f"*.{run_id}.json"))
    resolved: bool | None = None
    report_path: Path | None = None

    for report in reports:
        try:
            payload = read_json(report)
            # 格式 1: {instance_id: {resolved: true/false}}
            instance_result = payload.get(instance_id)
            if isinstance(instance_result, dict) and "resolved" in instance_result:
                resolved = bool(instance_result["resolved"])
                report_path = report
                break
            # 格式 2: {resolved_ids: [...], unresolved_ids: [...]}
            if "resolved_ids" in payload and "unresolved_ids" in payload:
                if instance_id in payload.get("resolved_ids", []):
                    resolved = True
                    report_path = report
                    break
                elif instance_id in payload.get("unresolved_ids", []):
                    resolved = False
                    report_path = report
                    break
                elif instance_id in payload.get("error_ids", []):
                    resolved = None  # 评分出错
                    report_path = report
                    break
        except Exception:
            continue

    if report_path:
        shutil.copy(report_path, artifact_dir / "official-report.json")

    info: dict[str, Any] = {
        "harness_duration": round(harness_duration, 3),
        "harness_exit_code": result.returncode,
        "report_found": report_path is not None,
    }

    if resolved is None and result.returncode != 0:
        info["error"] = f"harness exit={result.returncode}"

    return resolved, info


# ---------------------------------------------------------------------------
# 单个实例执行器
# ---------------------------------------------------------------------------


@dataclass
class InstanceResult:
    instance_id: str
    status: str = "pending"
    agent_duration: float = 0.0
    harness_duration: float = 0.0
    total_duration: float = 0.0
    patch_bytes: int = 0
    tool_steps: int = 0
    error: str = ""
    image_info: dict[str, Any] = field(default_factory=dict)
    artifact_dir: str = ""


def run_single_instance(
    config: BatchConfig,
    instance: dict[str, Any],
    batch_id: str,
    output_root: Path,
) -> InstanceResult:
    """执行单个实例的完整流水线。"""
    instance_id = instance["instance_id"]
    instance_dir = output_root / "instances" / instance_id
    result = InstanceResult(instance_id=instance_id, artifact_dir=str(instance_dir))

    # 锁
    lock = InstanceLock(instance_dir / ".lock")
    if not lock.acquire():
        timed_log(f"[{instance_id}] 已被其他进程锁定，跳过")
        result.status = "infra_error"
        result.error = "locked by another process"
        return result

    try:
        state_path = instance_dir / "state.json"
        state = {}
        if state_path.exists():
            state = read_json(state_path)

        current_state = state.get("status", "pending")

        # 跳过已完成状态
        if current_state in RESUMABLE_SKIP_STATES:
            timed_log(f"[{instance_id}] 已完成 ({current_state})，跳过")
            result.status = current_state
            if state_path.exists():
                prev = read_json(instance_dir / "result.json")
                result = InstanceResult(**prev)
            return result

        # 续跑逻辑
        if current_state == "patch_generated":
            timed_log(f"[{instance_id}] 已有 patch，继续评分")
            result = _evaluate_phase(config, instance, instance_dir, batch_id, result)
            return result

        if current_state == "evaluating":
            timed_log(f"[{instance_id}] 评分阶段，检查已有报告...")
            result = _evaluate_phase(config, instance, instance_dir, batch_id, result)
            return result

        instance_started = time.monotonic()

        # Phase 1: 镜像准备
        update_state(instance_dir, "preparing")
        image = instance_image_name(instance, config.namespace)
        metadata = {
            "instance_id": instance_id,
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "version": instance["version"],
            "image": image,
            "batch_id": batch_id,
            "started_at": now_iso(),
        }
        atomic_write_json(instance_dir / "metadata.json", metadata)

        if not image_exists(image):
            if config.dry_run:
                timed_log(f"[{instance_id}] DRY-RUN: 将拉取 {image}")
                result.status = "pending"
                return result
            if not pull_image(image):
                result.status = "infra_error"
                result.error = f"镜像拉取失败: {image}"
                update_state(instance_dir, "infra_error", result)
                return result

        img_info = inspect_image(image)
        if img_info:
            result.image_info = img_info
            timed_log(f"[{instance_id}] 镜像就绪: {img_info['id'][:19]} ({img_info['size']/1e9:.1f}GB)")

        update_state(instance_dir, "image_ready")

        if config.dry_run:
            timed_log(f"[{instance_id}] DRY-RUN: 将创建 worktree 并运行 Agent")
            result.status = "pending"
            return result

        # Phase 2: Worktree 创建
        worktree = instance_dir / "worktree"
        if not create_worktree(config, instance["repo"], instance["base_commit"], worktree):
            result.status = "infra_error"
            result.error = "worktree 创建失败"
            update_state(instance_dir, "infra_error", result)
            return result
        timed_log(f"[{instance_id}] worktree 就绪: {worktree}")

        # Phase 2.5: 预热 — 从镜像提取编译产物到 worktree
        warmup_worktree(image, worktree, instance_id)

        # Phase 3: 容器启动
        container = AgentContainer(instance_id, image, worktree, batch_id)
        if not container.start():
            result.status = "infra_error"
            result.error = "容器启动失败"
            update_state(instance_dir, "infra_error", result)
            return result
        timed_log(f"[{instance_id}] 容器启动: {container.name}")

        update_state(instance_dir, "worktree_ready")

        # Phase 4: Agent 运行
        update_state(instance_dir, "agent_running")
        agent_result = asyncio.run(
            run_agent(config, instance, worktree, container, instance_dir)
        )

        result.agent_duration = agent_result["agent_duration"]
        result.tool_steps = agent_result["tool_steps"]

        if agent_result["error"]:
            # Agent 执行中有超时或异常
            result.status = "agent_error"
            result.error = agent_result["error"]
            update_state(instance_dir, "agent_error", result)
            # 保留 worktree 用于排查
            container.stop()
            return result

        # Phase 5: Patch 生成
        patch, patch_bytes = generate_patch(worktree, instance_dir)
        result.patch_bytes = patch_bytes

        if not patch.strip():
            timed_log(f"[{instance_id}] patch 为空")
            result.status = "agent_error"
            result.error = "空 patch：Agent 未修改任何文件"
            update_state(instance_dir, "agent_error", result)
            container.stop()
            return result

        update_state(instance_dir, "patch_generated")
        timed_log(f"[{instance_id}] patch 生成: {patch_bytes} bytes")

        # 清理容器和 worktree（评分由官方 harness 自行创建容器）
        container.stop()
        remove_worktree(config, instance["repo"], worktree)

        # Phase 6: 官方评分
        result = _evaluate_phase(config, instance, instance_dir, batch_id, result)
        result.total_duration = round(time.monotonic() - instance_started, 3)
        return result

    finally:
        lock.release()


def _evaluate_phase(
    config: BatchConfig,
    instance: dict[str, Any],
    instance_dir: Path,
    batch_id: str,
    result: InstanceResult,
) -> InstanceResult:
    """评分阶段（可独立调用以支持续跑）。"""
    instance_id = instance["instance_id"]
    update_state(instance_dir, "evaluating")

    patch_path = instance_dir / "model.patch"
    if not patch_path.exists():
        result.status = "agent_error"
        result.error = "patch 文件不存在"
        update_state(instance_dir, "agent_error", result)
        return result

    patch = patch_path.read_text(encoding="utf-8")
    if not patch.strip():
        result.status = "agent_error"
        result.error = "空 patch"
        update_state(instance_dir, "agent_error", result)
        return result

    result.patch_bytes = len(patch.encode("utf-8"))

    if config.dry_run:
        timed_log(f"[{instance_id}] DRY-RUN: 将调用官方评分")
        return result

    model_name = f"pico/{config.provider}-{'default' if not config.model else config.model}"
    prediction_path = write_prediction_jsonl(instance_dir, instance_id, model_name, patch)

    run_id = f"{batch_id}-{instance_id}"
    resolved, info = run_official_scorer(config, instance_id, prediction_path, run_id, instance_dir)

    result.harness_duration = info.get("harness_duration", 0.0)

    if resolved is True:
        result.status = "resolved"
        timed_log(f"[{instance_id}] ★ RESOLVED")
    elif resolved is False:
        result.status = "unresolved"
        timed_log(f"[{instance_id}] ✗ UNRESOLVED")
    else:
        # 评分本身失败
        result.status = "infra_error"
        result.error = info.get("error", "评分异常")
        timed_log(f"[{instance_id}] ⚠ INFRA_ERROR: {result.error}")

    update_state(instance_dir, result.status, result)
    return result


def update_state(instance_dir: Path, status: str, result: InstanceResult | None = None) -> None:
    """原子写入状态文件。"""
    instance_dir.mkdir(parents=True, exist_ok=True)
    state = {"status": status, "updated_at": now_iso()}
    atomic_write_json(instance_dir / "state.json", state)
    if result is not None:
        result_data = {
            "instance_id": result.instance_id,
            "status": result.status,
            "agent_duration": result.agent_duration,
            "harness_duration": result.harness_duration,
            "total_duration": result.total_duration,
            "patch_bytes": result.patch_bytes,
            "tool_steps": result.tool_steps,
            "error": result.error,
            "artifact_dir": result.artifact_dir,
        }
        atomic_write_json(instance_dir / "result.json", result_data)


# ---------------------------------------------------------------------------
# 批量执行
# ---------------------------------------------------------------------------


def run_batch(config: BatchConfig) -> dict[str, Any]:
    """执行全部实例并返回汇总。支持 --max-parallel 并行调度。"""
    instances = load_instances(config)
    batch_id = f"five-astropy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_root = Path(config.output_root) / batch_id
    output_root.mkdir(parents=True, exist_ok=True)

    max_parallel = max(1, config.max_parallel)
    timed_log(f"批次 ID: {batch_id}")
    timed_log(f"实例数: {len(instances)}  并发数: {max_parallel}")
    timed_log(f"输出目录: {output_root}")

    batch_started = time.monotonic()
    batch_started_at = now_iso()

    # 保存批次配置
    batch_config = {
        "batch_id": batch_id,
        "provider": config.provider,
        "model": config.model or "default",
        "temperature": config.temperature,
        "max_steps": config.max_steps,
        "agent_timeout": config.agent_timeout,
        "evaluation_timeout": config.evaluation_timeout,
        "max_parallel": max_parallel,
        "instances": [inst["instance_id"] for inst in instances],
        "started_at": batch_started_at,
    }
    atomic_write_json(output_root / "batch.json", batch_config)

    # Phase 0: 串行预拉取所有镜像（避免 Docker 并发竞态）
    timed_log("\n--- 预拉取镜像 ---")
    for instance in instances:
        image = instance_image_name(instance, config.namespace)
        if not image_exists(image) and not config.dry_run:
            pull_image(image)
        timed_log(f"  镜像就绪: {instance['instance_id']}: {image}")

    # Phase 1: 并行执行各实例
    timed_log(f"\n--- 并行执行 (max_parallel={max_parallel}) ---")
    results: list[InstanceResult] = []
    results_lock = threading.Lock()

    def _run_one(idx: int, instance: dict[str, Any]) -> InstanceResult:
        timed_log(f"\n{'='*60}")
        timed_log(f"[{idx+1}/{len(instances)}] 开始: {instance['instance_id']}")
        timed_log(f"{'='*60}")

        result = run_single_instance(config, instance, batch_id, output_root)

        timed_log(f"[{instance['instance_id']}] 最终状态: {result.status}")

        # 线程安全地写入中间汇总
        with results_lock:
            results.append(result)
            _write_interim_summary(output_root, batch_id, list(results), batch_started_at)

        return result

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(_run_one, i, instance): instance
            for i, instance in enumerate(instances)
        }
        for future in as_completed(futures):
            instance = futures[future]
            try:
                future.result()
            except Exception as exc:
                timed_log(f"[{instance['instance_id']}] 执行异常: {exc}")

    # 按原始顺序排列结果
    instance_order = {inst["instance_id"]: i for i, inst in enumerate(instances)}
    results.sort(key=lambda r: instance_order.get(r.instance_id, 999))

    batch_duration = time.monotonic() - batch_started
    return _write_final_summary(output_root, batch_id, results, batch_started_at, batch_duration)


def _write_interim_summary(output_root: Path, batch_id: str, results: list[InstanceResult], started_at: str) -> None:
    """写入中间汇总（每完成一个实例后调用）。"""
    summary = _build_summary(batch_id, results, started_at)
    atomic_write_json(output_root / "summary.json", summary)


def _write_final_summary(
    output_root: Path, batch_id: str, results: list[InstanceResult], started_at: str, batch_duration: float
) -> dict[str, Any]:
    """写入最终汇总。"""
    summary = _build_summary(batch_id, results, started_at)
    summary["finished_at"] = now_iso()
    summary["total_duration_seconds"] = round(batch_duration, 1)
    atomic_write_json(output_root / "summary.json", summary)

    # 生成 Markdown
    md = _build_summary_md(summary)
    (output_root / "summary.md").write_text(md, encoding="utf-8")

    # 屏幕输出
    print(md)
    return summary


def _build_summary(batch_id: str, results: list[InstanceResult], started_at: str) -> dict[str, Any]:
    resolved = sum(1 for r in results if r.status == "resolved")
    unresolved = sum(1 for r in results if r.status == "unresolved")
    agent_error = sum(1 for r in results if r.status == "agent_error")
    infra_error = sum(1 for r in results if r.status == "infra_error")
    completed = resolved + unresolved
    completion_rate = resolved / completed if completed > 0 else 0.0

    return {
        "batch_id": batch_id,
        "started_at": started_at,
        "instance_count": len(results),
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "agent_error_count": agent_error,
        "infra_error_count": infra_error,
        "completion_rate": round(completion_rate, 4),
        "instances": [
            {
                "instance_id": r.instance_id,
                "status": r.status,
                "agent_duration": r.agent_duration,
                "harness_duration": r.harness_duration,
                "total_duration": r.total_duration,
                "patch_bytes": r.patch_bytes,
                "tool_steps": r.tool_steps,
                "error": r.error,
            }
            for r in results
        ],
    }


def _build_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# SWE-bench Lite 五实例评测汇总",
        "",
        f"**批次 ID:** {summary['batch_id']}",
        f"**开始时间:** {summary['started_at']}",
        f"**完成时间:** {summary.get('finished_at', '进行中...')}",
        f"**总耗时:** {summary.get('total_duration_seconds', 'N/A')}s",
        "",
        "## 结果概览",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| Resolved | {summary['resolved_count']} |",
        f"| Unresolved | {summary['unresolved_count']} |",
        f"| Agent Error | {summary['agent_error_count']} |",
        f"| Infra Error | {summary['infra_error_count']} |",
        f"| Completion Rate | {summary['completion_rate']:.2%} |",
        "",
        "## 实例详情",
        "",
        "| Instance | Status | Agent time | Eval time | Patch | Tool Steps | Error |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for inst in summary["instances"]:
        status_icon = {
            "resolved": "★",
            "unresolved": "✗",
            "agent_error": "⚠",
            "infra_error": "✘",
            "pending": "○",
        }.get(inst["status"], "?")
        lines.append(
            f"| {inst['instance_id']} | {status_icon} {inst['status']} | "
            f"{inst['agent_duration']:.0f}s | {inst['harness_duration']:.0f}s | "
            f"{inst['patch_bytes']}B | {inst['tool_steps']} | "
            f"{inst['error'][:80]} |"
        )

    lines.extend([
        "",
        "## 产物",
        "",
        f"- 运行根目录: `/root/swebench-runs/{summary['batch_id']}`",
        f"- 汇总 JSON: `/root/swebench-runs/{summary['batch_id']}/summary.json`",
        f"- 汇总报告: `/root/swebench-runs/{summary['batch_id']}/summary.md`",
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Gold Smoke Test
# ---------------------------------------------------------------------------


def gold_smoke_test(config: BatchConfig, instance_id: str) -> bool:
    """用 gold patch 验证官方评分链路正常。"""
    timed_log(f"\n{'='*60}")
    timed_log(f"Gold smoke test: {instance_id}")
    timed_log(f"{'='*60}")

    data = read_json(Path(config.dataset_path))
    instance = None
    gold_patch = None
    for item in data:
        if item.get("instance_id") == instance_id:
            instance = {
                "instance_id": str(item["instance_id"]),
                "repo": str(item["repo"]),
                "base_commit": str(item["base_commit"]),
                "version": str(item.get("version", "")),
                "problem_statement": str(item.get("problem_statement", "")),
            }
            gold_patch = str(item.get("patch", ""))
            break

    if not instance or not gold_patch:
        timed_log("  找不到 instance 或 gold patch")
        return False

    smoke_dir = Path(config.output_root) / "_gold_smoke" / instance_id
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # 写入 gold prediction
    pred_path = smoke_dir / "prediction.jsonl"
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": "gold",
        "model_patch": gold_patch,
    }
    pred_path.write_text(json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8")

    run_id = f"gold-smoke-{uuid.uuid4().hex[:8]}"

    # 确保镜像存在
    image = instance_image_name(instance, config.namespace)
    if not image_exists(image):
        if not pull_image(image):
            timed_log("  镜像拉取失败")
            return False

    resolved, info = run_official_scorer(config, instance_id, pred_path, run_id, smoke_dir)

    if resolved is True:
        timed_log(f"  ★ Gold smoke 通过: {instance_id} resolved")
        return True
    else:
        timed_log(f"  ✗ Gold smoke 失败: resolved={resolved}, info={info}")
        return False


# ---------------------------------------------------------------------------
# 环境检查
# ---------------------------------------------------------------------------


def check_environment(config: BatchConfig) -> bool:
    """检查运行环境是否就绪。"""
    checks = []

    # Docker
    docker_check = run_cmd(["docker", "info"], timeout=10, check=False)
    checks.append(("Docker", docker_check.returncode == 0))

    # 磁盘
    disk = shutil.disk_usage("/")
    checks.append((f"磁盘可用 ({disk.free/1e9:.1f}GB)", disk.free > 2e9))

    # .env
    env_exists = Path(config.env_file).exists()
    checks.append((".env 文件", env_exists))

    # 数据集
    data_exists = Path(config.dataset_path).exists()
    checks.append(("数据集", data_exists))

    # Git 缓存（数据集可能不存在时跳过）
    try:
        instances = load_instances(config)
        for instance in instances:
            cache = repo_cache_path(config, instance["repo"])
            if cache.exists():
                checks.append((f"缓存 {instance['repo']}", True))
                break
        else:
            checks.append(("缓存仓库", False))
    except (FileNotFoundError, ValueError):
        checks.append(("数据集加载", False))

    # SWE-bench
    swebench_exists = Path(config.swebench_path).exists()
    checks.append(("SWE-bench 仓库", swebench_exists))

    all_ok = True
    for name, ok in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}")
        if not ok:
            all_ok = False

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> BatchConfig:
    parser = argparse.ArgumentParser(description="SWE-bench Lite 批量运行器")
    parser.add_argument("--instances", nargs="*", help="实例 ID 列表")
    parser.add_argument("--resume", action="store_true", help="续跑模式")
    parser.add_argument("--max-parallel", type=int, default=1, help="最大并发数")
    parser.add_argument("--agent-timeout", type=int, default=2700, help="Agent 超时(秒)")
    parser.add_argument("--evaluation-timeout", type=int, default=1800, help="评分超时(秒)")
    parser.add_argument("--max-steps", type=int, default=40, help="Agent 最大步数")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="产物根目录")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH, help="数据集路径")
    parser.add_argument("--provider", type=str, default="deepseek", help="模型提供商")
    parser.add_argument("--model", type=str, default=None, help="模型名")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度")
    parser.add_argument("--dry-run", action="store_true", help="预演模式")
    parser.add_argument("--namespace", type=str, default="swebench", help="镜像 namespace")
    parser.add_argument("--gold-smoke", type=str, default=None, help="先执行 gold smoke 测试指定实例")

    args = parser.parse_args(argv)
    return BatchConfig(
        instance_ids=args.instances or FIVE_INSTANCE_IDS,
        resume=args.resume,
        max_parallel=args.max_parallel,
        agent_timeout=args.agent_timeout,
        evaluation_timeout=args.evaluation_timeout,
        max_steps=args.max_steps,
        output_root=args.output_root,
        dataset_path=args.dataset,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        dry_run=args.dry_run,
        namespace=args.namespace,
        gold_smoke=args.gold_smoke,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)

    timed_log("=== SWE-bench Lite 批量运行器 ===")
    timed_log(f"Provider: {config.provider}, Temperature: {config.temperature}")
    timed_log(f"Max steps: {config.max_steps}, Agent timeout: {config.agent_timeout}s")
    timed_log(f"Instances: {config.instance_ids}")

    # 环境检查
    if not check_environment(config):
        timed_log("环境检查未通过")
        return 1

    # Gold smoke test（可选）
    if config.gold_smoke:
        if not gold_smoke_test(config, config.gold_smoke):
            timed_log("Gold smoke 失败，请检查环境后重试")
            return 1
        timed_log("Gold smoke 通过！")

    # 执行
    summary = run_batch(config)

    # 最终输出
    resolved = summary["resolved_count"]
    unresolved = summary["unresolved_count"]
    agent_err = summary["agent_error_count"]
    infra_err = summary["infra_error_count"]

    print(f"\n{'='*60}")
    print("五实例评测完成")
    print(f"Resolved: {resolved}")
    print(f"Unresolved: {unresolved}")
    print(f"Agent error: {agent_err}")
    print(f"Infra error: {infra_err}")
    print(f"Completion rate: {summary['completion_rate']:.2%}")
    print(f"Results: {config.output_root}/{summary['batch_id']}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
