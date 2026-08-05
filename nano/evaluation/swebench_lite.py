"""独立运行 SWE-bench Lite 真实模型 pilot，并交由官方 harness 评分。"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from nano.runtime.runtime import AgentRuntime
from nano.storage.run_store import RunStore
from nano.storage.session_store import SessionStore
from nano.workspace.context import WorkspaceContext

from .swebench_sandbox import SwebenchSandbox


PILOT_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "benchmarks/swebench_lite_pilot.json"
DEFAULT_PERMISSIONS_PATH = PROJECT_ROOT / "benchmarks/swebench_permissions.json"
PUBLIC_INSTANCE_FIELDS = frozenset({"instance_id", "repo", "base_commit", "problem_statement"})
REQUIRED_TASK_FIELDS = frozenset(
    {
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "instance_image",
        "allowed_tools",
        "step_budget",
        "time_budget_seconds",
    }
)
PILOT_TOOLS = ("read_file", "list_files", "search", "grep_search", "patch_file", "run_shell")


def load_pilot_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """加载并校验不包含私有评测字段的 pilot manifest。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) - {"schema_version", "dataset", "description", "tasks"}:
        raise ValueError("invalid SWE-bench pilot manifest shape")
    if payload.get("schema_version") != PILOT_SCHEMA_VERSION:
        raise ValueError("unsupported SWE-bench pilot schema_version")
    if not isinstance(payload.get("dataset"), str) or not payload["dataset"].strip():
        raise ValueError("pilot dataset must be a non-empty string")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks or len(tasks) > 3:
        raise ValueError("pilot manifest must contain one to three locked tasks")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict) or set(task) != REQUIRED_TASK_FIELDS:
            raise ValueError("pilot task fields do not match the public schema")
        instance_id = str(task["instance_id"]).strip()
        if not instance_id or instance_id in seen:
            raise ValueError(f"duplicate or empty pilot instance_id: {instance_id}")
        if tuple(task["allowed_tools"]) != PILOT_TOOLS:
            raise ValueError(f"pilot task {instance_id} must use the locked tool list")
        if int(task["step_budget"]) < 1 or int(task["time_budget_seconds"]) < 1:
            raise ValueError(f"pilot task {instance_id} has an invalid budget")
        if not all(str(task[field]).strip() for field in ("repo", "base_commit", "problem_statement", "instance_image")):
            raise ValueError(f"pilot task {instance_id} has an empty public field")
        seen.add(instance_id)
        normalized.append({**task, "instance_id": instance_id, "step_budget": int(task["step_budget"]), "time_budget_seconds": int(task["time_budget_seconds"])})
    return {**payload, "tasks": normalized}


def load_public_instances(dataset_name: str, instance_ids: Iterable[str], swebench_path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """通过官方 SWE-bench 数据加载器只读取公开任务字段。"""
    if swebench_path is not None:
        source = str(Path(swebench_path).resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
    try:
        from swebench.harness.utils import load_swebench_dataset  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError("SWE-bench is required; run `uv sync --group benchmark`") from exc
    requested = set(instance_ids)
    dataset = load_swebench_dataset(dataset_name, "test")
    rows: dict[str, dict[str, str]] = {}
    for item in dataset:
        instance_id = str(item.get("instance_id", ""))
        if instance_id in requested:
            rows[instance_id] = {field: str(item[field]) for field in PUBLIC_INSTANCE_FIELDS}
    missing = requested - set(rows)
    if missing:
        raise ValueError(f"SWE-bench instances not found: {', '.join(sorted(missing))}")
    return rows


def build_task_prompt(task: dict[str, Any]) -> str:
    """构建仅包含公开问题描述和代码修改请求的模型输入。"""
    return textwrap.dedent(f"""\
        Repository: {task['repo']}
        Base commit: {task['base_commit']}

        {task['problem_statement']}

        Implement a minimal fix in the current workspace and follow this order:

        1. Explore with read_file, list_files, and search using repository-relative paths.
        2. Patch the source with patch_file before running verification. Supply path,
           exact old_text occurring once, and replacement new_text; patch_file does
           does not accept a unified diff. Leave a non-empty patch.
        3. Trace every downstream consumer and comparison point of the changed value or API.
        4. Run targeted tests. For parsing or serialization changes, perform a write→read round trip: the final verification command must construct the complete transformed input (for example, the entire input lowercased), read it through the changed public API, assert the parsed values, and inspect the result. A partial smoke test is not completion evidence.

        The shell already runs at the repository root. Use repository-relative paths only; never use host paths or /testbed.
        If an import or test fails because the environment lacks a dependency, record that as an environment limitation; do not install dependencies or repeat the same failed environment check.
        Do not access network resources or SWE-bench private test data. Do not stop after explaining the fix: edit the workspace.
    """)


def write_predictions(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    """将模型 git diff 写成官方 SWE-bench 接受的 predictions JSON 数组。"""
    predictions = []
    for row in rows:
        instance_id = str(row["instance_id"])
        model_name = str(row["model_name_or_path"])
        patch = str(row.get("model_patch", ""))
        if not instance_id or not model_name:
            raise ValueError("prediction requires instance_id and model_name_or_path")
        predictions.append({"instance_id": instance_id, "model_name_or_path": model_name, "model_patch": patch})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def scorer_command(
    swebench_python: str,
    dataset: str,
    predictions_path: Path,
    instance_id: str,
    run_id: str,
    timeout: int,
) -> list[str]:
    """构建调用官方 swebench.harness.run_evaluation 的确定性命令。"""
    return [
        swebench_python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        "1",
        "--instance_ids",
        instance_id,
        "--namespace",
        "",
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
    ]


def _model_client(provider: str, model: str | None, temperature: float, timeout: int):
    """复用 Pico 的真实 provider 装配逻辑，拒绝 fake client。"""
    from nano.cli import _build_model_client

    return _build_model_client(
        argparse.Namespace(provider=provider, model=model, base_url=None, temperature=temperature, openai_timeout=timeout, secret_env_names=[])
    )


def _swebench_python(swebench_path: Path) -> str:
    """优先使用指定官方仓库自己的虚拟环境执行评分。"""
    candidate = swebench_path / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _read_scorer_report(artifact_dir: Path, run_id: str, instance_id: str) -> tuple[bool | None, Path | None]:
    """读取官方评分报告中的单题 resolved 状态，而不解析测试输出。"""
    reports = sorted(artifact_dir.glob(f"*.{run_id}.json"))
    for report in reports:
        payload = json.loads(report.read_text(encoding="utf-8"))
        result = payload.get(instance_id)
        if isinstance(result, dict) and "resolved" in result:
            return bool(result["resolved"]), report
    return None, None


def _git_diff(worktree: Path, base_commit: str) -> str:
    """采集模型在干净基准提交上留下的唯一最终 diff。"""
    result = subprocess.run(
        ["git", "-C", str(worktree), "diff", "--binary", "--no-ext-diff", base_commit],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _copy_runtime_artifacts(worktree: Path, artifact_dir: Path) -> Path | None:
    """将模型工具轨迹和 prompt 运行记录移出临时工作区。"""
    source = worktree / ".nano"
    if not source.exists():
        return None
    target = artifact_dir / "agent-runtime"
    shutil.copytree(source, target)
    return target


def run_pilot(
    *,
    provider: str,
    model: str | None,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    swebench_path: str | Path,
    output_dir: str | Path,
    instance_ids: Iterable[str] | None = None,
    temperature: float = 0.0,
    max_steps: int | None = None,
    timeout: int = 900,
    repeats: int = 3,
) -> dict[str, Any]:
    """执行独立 SWE-bench Lite pilot，返回每题每次运行与官方评分汇总。"""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    manifest = load_pilot_manifest(manifest_path)
    selected = set(instance_ids or [task["instance_id"] for task in manifest["tasks"]])
    tasks = [task for task in manifest["tasks"] if task["instance_id"] in selected]
    if not tasks or selected - {task["instance_id"] for task in tasks}:
        raise ValueError("instance filter must select manifest tasks")
    swebench_root = Path(swebench_path).resolve()
    public_rows = load_public_instances(manifest["dataset"], (task["instance_id"] for task in tasks), swebench_root)
    for task in tasks:
        public = public_rows[task["instance_id"]]
        if any(task[field] != public[field] for field in ("repo", "base_commit")):
            raise ValueError(f"manifest public metadata differs from SWE-bench: {task['instance_id']}")
        task["problem_statement"] = public["problem_statement"]
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    client = _model_client(provider, model, temperature, timeout)
    run_id = f"pico-swe-lite-{uuid.uuid4().hex[:10]}"
    rows: list[dict[str, Any]] = []
    for attempt in range(1, repeats + 1):
        for task in tasks:
            rows.append(
                _run_task(
                    task=task,
                    dataset=manifest["dataset"],
                    provider=provider,
                    model_name=client.model,
                    model_client=client,
                    artifact_dir=root / task["instance_id"] / f"attempt-{attempt}",
                    swebench_root=swebench_root,
                    run_id=f"{run_id}-{task['instance_id']}-r{attempt}",
                    temperature=temperature,
                    max_steps=max_steps,
                    timeout=min(timeout, task["time_budget_seconds"]),
                )
            )
    resolved = sum(row["resolved"] is True for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    summary = {
        "run_id": run_id,
        "dataset": manifest["dataset"],
        "model": {"provider": provider, "name": client.model, "temperature": temperature, "repeats": repeats},
        "total_runs": len(rows),
        "resolved": resolved,
        "unresolved": sum(row["resolved"] is False for row in rows),
        "errors": errors,
        "resolved_rate": resolved / len(rows) if rows else 0.0,
        "rows": rows,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_task(
    *,
    task: dict[str, Any],
    dataset: str,
    provider: str,
    model_name: str,
    model_client: Any,
    artifact_dir: Path,
    swebench_root: Path,
    run_id: str,
    temperature: float,
    max_steps: int | None,
    timeout: int,
) -> dict[str, Any]:
    """在一套新 worktree、新容器中完成模型运行、预测写入和官方评分。"""
    started = time.monotonic()
    artifact_dir.mkdir(parents=True)
    prompt = build_task_prompt(task)
    (artifact_dir / "prompt.json").write_text(
        json.dumps({"prompt": prompt, "provider": provider, "model": model_name, "temperature": temperature, "max_steps": max_steps or task["step_budget"], "timeout": timeout}, indent=2) + "\n",
        encoding="utf-8",
    )
    worktree = artifact_dir / "worktree"
    patch_path = artifact_dir / "final.patch"
    prediction_path = artifact_dir / "predictions.json"
    runtime_artifacts: Path | None = None
    container_log: Path | None = None
    scorer_report: Path | None = None
    resolved: bool | None = None
    error = ""
    tool_steps = 0
    try:
        with SwebenchSandbox(task["instance_id"], task["instance_image"], worktree, artifact_dir) as sandbox:
            sandbox.prepare_worktree(task["base_commit"])
            sandbox.start()
            runtime = AgentRuntime(
                model_client=model_client,
                workspace=WorkspaceContext.build(worktree, repo_root_override=worktree),
                session_store=SessionStore(worktree / ".nano" / "sessions"),
                run_store=RunStore(worktree / ".nano" / "runs"),
                approval_policy="auto",
                max_steps=max_steps or task["step_budget"],
                allowed_tools=task["allowed_tools"],
                permissions_path=DEFAULT_PERMISSIONS_PATH,
                shell_executor=sandbox.execute_shell,
                feature_flags={"memory": False, "relevant_memory": False},
            )
            try:
                asyncio.run(asyncio.wait_for(runtime.ask_async(prompt), timeout=timeout))
            except asyncio.TimeoutError:
                error = f"agent time budget exceeded after {timeout} seconds"
            tool_steps = runtime.current_task_state.tool_steps if runtime.current_task_state is not None else 0
            patch_path.write_text(_git_diff(worktree, task["base_commit"]), encoding="utf-8")
            runtime_artifacts = _copy_runtime_artifacts(worktree, artifact_dir)
            container_log = sandbox.collect_logs()
        write_predictions(prediction_path, [{"instance_id": task["instance_id"], "model_name_or_path": model_name, "model_patch": patch_path.read_text(encoding="utf-8")}])
        command = scorer_command(_swebench_python(swebench_root), dataset, prediction_path, task["instance_id"], run_id, timeout)
        scorer = subprocess.run(command, cwd=artifact_dir, capture_output=True, text=True, timeout=timeout + 120, env={**dict(__import__("os").environ), "PYTHONPATH": str(swebench_root)})
        (artifact_dir / "scorer.stdout.log").write_text(scorer.stdout, encoding="utf-8")
        (artifact_dir / "scorer.stderr.log").write_text(scorer.stderr, encoding="utf-8")
        resolved, scorer_report = _read_scorer_report(artifact_dir, run_id, task["instance_id"])
        if scorer.returncode != 0 or resolved is None:
            error = error or f"official scorer failed with exit code {scorer.returncode}"
    except Exception as exc:
        error = str(exc)
    duration = time.monotonic() - started
    return {
        "instance_id": task["instance_id"],
        "status": "error" if error else "completed",
        "resolved": resolved,
        "error": error,
        "duration_seconds": round(duration, 3),
        "tool_steps": tool_steps,
        "artifact_dir": str(artifact_dir),
        "prompt_configuration": str(artifact_dir / "prompt.json"),
        "final_patch": str(patch_path),
        "predictions": str(prediction_path),
        "runtime_artifacts": str(runtime_artifacts) if runtime_artifacts else "",
        "container_log": str(container_log) if container_log else "",
        "swebench_report": str(scorer_report) if scorer_report else "",
        "scorer_stdout": str(artifact_dir / "scorer.stdout.log"),
        "scorer_stderr": str(artifact_dir / "scorer.stderr.log"),
    }
