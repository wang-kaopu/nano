"""SWE-bench Lite 题目的 Docker 隔离生命周期。"""

from __future__ import annotations

import re
import subprocess
import textwrap
import uuid
from pathlib import Path


class SwebenchSandbox:
    """从官方实例镜像导出基准工作区，并承载模型可用的隔离 shell。"""

    def __init__(self, instance_id: str, image: str, worktree: Path, artifacts: Path) -> None:
        """保存单题容器、挂载工作区和日志工件位置。"""
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+__[a-zA-Z0-9_.-]+-\d+", instance_id):
            raise ValueError("invalid SWE-bench instance_id")
        self.instance_id = instance_id
        self.image = image
        self.worktree = worktree.resolve()
        self.artifacts = artifacts.resolve()
        suffix = uuid.uuid4().hex[:10]
        prefix = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
        self.source_name = f"pico-swe-source-{prefix}-{suffix}"
        self.container_name = f"pico-swe-agent-{prefix}-{suffix}"
        self._started = False

    def _run(self, args: list[str], *, timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
        """执行仅供评测控制面使用的 Docker 命令。"""
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "docker command failed"
            raise RuntimeError(detail)
        return result

    def prepare_worktree(self, base_commit: str) -> Path:
        """从不可变实例镜像复制基准仓库，且在模型启动前复位到声明提交。"""
        if self.worktree.exists():
            raise ValueError(f"worktree already exists: {self.worktree}")
        self.worktree.mkdir(parents=True)
        try:
            self._run(["docker", "create", "--name", self.source_name, self.image, "tail", "-f", "/dev/null"])
            self._run(["docker", "cp", f"{self.source_name}:/testbed/.", str(self.worktree)])
        finally:
            self._run(["docker", "rm", "-f", self.source_name], check=False)
        self._run(["git", "-C", str(self.worktree), "reset", "--hard", base_commit])
        self._run(["git", "-C", str(self.worktree), "clean", "-fdx"])
        if self._run(["git", "-C", str(self.worktree), "status", "--porcelain"]).stdout.strip():
            raise RuntimeError("SWE-bench worktree is not clean after reset")
        return self.worktree

    def start(self) -> None:
        """启动无网络的 agent 容器，并只挂载本题可修改工作区。"""
        self._run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                self.container_name,
                "--network",
                "none",
                "--volume",
                f"{self.worktree}:/testbed:rw",
                "--workdir",
                "/testbed",
                "--entrypoint",
                "/bin/bash",
                self.image,
                "-lc",
                "while true; do sleep 3600; done",
            ]
        )
        self._started = True

    def execute_shell(self, command: str, timeout: int) -> str:
        """通过 docker exec 运行已获权限的模型 shell 命令并保留三段式输出。"""
        if not self._started:
            raise RuntimeError("SWE-bench sandbox is not running")
        try:
            result = self._run(
                ["docker", "exec", "--workdir", "/testbed", self.container_name, "/bin/bash", "-lc", command],
                timeout=timeout,
                check=False,
            )
            return textwrap.dedent(
                f"""\
                exit_code: {result.returncode}
                stdout:
                {result.stdout.strip() or "(no output)"}
                stderr:
                {result.stderr.strip() or "(no output)"}
                """
            ).strip()
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
            return f"exit_code: 124\nstdout:\n{stdout or '(no output)'}\nstderr:\n{stderr or 'command timed out'}"

    def collect_logs(self) -> Path:
        """保存模型容器 stdout/stderr，供每次运行审计。"""
        self.artifacts.mkdir(parents=True, exist_ok=True)
        result = self._run(["docker", "logs", self.container_name], check=False)
        path = self.artifacts / "container.log"
        path.write_text((result.stdout + result.stderr).strip() + "\n", encoding="utf-8")
        return path

    def cleanup(self) -> None:
        """无条件移除本题容器，不处理任何其他 Docker 资源。"""
        self._run(["docker", "rm", "-f", self.container_name], check=False)
        self._started = False

    def __enter__(self) -> "SwebenchSandbox":
        """允许评测器使用上下文管理器保证容器清理。"""
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        """退出评测范围时清理题目容器。"""
        del exc_type, exc, traceback
        self.cleanup()
