"""工作区快照工具。

这个模块负责在 agent 按需读文件之前，先给它一份便宜的“仓库第一印象”。
这份快照刻意保持小而稳定：主要包含 Git 事实和少量白名单项目文档。
"""

import subprocess
import hashlib
import json
import shlex
from pathlib import Path
from typing import Sequence

from nano.utils import clip

MAX_TOOL_OUTPUT = 5000
MAX_HISTORY = 12000
MAX_INCLUDE_DEPTH = 5
# 这些文件最可能直接影响 agent 的行动方式。
# 我们不会预加载整个仓库，只会先给模型一小份“导航包”。
DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
IGNORED_PATH_NAMES = {".git", ".nano", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


class WorkspaceContext:
    """描述用于构建提示词前缀的稳定工作区快照。"""

    def __init__(
        self,
        cwd: str,
        repo_root: str,
        branch: str,
        default_branch: str,
        status: str,
        recent_commits: Sequence[str],
        project_docs: dict[str, str],
    ) -> None:
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @staticmethod
    def _include_target(line: str) -> str | None:
        """解析一行独立的 @include 指令并返回其路径。"""
        try:
            parts = shlex.split(line.strip())
        except ValueError:
            return None
        if len(parts) == 2 and parts[0] == "@include":
            return parts[1]
        return None

    @classmethod
    def _expand_project_document(
        cls,
        path: Path,
        repo_root: Path,
        depth: int = 0,
        ancestors: tuple[Path, ...] = (),
    ) -> str:
        """展开仓库内项目文档的 @include 指令，最多递归五层。"""
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(repo_root)
        except (OSError, ValueError):
            return f"[include skipped: path is outside the repository: {path}]"
        if not resolved_path.is_file() or any(part in IGNORED_PATH_NAMES for part in resolved_path.relative_to(repo_root).parts):
            return f"[include skipped: unavailable file: {path}]"
        if resolved_path in ancestors:
            return f"[include skipped: cycle detected: {resolved_path.relative_to(repo_root)}]"

        lines = []
        for line in resolved_path.read_text(encoding="utf-8", errors="replace").splitlines():
            include_path = cls._include_target(line)
            if include_path is None:
                lines.append(line)
                continue
            if depth >= MAX_INCLUDE_DEPTH:
                lines.append(f"[include skipped: maximum depth {MAX_INCLUDE_DEPTH} reached: {include_path}]")
                continue
            child_path = (resolved_path.parent / include_path).resolve()
            try:
                child_relative_path = child_path.relative_to(repo_root)
            except ValueError:
                lines.append(f"[include skipped: path is outside the repository: {include_path}]")
                continue
            child_text = cls._expand_project_document(child_path, repo_root, depth + 1, (*ancestors, resolved_path))
            lines.extend((f"[included from {child_relative_path}]", child_text, f"[end included {child_relative_path}]"))
        return "\n".join(lines)

    @classmethod
    def build(cls, cwd: str | Path, repo_root_override: str | Path | None = None) -> "WorkspaceContext":
        cwd = Path(cwd).resolve()

        def git(args: Sequence[str], fallback: str = "") -> str:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = (
            Path(repo_root_override).resolve()
            if repo_root_override is not None
            else Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        )
        docs = {}
        # 同时扫描 repo_root 和 cwd，这样在子目录启动时也能看到本地文档；
        # 但用相对路径做 key，避免同一份文档被重复收集。
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                docs[key] = clip(cls._expand_project_document(path, repo_root), 1200)

        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                lambda branch: branch[len("origin/") :] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self) -> str:
        """渲染动态 system context 所需的 Git 状态和项目指令。"""
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return "\n".join(
            (
                "Git context:",
                f"- repository root: {self.repo_root}",
                f"- working directory: {self.cwd}",
                f"- branch: {self.branch}",
                f"- default_branch: {self.default_branch}",
                "- working tree status:",
                self.status,
                "- recent_commits:",
                commits,
                "Project instructions:",
                docs,
            )
        )

    def fingerprint(self) -> str:
        # 这个指纹用来判断仓库状态是否发生了足够大的变化，
        # 从而决定是否需要重建缓存中的 prompt prefix。
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
