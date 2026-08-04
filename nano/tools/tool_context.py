"""Narrow context passed from runtime into tool functions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from nano.permissions import ProjectPermissions


@dataclass
class FileReadCoverage:
    """维护一个文件在当前 agent 会话中的已读取范围。"""

    path: str
    file_mtime_ns: int
    total_lines: int
    covered_ranges: list[tuple[int, int]] = field(default_factory=list)
    next_start: int = 1
    duplicate_requests: int = 0


@dataclass(frozen=True)
class FileReadCursor:
    """记录服务端生成的下一页读取游标。"""

    path: str
    next_start: int
    page_size: int
    file_mtime_ns: int


@dataclass
class ToolContext:
    root: Path
    path_resolver: Callable[[str], Path]
    shell_env_provider: Callable[[], Mapping[str, str]]
    depth: int
    max_depth: int
    run_delegates: Callable[[list[Mapping[str, Any]]], Any]
    interrupt_agents: Callable[[list[str]], int]
    read_file_state: dict[str, int] = field(default_factory=dict)
    read_coverage_state: dict[str, FileReadCoverage] = field(default_factory=dict)
    read_cursors: dict[str, FileReadCursor] = field(default_factory=dict)
    permissions: ProjectPermissions = field(default_factory=ProjectPermissions.empty)

    def path(self, raw_path: str) -> Path:
        return self.path_resolver(str(raw_path))

    def shell_env(self) -> dict[str, str]:
        return dict(self.shell_env_provider())
