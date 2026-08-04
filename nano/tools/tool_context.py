"""Narrow context passed from runtime into tool functions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from nano.permissions import ProjectPermissions


MAX_EXPLORER_LIST_FILES_CALLS = 5


@dataclass
class FileReadCoverage:
    """维护一个文件在当前 agent 会话中的已读取范围。"""

    path: str
    file_mtime_ns: int
    total_lines: int
    covered_ranges: list[tuple[int, int]] = field(default_factory=list)
    next_start: int = 1
    duplicate_requests: int = 0


@dataclass
class RequiredTargetState:
    """维护 explorer 必须取得的目标文件证据。"""

    path: str
    exists: bool
    kind: str
    total_lines: int | None
    file_mtime_ns: int | None
    covered_ranges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def evidence_complete(self) -> bool:
        """判断当前目标是否已获得完成任务所需的全部证据。"""
        if not self.exists:
            return True
        if self.kind != "file":
            return False
        if self.total_lines == 0:
            return True
        if self.total_lines is None:
            return False
        return len(self.unread_ranges()) == 0

    def record_coverage(self, start: int, end: int) -> None:
        """合并一次成功读取产生的连续行范围。"""
        if end < start:
            return
        ranges = sorted([*self.covered_ranges, (start, end)])
        merged: list[tuple[int, int]] = []
        for range_start, range_end in ranges:
            if merged and range_start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], range_end))
            else:
                merged.append((range_start, range_end))
        self.covered_ranges = merged

    def unread_ranges(self) -> list[tuple[int, int]]:
        """返回尚未读取的目标文件行范围。"""
        if not self.exists or self.kind != "file" or self.total_lines is None or self.total_lines == 0:
            return []
        unread: list[tuple[int, int]] = []
        cursor = 1
        for start, end in self.covered_ranges:
            if cursor < start:
                unread.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= self.total_lines:
            unread.append((cursor, self.total_lines))
        return unread

    def covered_line_count(self) -> int:
        """返回已读取范围覆盖的有效行数。"""
        if self.total_lines is None:
            return 0
        return sum(max(0, min(end, self.total_lines) - max(start, 1) + 1) for start, end in self.covered_ranges)


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
    agent_type: str = "root"
    read_file_state: dict[str, int] = field(default_factory=dict)
    read_coverage_state: dict[str, FileReadCoverage] = field(default_factory=dict)
    read_cursors: dict[str, FileReadCursor] = field(default_factory=dict)
    required_targets: dict[str, RequiredTargetState] = field(default_factory=dict)
    permissions: ProjectPermissions = field(default_factory=ProjectPermissions.empty)

    def path(self, raw_path: str) -> Path:
        return self.path_resolver(str(raw_path))

    def shell_env(self) -> dict[str, str]:
        return dict(self.shell_env_provider())
