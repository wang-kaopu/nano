"""Narrow context passed from runtime into tool functions."""

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable

from nano.permissions import ProjectPermissions
from nano.types import Environment, ToolArguments


@dataclass
class ToolContext:
    root: Path
    path_resolver: Callable[[str], Path]
    shell_env_provider: Callable[[], Environment]
    depth: int
    max_depth: int
    spawn_delegate: Callable[[ToolArguments], str]
    read_file_state: dict[str, int] = field(default_factory=dict)
    permissions: ProjectPermissions = field(default_factory=ProjectPermissions.empty)

    def path(self, raw_path: str) -> Path:
        return self.path_resolver(str(raw_path))

    def shell_env(self) -> dict[str, str]:
        return dict(self.shell_env_provider())
