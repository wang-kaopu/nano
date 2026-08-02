"""Narrow context passed from runtime into tool functions."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from .types import Environment, ToolArguments


@dataclass
class ToolContext:
    root: Path
    path_resolver: Callable[[str], Path]
    shell_env_provider: Callable[[], Environment]
    depth: int
    max_depth: int
    spawn_delegate: Callable[[ToolArguments], str]

    def path(self, raw_path: str) -> Path:
        return self.path_resolver(str(raw_path))

    def shell_env(self) -> dict[str, str]:
        return dict(self.shell_env_provider())
