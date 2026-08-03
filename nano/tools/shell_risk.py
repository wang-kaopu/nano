"""Shell AST parsing utilities shared by permission evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import bashlex
from bashlex import ast
from bashlex.errors import ParsingError


class ShellCommandParseError(ValueError):
    """表示命令无法被 Bash AST 解析，因而不能安全执行。"""


@dataclass(frozen=True)
class ShellCommandSegment:
    """描述一段可独立匹配权限规则的 shell 命令。"""

    text: str


def shell_command_segments(command: str) -> tuple[ShellCommandSegment, ...]:
    """解析 Bash AST，并返回顶层、嵌套及命令替换中的每个命令片段。"""
    try:
        trees = bashlex.parse(command)
    except ParsingError as exc:
        raise ShellCommandParseError(f"shell command cannot be parsed: {exc}") from exc

    visitor = _CommandSegmentVisitor(command)
    for tree in trees:
        visitor.visit(tree)
    return tuple(visitor.segments)


class _CommandSegmentVisitor(ast.nodevisitor):
    """从 Bash AST 中提取 command 节点的原始命令文本。"""

    def __init__(self, source: str) -> None:
        """初始化命令文本和去重后的片段集合。"""
        self.source = source
        self.segments: list[ShellCommandSegment] = []
        self._seen: set[str] = set()

    def visitcommand(self, node: Any, parts: list[Any]) -> None:
        """记录当前 command 节点对应的源文本。"""
        start, end = node.pos
        text = self.source[start:end].strip()
        if text and text not in self._seen:
            self._seen.add(text)
            self.segments.append(ShellCommandSegment(text=text))
