"""Project-local tool permission policy loaded from permissions.json."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from nano.tools.shell_risk import ShellCommandSegment, shell_command_segments


PERMISSIONS_FILE_NAME = "permissions.json"
PermissionDecision = Literal["allow", "deny", "no_match"]
_TOOL_RULE_ALIASES = {"grep_search": "search"}


@dataclass(frozen=True)
class ProjectPermissions:
    """描述当前仓库声明的工具允许与拒绝规则。"""

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> "ProjectPermissions":
        """创建不自动放行任何工具调用的默认策略。"""
        return cls()

    def decision(self, tool_name: str, command: str | None = None) -> PermissionDecision:
        """按 deny 优先顺序返回本次工具调用的项目级权限决策。"""
        if tool_name != "run_shell":
            if any(self._matches_tool_rule(rule, tool_name) for rule in self.deny):
                return "deny"
            return "allow" if any(self._matches_tool_rule(rule, tool_name) for rule in self.allow) else "no_match"
        if command is None:
            return "no_match"
        segments = shell_command_segments(command)
        if self._shell_deny_matches(segments):
            return "deny"
        if self._shell_allow_matches(segments):
            return "allow"
        return "no_match"

    def _shell_deny_matches(self, segments: tuple[ShellCommandSegment, ...]) -> bool:
        """判断任一命令片段是否命中 deny；deny 必须能阻断复合命令。"""
        return "run_shell" in self.deny or any(
            self._matches_shell_pattern(rule, segment.text)
            for rule in self.deny
            if rule.startswith("run_shell(")
            for segment in segments
        )

    def _shell_allow_matches(self, segments: tuple[ShellCommandSegment, ...]) -> bool:
        """要求每个命令片段均匹配 allow，避免复合命令借安全前缀绕过审批。"""
        if "run_shell" in self.allow:
            return True
        return bool(segments) and all(
            any(self._matches_shell_pattern(rule, segment.text) for rule in self.allow if rule.startswith("run_shell("))
            for segment in segments
        )

    @staticmethod
    def _matches_tool_rule(rule: str, tool_name: str) -> bool:
        """匹配工具名称，并兼容 grep_search 这个项目策略别名。"""
        return _TOOL_RULE_ALIASES.get(rule, rule) == tool_name

    @staticmethod
    def _matches_shell_pattern(rule: str, command: str) -> bool:
        """匹配单条 run_shell glob 规则与一个 AST 命令片段。"""
        if not rule.endswith(")"):
            return False
        pattern = rule[len("run_shell(") : -1].strip()
        return bool(pattern) and fnmatch.fnmatchcase(command, pattern)


def load_project_permissions(root: str | Path) -> ProjectPermissions:
    """读取仓库根目录的 permissions.json，并验证配置结构。"""
    path = Path(root) / PERMISSIONS_FILE_NAME
    if not path.exists():
        return ProjectPermissions.empty()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"permissions"} or not isinstance(payload["permissions"], dict):
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: expected a permissions object")
    permissions = payload["permissions"]
    if set(permissions) - {"allow", "deny"}:
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: unknown permissions fields")
    allow = _rules(permissions.get("allow", []), "allow")
    deny = _rules(permissions.get("deny", []), "deny")
    return ProjectPermissions(allow=allow, deny=deny)


def _rules(value: object, field: str) -> tuple[str, ...]:
    """验证 allow 或 deny 中的规则均为非空字符串。"""
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: permissions.{field} must be a string list")
    return tuple(item.strip() for item in value)
