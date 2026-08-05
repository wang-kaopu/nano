"""Project-local tool and shell permission policies loaded from permissions.json."""

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
class PermissionRules:
    """描述一类操作的允许与拒绝规则。"""

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectPermissions:
    """描述当前仓库声明的工具与 shell 权限规则。"""

    tools: PermissionRules = PermissionRules()
    shell: PermissionRules = PermissionRules()

    @classmethod
    def empty(cls) -> "ProjectPermissions":
        """创建不自动放行任何工具调用的默认策略。"""
        return cls()

    def decision(self, tool_name: str, command: str | None = None) -> PermissionDecision:
        """按操作类型返回项目级权限决策。"""
        if tool_name == "run_shell":
            return self.shell_decision(command)
        return self.tool_decision(tool_name)

    def tool_decision(self, tool_name: str) -> PermissionDecision:
        """按 deny 优先顺序返回普通工具的项目级权限决策。"""
        if any(self._matches_tool_rule(rule, tool_name) for rule in self.tools.deny):
            return "deny"
        return "allow" if any(self._matches_tool_rule(rule, tool_name) for rule in self.tools.allow) else "no_match"

    def shell_decision(self, command: str | None) -> PermissionDecision:
        """按 deny 优先顺序返回 shell 命令的项目级权限决策。"""
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
        return any(self._matches_shell_pattern(rule, segment.text) for rule in self.shell.deny for segment in segments)

    def _shell_allow_matches(self, segments: tuple[ShellCommandSegment, ...]) -> bool:
        """要求每个命令片段均匹配 allow，避免复合命令借安全前缀绕过审批。"""
        return bool(segments) and all(
            any(self._matches_shell_pattern(rule, segment.text) for rule in self.shell.allow)
            for segment in segments
        )

    @staticmethod
    def _matches_tool_rule(rule: str, tool_name: str) -> bool:
        """匹配工具名称，并兼容 grep_search 这个项目策略别名。"""
        return _TOOL_RULE_ALIASES.get(rule, rule) == tool_name

    @staticmethod
    def _matches_shell_pattern(rule: str, command: str) -> bool:
        """匹配单条 shell glob 规则与一个 AST 命令片段。"""
        return fnmatch.fnmatchcase(command, rule)


def load_project_permissions(root: str | Path, permission_path: str | Path | None = None) -> ProjectPermissions:
    """读取默认或调用方注入的权限文件，并验证配置结构。"""
    path = Path(permission_path) if permission_path is not None else Path(root) / PERMISSIONS_FILE_NAME
    if not path.exists():
        return ProjectPermissions.empty()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"permissions"} or not isinstance(payload["permissions"], dict):
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: expected a permissions object")
    permissions = payload["permissions"]
    if set(permissions) != {"tools", "shell"}:
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: expected tools and shell objects")
    return ProjectPermissions(
        tools=_permission_rules(permissions["tools"], "tools"),
        shell=_permission_rules(permissions["shell"], "shell"),
    )


def _permission_rules(value: object, section: str) -> PermissionRules:
    """验证一个权限分组，并返回其中的 allow 与 deny 规则。"""
    if not isinstance(value, dict) or set(value) - {"allow", "deny"}:
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: permissions.{section} must be an object")
    return PermissionRules(
        allow=_rules(value.get("allow", []), f"{section}.allow"),
        deny=_rules(value.get("deny", []), f"{section}.deny"),
    )


def _rules(value: object, field: str) -> tuple[str, ...]:
    """验证 allow 或 deny 中的规则均为非空字符串。"""
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"invalid {PERMISSIONS_FILE_NAME}: permissions.{field} must be a string list")
    return tuple(item.strip() for item in value)
