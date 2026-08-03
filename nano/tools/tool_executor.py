"""Structured tool execution for the agent runtime."""

import asyncio
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nano.tools.tools import tool_definition
from nano.types import JsonObject, ToolArguments
from nano.utils import clip


class ToolExecutionMetadata(BaseModel):
    """描述工具执行结果的审计元数据。"""

    model_config = ConfigDict(extra="forbid")

    tool_status: str
    tool_error_code: str = ""
    security_event_type: str = ""
    risk_level: str = "low"
    read_only: bool = True
    affected_paths: list[str] = Field(default_factory=list)
    workspace_changed: bool = False
    diff_summary: list[str] = Field(default_factory=list)
    workspace_fingerprint: str = ""
    result_artifact_path: str = ""


class ToolExecutionResult(BaseModel):
    """统一承载工具文本结果和审计元数据。"""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: JsonObject


def _metadata(
    tool_status: str,
    tool_error_code: str = "",
    security_event_type: str = "",
    risk_level: str = "low",
    read_only: bool = True,
    affected_paths: Iterable[str] | None = None,
    workspace_changed: bool = False,
    workspace_fingerprint: str = "",
    diff_summary: Iterable[str] | None = None,
    result_artifact_path: str = "",
) -> JsonObject:
    """创建并序列化工具审计元数据。"""
    return ToolExecutionMetadata(
        tool_status=tool_status,
        tool_error_code=tool_error_code,
        security_event_type=security_event_type,
        risk_level=risk_level,
        read_only=read_only,
        affected_paths=list(affected_paths or []),
        workspace_changed=bool(workspace_changed),
        diff_summary=list(diff_summary or []),
        workspace_fingerprint=workspace_fingerprint,
        result_artifact_path=result_artifact_path,
    ).model_dump(mode="python")


class ToolExecutor:
    """执行工具并统一处理校验、审批、审计和异常。"""

    def __init__(self, agent: Any) -> None:
        """绑定运行时 agent。"""
        self.agent = agent

    def _render_result_content(self, tool: Any, content: str) -> tuple[str, str]:
        """将超长工具结果持久化，并返回模型可见预览与工件路径。"""
        content = str(content)
        if len(content) <= tool.max_result_size_chars:
            return content, ""
        artifact_path = self.agent.persist_tool_result(tool.name, content)
        if not artifact_path:
            return clip(content, tool.max_result_size_chars), ""
        notice = f"\n\nFull result persisted: {artifact_path}"
        preview_limit = max(61, tool.max_result_size_chars - len(notice))
        return clip(content, preview_limit) + notice, artifact_path

    def execute(self, name: str, args: ToolArguments) -> ToolExecutionResult:
        """执行一次工具调用并返回结构化结果。"""
        agent = self.agent
        tool = agent.tools.get(name)
        if tool is None:
            tool = next((candidate for candidate in agent.tools.values() if name in candidate.aliases), None)
        if tool is None:
            try:
                known_tool = tool_definition(name)
            except ValueError:
                known_tool = None
            if known_tool is not None and agent.allowed_tools is not None:
                return ToolExecutionResult(
                    content=f"error: tool '{name}' is not allowed in this run",
                    metadata=_metadata(
                        "rejected",
                        tool_error_code="tool_not_allowed",
                        risk_level="high",
                        read_only=False,
                    ),
                )
            return ToolExecutionResult(
                content=f"error: unknown tool '{name}'",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="unknown_tool",
                    risk_level="high",
                    read_only=False,
                ),
            )

        if agent.allowed_tools is not None and name not in agent.allowed_tools and tool.name not in agent.allowed_tools:
            return ToolExecutionResult(
                content=f"error: tool '{name}' is not allowed in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="tool_not_allowed",
                    risk_level="high",
                    read_only=False,
                ),
            )

        input_value: Any = None
        try:
            input_value = tool.parse_input(args)
            permission = tool.check_permissions(input_value, agent.tool_context())
            if permission.behavior != "allow":
                raise ValueError(permission.message)
            input_value = permission.updated_input or input_value
            args = input_value.model_dump(mode="python")
        except Exception as exc:
            example = agent.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            return ToolExecutionResult(
                content=message,
                metadata=_metadata(
                    "rejected",
                    tool_error_code="invalid_arguments",
                    security_event_type=security_event_type,
                    risk_level="high" if not tool.is_read_only(input_value) else "low",
                    read_only=tool.is_read_only(input_value),
                ),
            )

        if agent.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="repeated_identical_call",
                    risk_level="high" if not tool.is_read_only(input_value) else "low",
                    read_only=tool.is_read_only(input_value),
                ),
            )

        if not tool.is_read_only(input_value) and agent.read_only:
            return ToolExecutionResult(
                content=f"error: approval denied for {name}",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="approval_denied",
                    security_event_type="read_only_block",
                    risk_level="high",
                    read_only=False,
                ),
            )

        if tool.requires_approval(input_value, agent.tool_context()) and not agent.approve(tool.name, args):
            return ToolExecutionResult(
                content=f"error: approval denied for {name}",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="approval_denied",
                    security_event_type="approval_denied",
                    risk_level="high",
                    read_only=False,
                ),
            )

        before_snapshot = agent.capture_workspace_snapshot() if not tool.is_read_only(input_value) else {}
        after_snapshot = before_snapshot
        try:
            parent_message = next(
                (str(item.get("content", "")) for item in reversed(agent.session["history"]) if item.get("role") == "user"),
                None,
            )
            result = tool.call(
                input_value,
                agent.tool_context(),
                lambda candidate, _: agent.allowed_tools is None or candidate.name in agent.allowed_tools or name in agent.allowed_tools,
                parent_message,
            )
            content, result_artifact_path = self._render_result_content(tool, result.content)
            if not content.strip():
                content = "(no output)"
            after_snapshot = agent.capture_workspace_snapshot() if not tool.is_read_only(input_value) else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if tool.name == "run_shell":
                match = re.search(r"exit_code:\s*(-?\d+)", content)
                exit_code = int(match.group(1)) if match else 0
                if exit_code != 0 and workspace_changed:
                    tool_status = "partial_success"
                    tool_error_code = "tool_partial_success"
                elif exit_code != 0:
                    tool_status = "error"
                    tool_error_code = "tool_failed"
            agent.update_memory_after_tool(name, args, content)
            metadata = _metadata(
                tool_status,
                tool_error_code=tool_error_code,
                risk_level="high" if not tool.is_read_only(input_value) else "low",
                read_only=tool.is_read_only(input_value),
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
                result_artifact_path=result_artifact_path,
            )
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)
        except Exception as exc:
            after_snapshot = agent.capture_workspace_snapshot() if not tool.is_read_only(input_value) else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            metadata = _metadata(
                "partial_success" if workspace_changed else "error",
                tool_error_code="tool_partial_success" if workspace_changed else "tool_failed",
                security_event_type=security_event_type,
                risk_level="high" if not tool.is_read_only(input_value) else "low",
                read_only=tool.is_read_only(input_value),
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
            )
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=f"error: tool {name} failed: {exc}", metadata=metadata)

    async def execute_async(self, name: str, args: ToolArguments) -> ToolExecutionResult:
        """在不阻塞事件循环的前提下执行同步安全闸口。"""
        return await asyncio.to_thread(self.execute, name, args)
