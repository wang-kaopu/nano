"""Structured tool execution for the agent runtime."""

import asyncio
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .workspace import clip
from .types import JsonObject, ToolArguments


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
    ).model_dump(mode="python")


class ToolExecutor:
    """执行工具并统一处理校验、审批、审计和异常。"""

    def __init__(self, agent: Any) -> None:
        """绑定运行时 agent。"""
        self.agent = agent

    def execute(self, name: str, args: ToolArguments) -> ToolExecutionResult:
        """执行一次工具调用并返回结构化结果。"""
        agent = self.agent
        if agent.allowed_tools is not None and name not in agent.allowed_tools:
            return ToolExecutionResult(
                content=f"error: tool '{name}' is not allowed in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="tool_not_allowed",
                    risk_level="high",
                    read_only=False,
                ),
            )

        tool = agent.tools.get(name)
        if tool is None:
            return ToolExecutionResult(
                content=f"error: unknown tool '{name}'",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="unknown_tool",
                    risk_level="high",
                    read_only=False,
                ),
            )

        try:
            args = agent.validate_tool(name, args)
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
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                ),
            )

        if agent.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="repeated_identical_call",
                    risk_level="high" if tool["risky"] else "low",
                    read_only=not tool["risky"],
                ),
            )

        if tool["risky"] and not agent.approve(name, args):
            return ToolExecutionResult(
                content=f"error: approval denied for {name}",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="approval_denied",
                    security_event_type="read_only_block" if agent.read_only else "approval_denied",
                    risk_level="high",
                    read_only=False,
                ),
            )

        before_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else {}
        after_snapshot = before_snapshot
        try:
            content = clip(tool["run"](args))
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            tool_status = "ok"
            tool_error_code = ""
            if name == "run_shell":
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
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=agent.workspace.fingerprint(),
                diff_summary=diff_summary,
            )
            agent.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)
        except Exception as exc:
            after_snapshot = agent.capture_workspace_snapshot() if tool["risky"] else before_snapshot
            affected_paths, diff_summary = agent.diff_workspace_snapshots(before_snapshot, after_snapshot)
            workspace_changed = bool(affected_paths)
            security_event_type = "path_escape" if "path escapes workspace" in str(exc) else ""
            metadata = _metadata(
                "partial_success" if workspace_changed else "error",
                tool_error_code="tool_partial_success" if workspace_changed else "tool_failed",
                security_event_type=security_event_type,
                risk_level="high" if tool["risky"] else "low",
                read_only=not tool["risky"],
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
