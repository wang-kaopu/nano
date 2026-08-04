"""Structured tool execution for the agent runtime."""

import asyncio
import json
import re
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from nano.runtime.runtime import AgentRuntime
from nano.tools.tools import WorkspaceTool, tool_definition
from nano.utils.text import clip


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
    progress_made: bool = True
    counts_as_tool_step: bool = True
    duplicate_read: bool = False
    explorer_list_files_calls: int = 0
    explorer_list_files_limit: int = 0


class ToolExecutionResult(BaseModel):
    """统一承载工具文本结果和审计元数据。"""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: dict[str, Any]


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
    progress_made: bool = True,
    counts_as_tool_step: bool = True,
    duplicate_read: bool = False,
    explorer_list_files_calls: int = 0,
    explorer_list_files_limit: int = 0,
) -> dict[str, Any]:
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
        progress_made=progress_made,
        counts_as_tool_step=counts_as_tool_step,
        duplicate_read=duplicate_read,
        explorer_list_files_calls=explorer_list_files_calls,
        explorer_list_files_limit=explorer_list_files_limit,
    ).model_dump(mode="python")


class ToolExecutor:
    """执行工具并统一处理校验、审批、审计和异常。"""

    def __init__(self, runtime: AgentRuntime) -> None:
        """绑定运行时 agent。"""
        self.runtime = runtime

    def _render_result_content(self, tool: Any, content: str) -> tuple[str, str]:
        """将超长工具结果持久化，并返回模型可见预览与工件路径。"""
        content = str(content)
        if len(content) <= tool.max_result_size_chars:
            return content, ""
        artifact_path = self.runtime.persist_tool_result(tool.name, content)
        if not artifact_path:
            return clip(content, tool.max_result_size_chars), ""
        notice = f"\n\nFull result persisted: {artifact_path}"
        preview_limit = max(61, tool.max_result_size_chars - len(notice))
        return clip(content, preview_limit) + notice, artifact_path

    def _explorer_list_files_metadata(self, name: str) -> tuple[bool, int, int]:
        """返回当前工具是否使用 explorer 的独立目录枚举配额。"""
        if name != "list_files" or self.runtime.agent_type != "explorer":
            return True, 0, 0
        return False, self.runtime.explorer_list_files_calls, self.runtime.limits.explorer_list_files_limit

    def _render_read_file_result(self, content: str, max_result_size_chars: int) -> tuple[str, str]:
        """裁剪 read_file 正文时保留范围、游标和覆盖等分页元数据。"""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return self._render_result_content(type("Tool", (), {"name": "read_file", "max_result_size_chars": max_result_size_chars})(), content)
        if len(content) <= max_result_size_chars:
            return content, ""
        body = str(payload.get("content", ""))
        artifact_path = self.runtime.persist_tool_result("read_file", body)
        payload["content"] = clip(body, min(1_500, max_result_size_chars // 2))
        payload["contentTruncated"] = True
        payload["resultArtifactPath"] = artifact_path
        return json.dumps(payload, ensure_ascii=False), artifact_path

    @staticmethod
    def _read_file_execution_flags(content: str) -> tuple[str, str, bool, bool, bool]:
        """从 read_file 结构化状态推导进度和工具预算计数。"""
        try:
            status = str(json.loads(content).get("status", "ok"))
        except (json.JSONDecodeError, AttributeError):
            return "ok", "", True, True, False
        if status == "already_covered":
            duplicate_count = int(json.loads(content).get("duplicateReadCalls", 1))
            return "ok", "", False, duplicate_count >= 2, True
        if status == "cursor_stale":
            return "ok", "", False, False, False
        if status == "rejected":
            return "rejected", str(json.loads(content).get("errorCode", "repeated_read_range")), False, False, True
        return "ok", "", True, True, False

    def execute(self, name: str, args: Mapping[str, Any]) -> ToolExecutionResult:
        """执行一次工具调用并返回结构化结果。"""
        runtime = self.runtime
        tool = runtime.tools.get(name)
        if tool is None:
            tool = next((candidate for candidate in runtime.tools.values() if name in candidate.aliases), None)
        if tool is None:
            try:
                known_tool = tool_definition(name)
            except ValueError:
                known_tool = None
            if known_tool is not None and runtime.allowed_tools is not None:
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

        if runtime.allowed_tools is not None and name not in runtime.allowed_tools and tool.name not in runtime.allowed_tools:
            return ToolExecutionResult(
                content=f"error: tool '{name}' is not allowed in this run",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="tool_not_allowed",
                    risk_level="high",
                    read_only=False,
                ),
            )

        is_explorer_list_files = name == "list_files" and runtime.agent_type == "explorer"
        if is_explorer_list_files and runtime.explorer_list_files_calls >= runtime.limits.explorer_list_files_limit:
            return ToolExecutionResult(
                content=f"error: explorer list_files limit reached ({runtime.limits.explorer_list_files_limit})",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="explorer_list_files_limit_reached",
                    risk_level="low",
                    read_only=True,
                    progress_made=False,
                    counts_as_tool_step=False,
                    explorer_list_files_calls=runtime.explorer_list_files_calls,
                    explorer_list_files_limit=runtime.limits.explorer_list_files_limit,
                ),
            )

        input_value: Any = None
        try:
            input_value = tool.parse_input(args)
            permission = tool.check_permissions(input_value, runtime.tool_context())
            if permission.behavior != "allow":
                raise ValueError(permission.message)
            input_value = permission.updated_input or input_value
            args = input_value.model_dump(mode="python")
        except Exception as exc:
            example = runtime.tool_example(name)
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

        if tool.name != "read_file" and runtime.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata(
                    "rejected",
                    tool_error_code="repeated_identical_call",
                    risk_level="high" if not tool.is_read_only(input_value) else "low",
                    read_only=tool.is_read_only(input_value),
                ),
            )

        if not tool.is_read_only(input_value) and runtime.read_only:
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

        if tool.requires_approval(input_value, runtime.tool_context()) and not runtime.approve(tool.name, args):
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

        before_snapshot = runtime.capture_workspace_snapshot() if not tool.is_read_only(input_value) else {}
        after_snapshot = before_snapshot
        try:
            parent_message = next(
                (str(item.get("content", "")) for item in reversed(runtime.session["history"]) if item.get("role") == "user"),
                None,
            )
            result = tool.call(
                input_value,
                runtime.tool_context(),
                lambda candidate, _: runtime.allowed_tools is None or candidate.name in runtime.allowed_tools or name in runtime.allowed_tools,
                parent_message,
            )
            content, result_artifact_path = self._render_read_file_result(result.content, tool.max_result_size_chars) if tool.name == "read_file" else self._render_result_content(tool, result.content)
            if not content.strip():
                content = "(no output)"
            after_snapshot = runtime.capture_workspace_snapshot() if not tool.is_read_only(input_value) else before_snapshot
            affected_paths, diff_summary = runtime.diff_workspace_snapshots(before_snapshot, after_snapshot)
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
            runtime.update_memory_after_tool(name, args, content)
            read_status, read_error_code, progress_made, counts_as_tool_step, duplicate_read = self._read_file_execution_flags(content) if tool.name == "read_file" else ("ok", "", True, True, False)
            if is_explorer_list_files:
                runtime.explorer_list_files_calls += 1
                counts_as_tool_step, explorer_list_files_calls, explorer_list_files_limit = self._explorer_list_files_metadata(name)
            else:
                explorer_list_files_calls, explorer_list_files_limit = 0, 0
            metadata = _metadata(
                read_status if tool.name == "read_file" else tool_status,
                tool_error_code=read_error_code or tool_error_code,
                risk_level="high" if not tool.is_read_only(input_value) else "low",
                read_only=tool.is_read_only(input_value),
                affected_paths=affected_paths,
                workspace_changed=workspace_changed,
                workspace_fingerprint=runtime.workspace.fingerprint(),
                diff_summary=diff_summary,
                result_artifact_path=result_artifact_path,
                progress_made=progress_made,
                counts_as_tool_step=counts_as_tool_step,
                duplicate_read=duplicate_read,
                explorer_list_files_calls=explorer_list_files_calls,
                explorer_list_files_limit=explorer_list_files_limit,
            )
            runtime.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)
        except Exception as exc:
            after_snapshot = runtime.capture_workspace_snapshot() if not tool.is_read_only(input_value) else before_snapshot
            affected_paths, diff_summary = runtime.diff_workspace_snapshots(before_snapshot, after_snapshot)
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
                workspace_fingerprint=runtime.workspace.fingerprint(),
                diff_summary=diff_summary,
            )
            runtime.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=f"error: tool {name} failed: {exc}", metadata=metadata)

    async def execute_async(self, name: str, args: Mapping[str, Any]) -> ToolExecutionResult:
        """异步执行工具，并在事件循环内处理后台 agent 管理动作。"""
        tool = self.runtime.tools.get(name)
        if tool is None:
            tool = next((candidate for candidate in self.runtime.tools.values() if name in candidate.aliases), None)
        if not isinstance(tool, WorkspaceTool) or tool.async_runner is None:
            try:
                concurrency_safe = tool is not None and tool.is_concurrency_safe(tool.parse_input(args))
            except Exception:
                concurrency_safe = False
            if not concurrency_safe:
                async with self.runtime.workspace_mutation_lock:
                    return await asyncio.to_thread(self.execute, name, args)
            return await asyncio.to_thread(self.execute, name, args)

        runtime = self.runtime
        input_value: Any = None
        try:
            input_value = tool.parse_input(args)
            permission = tool.check_permissions(input_value, runtime.tool_context())
            if permission.behavior != "allow":
                raise ValueError(permission.message)
            input_value = permission.updated_input or input_value
            args = input_value.model_dump(mode="python")
        except Exception as exc:
            example = runtime.tool_example(name)
            message = f"error: invalid arguments for {name}: {exc}"
            if example:
                message += f"\nexample: {example}"
            return ToolExecutionResult(
                content=message,
                metadata=_metadata("rejected", tool_error_code="invalid_arguments", risk_level="low", read_only=True),
            )

        if tool.name != "read_file" and runtime.repeated_tool_call(name, args):
            return ToolExecutionResult(
                content=f"error: repeated identical tool call for {name}; choose a different tool or return a final answer",
                metadata=_metadata("rejected", tool_error_code="repeated_identical_call", risk_level="low", read_only=True),
            )

        try:
            parent_message = next(
                (str(item.get("content", "")) for item in reversed(runtime.session["history"]) if item.get("role") == "user"),
                None,
            )
            result = await tool.call_async(
                input_value,
                runtime.tool_context(),
                lambda candidate, _: runtime.allowed_tools is None or candidate.name in runtime.allowed_tools or name in runtime.allowed_tools,
                parent_message,
            )
            content, result_artifact_path = self._render_read_file_result(result.content, tool.max_result_size_chars) if tool.name == "read_file" else self._render_result_content(tool, result.content)
            if not content.strip():
                content = "(no output)"
            runtime.update_memory_after_tool(name, args, content)
            read_status, read_error_code, progress_made, counts_as_tool_step, duplicate_read = self._read_file_execution_flags(content) if tool.name == "read_file" else ("ok", "", True, True, False)
            metadata = _metadata(
                read_status,
                tool_error_code=read_error_code,
                risk_level="low",
                read_only=True,
                workspace_fingerprint=runtime.workspace.fingerprint(),
                result_artifact_path=result_artifact_path,
                progress_made=progress_made,
                counts_as_tool_step=counts_as_tool_step,
                duplicate_read=duplicate_read,
            )
            runtime.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=content, metadata=metadata)
        except Exception as exc:
            metadata = _metadata(
                "error",
                tool_error_code="tool_failed",
                risk_level="low",
                read_only=True,
                workspace_fingerprint=runtime.workspace.fingerprint(),
            )
            runtime.record_process_note_for_tool(name, metadata)
            return ToolExecutionResult(content=f"error: tool {name} failed: {exc}", metadata=metadata)
