"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import json
import secrets
import shutil
import subprocess
import textwrap
from typing import Annotated, Any, Callable, Literal, Mapping, Type

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nano.tools.tool import CanUseTool, PermissionResult, ProgressCallback, Tool, ToolProgressData, ToolResult
from nano.tools.tool_context import FileReadCoverage, FileReadCursor, ToolContext
from nano.tools.shell_risk import ShellCommandParseError, shell_command_segments
from nano.permissions import PERMISSIONS_FILE_NAME
from nano.skills import execute_skill
from nano.types import ToolArguments as ToolArgumentsPayload
from nano.workspace.context import IGNORED_PATH_NAMES


class ToolArguments(BaseModel):
    """所有模型生成工具参数的共同校验配置。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class ListFilesArguments(ToolArguments):
    """列出目录内容所需的参数。"""

    path: str = "."


class ReadFileArguments(ToolArguments):
    """读取文件所需的参数。"""

    path: NonEmptyText
    start: int | None = Field(default=None, ge=1)
    end: int | None = Field(default=None, ge=1)
    cursor: str = ""
    page_size: int = Field(default=200, ge=1, le=500)

    @model_validator(mode="after")
    def validate_line_range(self) -> "ReadFileArguments":
        """确保游标模式和显式范围模式不会混用。"""
        if self.cursor:
            if self.start is not None or self.end is not None:
                raise ValueError("cursor cannot be combined with start or end")
            return self
        if self.start is None:
            self.start = 1
        if self.end is None:
            self.end = self.start + self.page_size - 1
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class SearchArguments(ToolArguments):
    """搜索文本所需的参数。"""

    pattern: NonEmptyText
    path: str = "."


class RunShellArguments(ToolArguments):
    """执行 shell 命令所需的参数。"""

    command: NonEmptyText
    timeout: int = Field(default=20, ge=1, le=120)


def requires_run_shell_approval(args: ToolArguments, context: ToolContext | None = None) -> bool:
    """根据已校验的 shell 命令 AST 决定是否需要人工审批。"""
    if not isinstance(args, RunShellArguments):
        raise TypeError("run_shell approval requires RunShellArguments")
    if context is not None and context.permissions.decision("run_shell", args.command) == "allow":
        return False
    return True


class WriteFileArguments(ToolArguments):
    """写入文件所需的参数。"""

    path: NonEmptyText
    content: str


class PatchFileArguments(ToolArguments):
    """精确替换文件内容所需的参数。"""

    path: NonEmptyText
    old_text: NonEmptyText
    new_text: str


class DelegateTaskSpec(ToolArguments):
    """描述一个需要由子 agent 完成的明确任务。"""

    task: NonEmptyText
    type: Literal["explorer", "worker", "default"] = Field(
        default="default",
        description="explorer is required for reading/searching/reviewing; worker is only for scoped code changes; default inherits the parent context.",
    )
    scope: str = Field(
        default="",
        description="Only for worker: an existing non-root directory such as 'nano/runtime', never a file name or a natural-language task description.",
    )
    targets: list[NonEmptyText] = Field(default_factory=list)
    requested_max_steps: int | None = Field(default=None, ge=1)


class DelegateArguments(ToolArguments):
    """批量委派并结构化等待子任务所需的参数。"""

    tasks: list[DelegateTaskSpec] = Field(min_length=1, max_length=8)


class InterruptAgentsArguments(ToolArguments):
    """中断后台子任务所需的参数。"""

    async_agent_task_ids: list[str] = Field(default_factory=list)


class SkillArguments(ToolArguments):
    """调用项目 Skill 所需的参数。"""

    skill_name: NonEmptyText
    args: str = ""


class WorkspaceTool(Tool[ToolArguments, str, ToolProgressData]):
    """将工作区工具函数适配为统一 Tool 契约。"""

    def __init__(
        self,
        *,
        name: str,
        input_schema: Type[ToolArguments],
        description_text: str,
        prompt_text: str,
        runner: Callable[[ToolContext, ToolArgumentsPayload], str] | None,
        read_only: bool,
        concurrency_safe: bool,
        destructive: bool = False,
        approval_required: Callable[[ToolArguments, ToolContext | None], bool] | None = None,
        aliases: tuple[str, ...] = (),
        async_runner: Callable[[ToolContext, ToolArgumentsPayload], Any] | None = None,
    ) -> None:
        """初始化由已有工作区执行函数驱动的工具。"""
        super().__init__(name=name, input_schema=input_schema, aliases=aliases)
        self.description_text = description_text
        self.prompt_text = prompt_text
        self.runner = runner
        self.async_runner = async_runner
        self.read_only = read_only
        self.concurrency_safe = concurrency_safe
        self.destructive = destructive
        self.approval_required = approval_required

    def description(self, input_value: ToolArguments | None, options: Mapping[str, Any] | None = None) -> str:
        """返回工具能力说明。"""
        return self.description_text

    def prompt(self, options: Mapping[str, Any] | None = None) -> str:
        """返回工具调用指南。"""
        return self.prompt_text

    def is_concurrency_safe(self, input_value: ToolArguments) -> bool:
        """返回当前工具输入的并发安全性。"""
        return self.concurrency_safe

    def is_read_only(self, input_value: ToolArguments) -> bool:
        """返回当前工具输入是否只读。"""
        return self.read_only

    def is_destructive(self, input_value: ToolArguments) -> bool:
        """返回当前工具输入是否可能修改工作区或执行命令。"""
        return self.destructive

    def requires_approval(self, input_value: ToolArguments, context: ToolContext | None = None) -> bool:
        """返回当前输入是否需要按运行时审批策略人工确认。"""
        command = input_value.command if isinstance(input_value, RunShellArguments) else None
        if context is not None and context.permissions.decision(self.name, command) == "allow":
            return False
        if self.approval_required is not None:
            return self.approval_required(input_value, context)
        return super().requires_approval(input_value, context)

    def check_permissions(self, input_value: ToolArguments, context: ToolContext) -> PermissionResult:
        """执行工作区路径、文件状态和委派深度检查。"""
        try:
            _validate_workspace_input(context, self.name, input_value.model_dump(mode="python"))
            command = input_value.command if isinstance(input_value, RunShellArguments) else None
            if context.permissions.decision(self.name, command) == "deny":
                return PermissionResult.deny(f"permission denied by {PERMISSIONS_FILE_NAME}", input_value)
            if self.name == "run_shell" and isinstance(input_value, RunShellArguments):
                shell_command_segments(input_value.command)
        except (ShellCommandParseError, ValueError) as exc:
            return PermissionResult.deny(str(exc), input_value)
        return PermissionResult.allow(input_value)

    def call(
        self,
        args: ToolArguments,
        context: ToolContext,
        can_use_tool: CanUseTool,
        parent_message: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult[str]:
        """执行已通过权限检查的底层工作区工具。"""
        if self.runner is None:
            raise RuntimeError(f"tool '{self.name}' must be executed asynchronously")
        if not can_use_tool(self, args):
            raise PermissionError(f"tool '{self.name}' is not allowed in this run")
        if on_progress is not None:
            on_progress(ToolProgressData(stage="running"))
        output = self.runner(context, args.model_dump(mode="python"))
        progress = ToolProgressData(stage="completed")
        if on_progress is not None:
            on_progress(progress)
        return ToolResult(output=output, content=output, progress=progress)

    async def call_async(
        self,
        args: ToolArguments,
        context: ToolContext,
        can_use_tool: CanUseTool,
        parent_message: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult[str]:
        """执行显式声明的异步 runner。"""
        if self.async_runner is None:
            raise RuntimeError(f"tool '{self.name}' does not provide an async runner")
        if not can_use_tool(self, args):
            raise PermissionError(f"tool '{self.name}' is not allowed in this run")
        if on_progress is not None:
            on_progress(ToolProgressData(stage="running"))
        output = await self.async_runner(context, args.model_dump(mode="python"))
        progress = ToolProgressData(stage="completed")
        if on_progress is not None:
            on_progress(progress)
        return ToolResult(output=output, content=output, progress=progress)


def tool_arguments_model(name: str) -> Type[ToolArguments]:
    """返回工具名称对应的 Pydantic 参数模型。"""
    return tool_definition(name).input_schema


def tool_json_schema(name: str) -> dict[str, Any]:
    """返回供模型调用协议使用的 JSON Schema。"""
    return tool_arguments_model(name).model_json_schema()


def validate_tool_arguments(name: str, args: ToolArgumentsPayload | None) -> dict[str, Any]:
    """校验工具参数并返回规范化后的普通字典。"""
    model = tool_arguments_model(name).model_validate(args or {})
    return model.model_dump(mode="python")


def _human_schema(model: Type[ToolArguments]) -> dict[str, str]:
    """把 Pydantic 字段转换为 prompt 中使用的简短字段描述。"""
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    result = {}
    for name, value in properties.items():
        field_type = value.get("type", "value")
        default = value.get("default")
        suffix = "" if name in required else f"={default!r}"
        result[name] = f"{field_type}{suffix}"
    return result

def legal_tool_names() -> set[str]:
    """返回包括废弃别名在内的全部可识别工具名。"""
    return {name for tool in TOOL_DEFINITIONS for name in (tool.name, *tool.aliases)}

TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"tasks":[{"task":"inspect README.md","type":"explorer","targets":["README.md"]}]}}</tool>',
    "interrupt_agents": '<tool>{"name":"interrupt_agents","args":{"async_agent_task_ids":["task-id"]}}</tool>',
    "skill": '<tool>{"name":"skill","args":{"skill_name":"commit","args":"stage the current changes"}}</tool>',
}


def build_tool_registry(context: ToolContext) -> dict[str, Tool[ToolArguments, str, ToolProgressData]]:
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools: dict[str, Tool[ToolArguments, str, ToolProgressData]] = {
        tool.name: tool for tool in TOOL_DEFINITIONS if tool.name != "delegate"
    }
    # 子 agent 的递归委派需要明确受深度上限约束。
    if context.depth < context.max_depth:
        delegate = next(tool for tool in TOOL_DEFINITIONS if tool.name == "delegate")
        tools[delegate.name] = delegate
    return tools


def tool_example(name: str) -> str:
    return TOOL_EXAMPLES.get(name, "")


def _validate_workspace_input(context: ToolContext, name: str, args: ToolArgumentsPayload) -> None:
    """校验工具专属的工作区状态，不重复 Pydantic 的输入 schema 校验。"""

    if name == "list_files":
        path = context.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return

    if name == "read_file":
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = args.get("start")
        end = args.get("end")
        if not args.get("cursor") and (start is None or end is None or int(start) < 1 or int(end) < int(start)):
            raise ValueError("invalid line range")
        return

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        context.path(args.get("path", "."))
        return

    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return

    if name == "write_file":
        path = context.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        if path.exists():
            # 既有文件必须基于本 agent 已读取的版本修改，避免盲写覆盖外部变更。
            recorded_mtime = context.read_file_state.get(str(path))
            if recorded_mtime is None:
                raise ValueError("you must read this file before modifying it; use read_file first")
            if path.stat().st_mtime_ns // 1_000_000 != recorded_mtime:
                raise ValueError("warning: file was modified externally; use read_file again")
        return

    if name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        recorded_mtime = context.read_file_state.get(str(path))
        if recorded_mtime is None:
            raise ValueError("you must read this file before modifying it; use read_file first")
        if path.stat().st_mtime_ns // 1_000_000 != recorded_mtime:
            raise ValueError("warning: file was modified externally; use read_file again")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        return

    if name == "delegate":
        if context.depth >= context.max_depth:
            raise ValueError("delegate depth exceeded")
        for task_spec in args.get("tasks", []):
            task = str(task_spec.get("task", "")).strip()
            if not task:
                raise ValueError("delegate task must not be empty")
            agent_type = str(task_spec.get("type", "default"))
            targets = task_spec.get("targets", [])
            if agent_type == "explorer" and not targets:
                raise ValueError("explorer delegate requires targets")
            if agent_type != "worker" and str(task_spec.get("scope", "")).strip():
                raise ValueError("delegate scope is only supported for worker tasks")
            if agent_type == "worker":
                scope = str(task_spec.get("scope", "")).strip()
                if not scope or scope == ".":
                    raise ValueError("worker delegate requires an existing non-root directory")
                scope_path = context.path(scope)
                if not scope_path.is_dir():
                    raise ValueError("worker scope must be an existing directory")
        return

    if name == "interrupt_agents":
        if any(not str(task_id).strip() for task_id in args.get("async_agent_task_ids", [])):
            raise ValueError("async_agent_task_ids must not contain empty values")
        return

    if name == "skill":
        if not str(args.get("skill_name", "")).strip():
            raise ValueError("skill_name must not be empty")
        return

    raise ValueError(f"unknown tool: {name}")


def tool_list_files(context: ToolContext, args: ToolArgumentsPayload) -> str:
    path = context.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(context.root)}")
    return "\n".join(lines) or "(empty)"


def tool_read_file(context: ToolContext, args: ToolArgumentsPayload) -> str:
    """读取文件的一页，并返回范围、覆盖和下一页游标的结构化信息。"""
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    path_key = str(path)
    mtime_ns = path.stat().st_mtime_ns
    page_size = int(args.get("page_size", 200))
    cursor = str(args.get("cursor", ""))
    if cursor:
        cursor_state = context.read_cursors.get(cursor)
        if cursor_state is None or cursor_state.path != path_key:
            return json.dumps({"status": "cursor_stale", "path": str(path.relative_to(context.root)), "message": "The file changed after the previous read. Start a new read sequence."}, ensure_ascii=False)
        if cursor_state.file_mtime_ns != mtime_ns:
            return json.dumps({"status": "cursor_stale", "path": str(path.relative_to(context.root)), "message": "The file changed after the previous read. Start a new read sequence."}, ensure_ascii=False)
        start = cursor_state.next_start
        page_size = cursor_state.page_size
        end = start + page_size - 1
    else:
        start = int(args.get("start") or 1)
        end = int(args.get("end") or start + page_size - 1)
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total_lines = len(lines)
    required_target = context.required_targets.get(str(path.relative_to(context.root)))
    if required_target is not None and required_target.file_mtime_ns != mtime_ns:
        # 目标在子 agent 启动后发生变化时，旧的读取证据不能证明当前文件已完整覆盖。
        required_target.file_mtime_ns = mtime_ns
        required_target.total_lines = total_lines
        required_target.covered_ranges = []
    coverage = context.read_coverage_state.get(path_key)
    if coverage is None or coverage.file_mtime_ns != mtime_ns:
        coverage = FileReadCoverage(path=path_key, file_mtime_ns=mtime_ns, total_lines=total_lines)
        context.read_coverage_state[path_key] = coverage
    returned_start = min(start, total_lines + 1)
    returned_end = min(end, total_lines)
    next_start = min(end + 1, total_lines + 1)
    next_cursor = ""
    has_more = next_start <= total_lines
    is_already_covered = any(existing_start <= returned_start and returned_end <= existing_end for existing_start, existing_end in coverage.covered_ranges)
    if total_lines and returned_start <= returned_end and is_already_covered:
        coverage.duplicate_requests += 1
        if coverage.duplicate_requests >= 3:
            return json.dumps({"status": "rejected", "errorCode": "repeated_read_range", "path": str(path.relative_to(context.root)), "requestedRange": {"start": start, "end": end}, "coveredRanges": [{"start": item[0], "end": item[1]} for item in coverage.covered_ranges], "duplicateReadCalls": coverage.duplicate_requests}, ensure_ascii=False)
        next_cursor = secrets.token_urlsafe(18) if has_more else ""
        if next_cursor:
            context.read_cursors[next_cursor] = FileReadCursor(path_key, next_start, page_size, mtime_ns)
        return json.dumps({"status": "already_covered", "path": str(path.relative_to(context.root)), "requestedRange": {"start": start, "end": end}, "coveredRanges": [{"start": item[0], "end": item[1]} for item in coverage.covered_ranges], "nextRange": {"start": next_start, "end": min(next_start + page_size - 1, total_lines)} if has_more else None, "nextCursor": next_cursor, "duplicateReadCalls": coverage.duplicate_requests}, ensure_ascii=False)
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    if total_lines and returned_start <= returned_end:
        ranges = sorted([*coverage.covered_ranges, (returned_start, returned_end)])
        merged: list[tuple[int, int]] = []
        for range_start, range_end in ranges:
            if merged and range_start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], range_end))
            else:
                merged.append((range_start, range_end))
        coverage.covered_ranges = merged
        if required_target is not None:
            required_target.record_coverage(returned_start, returned_end)
    coverage.next_start = next_start
    if has_more:
        next_cursor = secrets.token_urlsafe(18)
        context.read_cursors[next_cursor] = FileReadCursor(path_key, next_start, page_size, mtime_ns)
    try:
        context.read_file_state[str(path)] = mtime_ns // 1_000_000
    except OSError:
        pass
    return json.dumps({"status": "ok", "path": str(path.relative_to(context.root)), "requestedRange": {"start": start, "end": end}, "returnedRange": {"start": returned_start, "end": returned_end}, "totalLines": total_lines, "hasMore": has_more, "nextRange": {"start": next_start, "end": min(next_start + page_size - 1, total_lines)} if has_more else None, "nextCursor": next_cursor, "coveredRanges": [{"start": item[0], "end": item[1]} for item in coverage.covered_ranges], "contentTruncated": False, "resultArtifactPath": "", "content": body}, ensure_ascii=False)


def tool_search(context: ToolContext, args: ToolArgumentsPayload) -> str:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = context.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=context.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(context.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(context.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_run_shell(context: ToolContext, args: ToolArgumentsPayload) -> str:
    """在本地或运行时注入的隔离执行器中运行受限 shell 命令。"""
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    if context.shell_executor is not None:
        return context.execute_shell(command, timeout)
    result = subprocess.run(
        command,
        cwd=context.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
        env=context.shell_env(),
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(no output)"}
        stderr:
        {result.stderr.strip() or "(no output)"}
        """
    ).strip()


def tool_write_file(context: ToolContext, args: ToolArgumentsPayload) -> str:
    path = context.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    try:
        context.read_file_state[str(path)] = path.stat().st_mtime_ns // 1_000_000
    except OSError:
        pass
    return f"wrote {path.relative_to(context.root)} ({len(content)} chars)"


def tool_patch_file(context: ToolContext, args: ToolArgumentsPayload) -> str:
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    try:
        context.read_file_state[str(path)] = path.stat().st_mtime_ns // 1_000_000
    except OSError:
        pass
    return f"patched {path.relative_to(context.root)}"


async def tool_delegate(context: ToolContext, args: ToolArgumentsPayload) -> str:
    """并发执行全部子任务，并在所有子任务结束后返回。"""
    if context.depth >= context.max_depth:
        raise ValueError("delegate depth exceeded")
    tasks = list(args["tasks"])
    return await context.run_delegates(tasks)


async def tool_interrupt_agents(context: ToolContext, args: ToolArgumentsPayload) -> str:
    """中断指定后台子任务；未指定时中断全部子任务。"""
    task_ids = [str(task_id).strip() for task_id in args.get("async_agent_task_ids", [])]
    cancelled_count = context.interrupt_agents(task_ids)
    return json.dumps({"status": "interrupt_requested", "cancelledCount": cancelled_count, "asyncAgentTaskIds": task_ids})


def tool_skill(context: ToolContext, args: ToolArgumentsPayload) -> str:
    """展开项目 Skill，并将其提示词作为模型下一步指令返回。"""
    skill_name = str(args["skill_name"])
    result = execute_skill(skill_name, str(args.get("args", "")), workspace_root=context.root)
    if result is None:
        return f"Unknown skill: {skill_name}"
    return f'[Skill "{skill_name}" activated]\n\n{result["prompt"]}'


TOOL_DEFINITIONS: tuple[WorkspaceTool, ...] = (
    WorkspaceTool(
        name="list_files",
        input_schema=ListFilesArguments,
        description_text="List files in the workspace.",
        prompt_text="Use list_files to inspect a directory before making assumptions about its contents.",
        runner=tool_list_files,
        read_only=True,
        concurrency_safe=True,
    ),
    WorkspaceTool(
        name="read_file",
        input_schema=ReadFileArguments,
        description_text="Read a UTF-8 file by range or opaque cursor and return pagination metadata.",
        prompt_text="Use read_file before modifying an existing file. Prefer its nextCursor for pagination; do not reread covered ranges.",
        runner=tool_read_file,
        read_only=True,
        concurrency_safe=True,
    ),
    WorkspaceTool(
        name="search",
        input_schema=SearchArguments,
        description_text="Search the workspace with rg or a simple fallback.",
        prompt_text="Use search to locate symbols or text across the workspace.",
        runner=tool_search,
        read_only=True,
        concurrency_safe=True,
        aliases=("grep_search",),
    ),
    WorkspaceTool(
        name="run_shell",
        input_schema=RunShellArguments,
        description_text="Run a shell command in the repo root.",
        prompt_text="Use run_shell only when file tools cannot perform the required command or verification.",
        runner=tool_run_shell,
        read_only=False,
        concurrency_safe=False,
        destructive=True,
        approval_required=requires_run_shell_approval,
    ),
    WorkspaceTool(
        name="write_file",
        input_schema=WriteFileArguments,
        description_text="Write a new text file or overwrite a previously read file.",
        prompt_text="Use write_file for new files or after read_file has confirmed an existing file version.",
        runner=tool_write_file,
        read_only=False,
        concurrency_safe=False,
        destructive=True,
    ),
    WorkspaceTool(
        name="patch_file",
        input_schema=PatchFileArguments,
        description_text="Replace one exact text block in a previously read file.",
        prompt_text="Use patch_file after read_file when one exact, unique text block should change.",
        runner=tool_patch_file,
        read_only=False,
        concurrency_safe=False,
        destructive=True,
    ),
    WorkspaceTool(
        name="delegate",
        input_schema=DelegateArguments,
        description_text="Launch all child agents concurrently and wait for every child to reach a terminal state.",
        prompt_text=(
            "Submit every currently planned child task in one delegate call. The call launches all children concurrently and returns only after each child "
            "finishes, stops, fails, or is cancelled. Explorer tasks that inspect files must include targets. Worker tasks require an existing non-root scope. "
            "requested_max_steps is only a suggestion: the runtime raises it when pagination requires more reads. Default shares the main workspace mutation lock, "
            "while explorer and worker start with empty history. After delegate returns, treat a child answer as complete only when evidenceComplete is true. "
            "Do not reread targets from children with complete evidence. When evidenceComplete is false, inspect missingTargets and recover only missing targets or unread ranges."
        ),
        runner=None,
        async_runner=tool_delegate,
        read_only=True,
        concurrency_safe=False,
    ),
    WorkspaceTool(
        name="interrupt_agents",
        input_schema=InterruptAgentsArguments,
        description_text="Interrupt selected background child agents, or all of them when no IDs are supplied.",
        prompt_text="Use interrupt_agents only when waiting for a child agent is no longer useful.",
        runner=None,
        async_runner=tool_interrupt_agents,
        read_only=True,
        concurrency_safe=True,
    ),
    WorkspaceTool(
        name="skill",
        input_schema=SkillArguments,
        description_text="Invoke a registered project skill and return its expanded instructions.",
        prompt_text="Use skill when an available Skill matches the user's request; follow its returned instructions in the next response.",
        runner=tool_skill,
        read_only=True,
        concurrency_safe=True,
    ),
)


def tool_definition(name: str) -> WorkspaceTool:
    """按规范名称或废弃别名获取工具定义。"""
    for tool in TOOL_DEFINITIONS:
        if name == tool.name or name in tool.aliases:
            return tool
    raise ValueError(f"unknown tool: {name}")


def validate_tool(context: ToolContext, name: str, args: ToolArgumentsPayload | None) -> dict[str, Any]:
    """校验输入 schema 与工具专属权限，并返回最终可执行输入。"""
    tool = tool_definition(name)
    input_value = tool.parse_input(args)
    permission = tool.check_permissions(input_value, context)
    if permission.behavior != "allow":
        raise ValueError(permission.message)
    updated_input = permission.updated_input or input_value
    return updated_input.model_dump(mode="python")
