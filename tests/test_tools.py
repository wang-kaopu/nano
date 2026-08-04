import asyncio
import json
from pathlib import Path

from nano.tools.tool import CanUseTool, Tool, ToolProgressData, ToolResult
from nano.tools.tool_context import ToolContext
from nano.tools.tools import DelegateArguments, InterruptAgentsArguments, ReadFileArguments, build_tool_registry, tool_definition, tool_delegate, tool_interrupt_agents, tool_json_schema, tool_read_file


class DefaultTool(Tool[ReadFileArguments, str, ToolProgressData]):
    """验证 Tool 基类默认执行语义。"""

    def description(self, input_value, options=None):
        """返回测试工具描述。"""
        return "default tool"

    def prompt(self, options=None):
        """返回测试工具提示。"""
        return "use default tool"

    def call(self, args, context, can_use_tool: CanUseTool, parent_message, on_progress=None):
        """返回固定测试结果。"""
        return ToolResult(output="ok", content="ok")


def test_tool_context_supports_file_tools_without_full_nano(tmp_path):
    (tmp_path / "sample.txt").write_text("alpha\n", encoding="utf-8")
    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: (tmp_path / raw_path).resolve(),
        shell_env_provider=lambda: {"PWD": str(tmp_path)},
        depth=0,
        max_depth=1,
        run_delegates=lambda specs: "unused",
        interrupt_agents=lambda task_ids: 0,
    )

    result = tool_read_file(context, {"path": "sample.txt", "start": 1, "end": 1})

    payload = json.loads(result)
    assert payload["path"] == "sample.txt"
    assert payload["totalLines"] == 1
    assert "alpha" in payload["content"]


def test_tool_base_class_uses_safe_default_semantics():
    tool = DefaultTool(name="default", input_schema=ReadFileArguments)
    input_value = tool.parse_input({"path": "README.md"})

    assert tool.is_concurrency_safe(input_value) is False
    assert tool.is_read_only(input_value) is False
    assert tool.is_destructive(input_value) is False
    assert tool.check_permissions(input_value, None).behavior == "allow"
    assert tool.check_permissions(input_value, None).updated_input == input_value


def test_delegate_runs_task_batch_through_context_without_runtime_import(tmp_path):
    calls = []

    async def run_delegates(specs):
        """记录委派批次，并模拟全部子任务已结束的结果。"""
        calls.append(specs)
        return '{"status":"completed","children":[]}'

    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: Path(tmp_path / raw_path),
        shell_env_provider=lambda: {"PWD": str(tmp_path)},
        depth=0,
        max_depth=1,
        run_delegates=run_delegates,
        interrupt_agents=lambda task_ids: 0,
    )

    result = asyncio.run(tool_delegate(context, {"tasks": [{"task": "inspect README.md", "type": "explorer", "targets": ["README.md"]}]}))

    assert result == '{"status":"completed","children":[]}'
    assert calls == [[{"task": "inspect README.md", "type": "explorer", "targets": ["README.md"]}]]


def test_delegate_prompt_makes_read_only_work_an_explorer_task():
    delegate = tool_definition("delegate")
    prompt = delegate.prompt(None)
    schema = delegate.input_json_schema

    assert "Submit every currently planned child task in one delegate call" in prompt
    assert "returns only after each child" in prompt
    assert "evidenceComplete" in prompt
    assert "missingTargets" in prompt
    assert delegate.concurrency_safe is False
    assert "tasks" in schema["properties"]
    assert "task" not in schema["properties"]
    assert DelegateArguments.model_validate({"tasks": [{"task": "inspect", "type": "explorer", "targets": ["README.md"]}]}).tasks[0].task == "inspect"


def test_interrupt_agents_uses_context_task_cancellation(tmp_path):
    calls = []

    def interrupt_agents(task_ids):
        """记录取消目标，并模拟成功取消一个后台任务。"""
        calls.append(task_ids)
        return 1

    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: Path(tmp_path / raw_path),
        shell_env_provider=lambda: {"PWD": str(tmp_path)},
        depth=0,
        max_depth=1,
        run_delegates=lambda specs: "unused",
        interrupt_agents=interrupt_agents,
    )

    result = asyncio.run(tool_interrupt_agents(context, {"async_agent_task_ids": ["task-one"]}))

    assert result == '{"status": "interrupt_requested", "cancelledCount": 1, "asyncAgentTaskIds": ["task-one"]}'
    assert calls == [["task-one"]]
    assert InterruptAgentsArguments.model_validate({}).async_agent_task_ids == []


def test_build_tool_registry_exposes_tool_contracts(tmp_path):
    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: Path(tmp_path / raw_path),
        shell_env_provider=lambda: {"PWD": str(tmp_path)},
        depth=1,
        max_depth=1,
        run_delegates=lambda specs: "unused",
        interrupt_agents=lambda task_ids: 0,
    )

    tools = build_tool_registry(context)

    assert "read_file" in tools
    assert "delegate" not in tools
    read_file = tools["read_file"]
    assert isinstance(read_file, Tool)
    assert read_file.max_result_size_chars == 5000
    assert read_file.is_concurrency_safe(read_file.parse_input({"path": "README.md"})) is True
    assert read_file.is_read_only(read_file.parse_input({"path": "README.md"})) is True
    permission = read_file.check_permissions(read_file.parse_input({"path": "README.md"}), context)
    assert permission.behavior == "deny"
    assert permission.updated_input is not None


def test_tool_schema_is_generated_from_pydantic_arguments():
    schema = tool_json_schema("run_shell")

    assert schema["properties"]["timeout"]["default"] == 20
    assert schema["properties"]["timeout"]["maximum"] == 120
    assert "command" in schema["required"]


def test_read_file_returns_cursor_and_rejects_repeated_ranges(tmp_path):
    """验证分页状态由运行时维护，重复读取遵循三次防护规则。"""
    (tmp_path / "sample.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    context = ToolContext(root=tmp_path, path_resolver=lambda raw_path: (tmp_path / raw_path).resolve(), shell_env_provider=lambda: {}, depth=0, max_depth=1, run_delegates=lambda specs: "unused", interrupt_agents=lambda task_ids: 0)

    first = json.loads(tool_read_file(context, {"path": "sample.txt", "start": 1, "end": 2}))
    next_page = json.loads(tool_read_file(context, {"path": "sample.txt", "cursor": first["nextCursor"]}))
    duplicate_one = json.loads(tool_read_file(context, {"path": "sample.txt", "start": 1, "end": 2}))
    duplicate_two = json.loads(tool_read_file(context, {"path": "sample.txt", "start": 1, "end": 2}))
    duplicate_three = json.loads(tool_read_file(context, {"path": "sample.txt", "start": 1, "end": 2}))

    assert first["totalLines"] == 3
    assert first["hasMore"] is True
    assert next_page["returnedRange"] == {"start": 3, "end": 3}
    assert duplicate_one["status"] == duplicate_two["status"] == "already_covered"
    assert duplicate_two["duplicateReadCalls"] == 2
    assert duplicate_three["errorCode"] == "repeated_read_range"


def test_read_file_rejects_stale_cursor_and_delegate_rejects_old_protocol(tmp_path):
    """验证游标与文件版本绑定，且旧 delegate 参数不会被兼容接受。"""
    path = tmp_path / "sample.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    context = ToolContext(root=tmp_path, path_resolver=lambda raw_path: (tmp_path / raw_path).resolve(), shell_env_provider=lambda: {}, depth=0, max_depth=1, run_delegates=lambda specs: "unused", interrupt_agents=lambda task_ids: 0)
    page = json.loads(tool_read_file(context, {"path": "sample.txt", "start": 1, "end": 1}))
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    assert json.loads(tool_read_file(context, {"path": "sample.txt", "cursor": page["nextCursor"]}))["status"] == "cursor_stale"
    try:
        DelegateArguments.model_validate({"task": "old protocol"})
    except Exception:
        pass
    else:
        raise AssertionError("old delegate protocol was unexpectedly accepted")
