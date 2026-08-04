import asyncio
from pathlib import Path

from nano.tools.tool import CanUseTool, Tool, ToolProgressData, ToolResult
from nano.tools.tool_context import ToolContext
from nano.tools.tools import InterruptAgentsArguments, ReadFileArguments, build_tool_registry, tool_definition, tool_delegate, tool_interrupt_agents, tool_json_schema, tool_read_file


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
        spawn_delegate=lambda args: "unused",
        interrupt_agents=lambda task_ids: 0,
    )

    result = tool_read_file(context, {"path": "sample.txt", "start": 1, "end": 1})

    assert "# sample.txt" in result
    assert "alpha" in result


def test_tool_base_class_uses_safe_default_semantics():
    tool = DefaultTool(name="default", input_schema=ReadFileArguments)
    input_value = tool.parse_input({"path": "README.md"})

    assert tool.is_concurrency_safe(input_value) is False
    assert tool.is_read_only(input_value) is False
    assert tool.is_destructive(input_value) is False
    assert tool.check_permissions(input_value, None).behavior == "allow"
    assert tool.check_permissions(input_value, None).updated_input == input_value


def test_delegate_launches_through_context_without_runtime_import(tmp_path):
    calls = []

    async def spawn_delegate(args):
        """记录委派请求，并模拟异步任务登记结果。"""
        calls.append(args)
        return '{"status":"async_launched","asyncAgentTaskId":"task_test"}'

    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: Path(tmp_path / raw_path),
        shell_env_provider=lambda: {"PWD": str(tmp_path)},
        depth=0,
        max_depth=1,
        spawn_delegate=spawn_delegate,
        interrupt_agents=lambda task_ids: 0,
    )

    result = asyncio.run(tool_delegate(context, {"task": "inspect README.md", "max_steps": 2}))

    assert result == '{"status":"async_launched","asyncAgentTaskId":"task_test"}'
    assert calls == [{"task": "inspect README.md", "max_steps": 2}]


def test_delegate_prompt_makes_read_only_work_an_explorer_task():
    delegate = tool_definition("delegate")
    prompt = delegate.prompt(None)
    schema = delegate.input_json_schema

    assert "type=explorer is mandatory for reading files" in prompt
    assert "never '.', a file path, or a task sentence" in prompt
    assert "use 4 when a file may need multiple reads" in prompt
    assert "Do not use max_steps 1 or 2" in prompt
    assert "your next response must be final" in prompt
    assert "Never poll child status with run_shell, sleep, ls, find, cat, Python" in prompt
    assert delegate.concurrency_safe is False
    assert "reading/searching/reviewing" in schema["properties"]["type"]["description"]


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
        spawn_delegate=lambda args: "unused",
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
        spawn_delegate=lambda args: "unused",
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
