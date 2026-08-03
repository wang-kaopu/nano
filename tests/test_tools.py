from pathlib import Path

from nano.tools.tool import CanUseTool, Tool, ToolProgressData, ToolResult
from nano.tools.tool_context import ToolContext
from nano.tools.tools import ReadFileArguments, build_tool_registry, tool_delegate, tool_json_schema, tool_read_file


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


def test_delegate_uses_context_spawn_without_runtime_import(tmp_path):
    calls = []
    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: Path(tmp_path / raw_path),
        shell_env_provider=lambda: {"PWD": str(tmp_path)},
        depth=0,
        max_depth=1,
        spawn_delegate=lambda args: calls.append(args) or "delegate_result:\nDone",
    )

    result = tool_delegate(context, {"task": "inspect README.md", "max_steps": 2})

    assert result == "delegate_result:\nDone"
    assert calls == [{"task": "inspect README.md", "max_steps": 2}]


def test_build_tool_registry_exposes_tool_contracts(tmp_path):
    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: Path(tmp_path / raw_path),
        shell_env_provider=lambda: {"PWD": str(tmp_path)},
        depth=1,
        max_depth=1,
        spawn_delegate=lambda args: "unused",
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
