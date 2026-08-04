from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano.tools.tool_executor import ToolExecutor, ToolExecutionResult


def build_agent(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".nano" / "sessions")
    return AgentRuntime(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )


def test_tool_executor_returns_content_and_metadata_without_side_channel(tmp_path):
    runtime = build_agent(tmp_path)

    result = ToolExecutor(runtime).execute("read_file", {"path": "README.md", "start": 1, "end": 1})

    assert isinstance(result, ToolExecutionResult)
    assert "# README.md" in result.content
    assert result.metadata["tool_status"] == "ok"
    assert result.metadata["read_only"] is True
    assert result.metadata["workspace_changed"] is False


def test_nano_run_tool_keeps_compatibility_metadata(tmp_path):
    runtime = build_agent(tmp_path)

    content = runtime.run_tool("read_file", {"path": "README.md", "start": 1, "end": 1})

    assert "# README.md" in content
    assert runtime._last_tool_result_metadata["tool_status"] == "ok"
