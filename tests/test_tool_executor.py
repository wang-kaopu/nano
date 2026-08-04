import json

from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano.runtime.task_state import TaskState
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
    assert json.loads(result.content)["path"] == "README.md"
    assert result.metadata["tool_status"] == "ok"
    assert result.metadata["read_only"] is True
    assert result.metadata["workspace_changed"] is False


def test_nano_run_tool_keeps_compatibility_metadata(tmp_path):
    runtime = build_agent(tmp_path)

    content = runtime.run_tool("read_file", {"path": "README.md", "start": 1, "end": 1})

    assert json.loads(content)["path"] == "README.md"
    assert runtime._last_tool_result_metadata["tool_status"] == "ok"


def test_read_file_artifact_clipping_preserves_pagination_metadata(tmp_path):
    """验证超长文件正文被裁剪后，游标与覆盖范围仍完整可见。"""
    runtime = build_agent(tmp_path)
    (tmp_path / "README.md").write_text("x" * 100 + "\n", encoding="utf-8")
    state = TaskState.create(task_id="task_read", user_request="Read file")
    runtime.current_task_state = state
    runtime.current_run_dir = runtime.run_store.start_run(state)
    read_file_tool = runtime.tools["read_file"]
    original_limit = read_file_tool.max_result_size_chars
    read_file_tool.max_result_size_chars = 80
    try:
        payload = json.loads(ToolExecutor(runtime).execute("read_file", {"path": "README.md", "start": 1, "end": 1}).content)
    finally:
        read_file_tool.max_result_size_chars = original_limit

    assert payload["contentTruncated"] is True
    assert payload["resultArtifactPath"]
    assert payload["totalLines"] == 1
    assert "coveredRanges" in payload
