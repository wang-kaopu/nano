import json

from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano.runtime.task_state import TaskState
from nano.tools.tool_executor import ToolExecutor, ToolExecutionResult


def build_agent(tmp_path, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".nano" / "sessions")
    return AgentRuntime(
        model_client=FakeModelClient([]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
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


def test_explorer_list_files_uses_a_separate_five_call_quota(tmp_path):
    """验证 explorer 的目录枚举不消耗工具步骤且独立限制为五次。"""
    runtime = build_agent(tmp_path, agent_type="explorer")
    executor = ToolExecutor(runtime)

    for call_number in range(1, 6):
        result = executor.execute("list_files", {"path": "."})
        assert result.metadata["tool_status"] == "ok"
        assert result.metadata["counts_as_tool_step"] is False
        assert result.metadata["explorer_list_files_calls"] == call_number
        assert result.metadata["explorer_list_files_limit"] == 5

    rejected = executor.execute("list_files", {"path": "."})

    assert rejected.metadata["tool_status"] == "rejected"
    assert rejected.metadata["tool_error_code"] == "explorer_list_files_limit_reached"
    assert rejected.metadata["counts_as_tool_step"] is False


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


def test_run_shell_delegates_to_injected_executor_after_permission_check(tmp_path):
    calls = []

    def sandbox_executor(command, timeout):
        """记录隔离 shell 调用，并返回标准化执行结果。"""
        calls.append((command, timeout))
        return "exit_code: 0\nstdout:\nsandbox\nstderr:\n(no output)"

    runtime = build_agent(tmp_path, shell_executor=sandbox_executor)
    result = ToolExecutor(runtime).execute("run_shell", {"command": "git status", "timeout": 7})

    assert result.metadata["tool_status"] == "ok"
    assert calls == [("git status", 7)]
    assert "sandbox" in result.content


def test_invalid_shell_history_does_not_crash_followup_tool_call(tmp_path):
    runtime = build_agent(tmp_path)
    runtime.session["history"] = [
        {"role": "tool", "name": "run_shell", "args": {"command": "git status", "timeout": 600}, "content": "error"},
        {"role": "tool", "name": "run_shell", "args": {"command": "git status", "timeout": 600}, "content": "error"},
    ]

    assert runtime.repeated_tool_call("run_shell", {"command": "git status", "timeout": 20}) is False
