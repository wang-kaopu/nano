from pathlib import Path

from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano.memory.memory import RelevantMemory
from nano.runtime.context_manager import ContextManager


def build_agent(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return AgentRuntime(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".nano" / "sessions"),
        approval_policy="auto",
    )


def test_context_manager_injects_prefetched_file_memory_as_system_reminder(tmp_path):
    runtime = build_agent(tmp_path)
    selected = RelevantMemory(
        path=Path(".nano/projects/demo/memory/reference_ci-dashboard.md"),
        filename="reference_ci-dashboard.md",
        content="Deploy through the release workflow, then inspect the CI dashboard.",
        mtime_ms=0,
        header="Memory (saved less than one hour ago): reference_ci-dashboard.md:",
    )

    prompt, metadata = ContextManager(runtime).build(
        "Where should I check a failed deployment?",
        relevant_memories=[selected],
    )

    assert "# Memory System" in prompt
    assert prompt.index("Memory:") < prompt.index("Relevant memories:") < prompt.index("Transcript:")
    assert "<system-reminder>" in prompt
    assert "Deploy through the release workflow" in prompt
    assert metadata["relevant_memory"]["selected_filenames"] == ["reference_ci-dashboard.md"]


def test_context_manager_never_starts_a_side_query(tmp_path):
    runtime = build_agent(tmp_path)
    runtime.side_query = lambda system_prompt, user_prompt: (_ for _ in ()).throw(AssertionError("side query must be prefetched"))

    prompt, metadata = ContextManager(runtime).build("Where should I check a failed deployment?")

    assert "Relevant memories:\n- none" in prompt
    assert metadata["relevant_memory"]["selected_count"] == 0
