from nano import FakeModelClient, Nano, SessionStore, WorkspaceContext
from nano.memory.memory import save_memory
from nano.runtime.context_manager import ContextManager


def build_agent(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Nano(
        model_client=FakeModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".nano" / "sessions"),
        approval_policy="auto",
    )


def test_context_manager_injects_memory_system_and_semantically_selected_file(tmp_path):
    agent = build_agent(tmp_path)
    assert agent.memory.memory_dir is not None
    filename = save_memory(
        "CI dashboard",
        "Deployment workflow and dashboard URL.",
        "reference",
        "Deploy through the release workflow, then inspect the CI dashboard.",
        agent.memory.memory_dir,
    )
    agent.side_query = lambda system_prompt, user_prompt: f'{{"selected_memories": ["{filename}"]}}'

    prompt, metadata = ContextManager(agent).build("Where should I check a failed deployment?")

    assert "# Memory System" in prompt
    assert "## Current Memory Index" in prompt
    assert "name: Short memory name" in prompt
    assert prompt.index("Memory:") < prompt.index("Relevant memories:") < prompt.index("Transcript:")
    assert "Deploy through the release workflow" in prompt
    assert metadata["relevant_memory"]["selected_filenames"] == [filename]


def test_context_manager_does_not_surface_a_file_memory_twice_in_one_session(tmp_path):
    agent = build_agent(tmp_path)
    assert agent.memory.memory_dir is not None
    filename = save_memory(
        "User prefers concise replies",
        "Avoid lengthy progress summaries.",
        "user",
        "Keep final responses concise.",
        agent.memory.memory_dir,
    )
    calls = 0

    def side_query(system_prompt, user_prompt):
        nonlocal calls
        calls += 1
        return f'{{"selected_memories": ["{filename}"]}}'

    agent.side_query = side_query
    first_prompt, _ = ContextManager(agent).build("How should you respond?")
    second_prompt, _ = ContextManager(agent).build("How should you respond next?")

    assert "Keep final responses concise." in first_prompt
    assert "Keep final responses concise." not in second_prompt
    assert calls == 1
