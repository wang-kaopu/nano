import asyncio

from nano import AgentDefinition, FakeModelClient, SessionStore, WorkspaceContext, build_runtime, run_agent
from nano.tools.tool_context import ToolContext


def build_definition(tmp_path, outputs, **kwargs):
    """构造对象化启动 API 所需的最小 agent 配置。"""
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return AgentDefinition(
        model_client=FakeModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".nano" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


def test_run_agent_streams_events_from_object_configuration(tmp_path):
    definition = build_definition(
        tmp_path,
        ["<final>Configured run.</final>"],
        tools=("read_file",),
        instructions="Always answer with the configured result.",
    )

    async def collect():
        return [event async for event in run_agent(
            agent_definition=definition,
            prompt_messages=[{"role": "user", "content": "First instruction"}, "Second instruction"],
            use_exact_tools=True,
            max_turns=2,
        )]

    events = asyncio.run(collect())

    assert events[0].type == "run_started"
    assert events[0].payload["use_exact_tools"] is True
    assert events[0].payload["max_turns"] == 2
    assert events[-1].type == "final"
    assert events[-1].payload["answer"] == "Configured run."
    prompt = definition.model_client.prompts[0]
    assert "Always answer with the configured result." in prompt
    assert "First instruction\n\nSecond instruction" in prompt


def test_run_agent_applies_the_turn_limit(tmp_path):
    definition = build_definition(tmp_path, ["", "<final>Too late.</final>"], max_steps=4)

    async def collect():
        return [event async for event in run_agent(
            agent_definition=definition,
            prompt_messages="Retry once",
            max_turns=1,
        )]

    events = asyncio.run(collect())

    assert events[-1].type == "stopped"
    assert events[-1].payload["reason"] == "turn_limit_reached"
    assert events[-1].payload["answer"] == "Stopped after reaching the turn limit without a final answer."


def test_build_runtime_uses_the_supplied_tool_context_and_exact_tool_setting(tmp_path):
    definition = build_definition(tmp_path, [], tools=("read_file",))
    context = ToolContext(
        root=tmp_path,
        path_resolver=lambda raw_path: tmp_path / raw_path,
        shell_env_provider=lambda: {},
        depth=0,
        max_depth=0,
        run_delegates=lambda specs: "unused",
        interrupt_agents=lambda task_ids: 0,
    )

    runtime = build_runtime(definition, tool_use_context=context, use_exact_tools=True, max_turns=3)

    assert runtime.tool_context() is context
    assert runtime.use_exact_tools is True
    assert runtime.max_turns == 3
    assert set(runtime.tools) == {"read_file"}
