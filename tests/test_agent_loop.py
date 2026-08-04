import asyncio

from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano.runtime.agent_loop import QueryEngine


def build_agent(tmp_path, outputs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".nano" / "sessions")
    return AgentRuntime(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
    )


def test_query_engine_runs_same_control_flow_as_nano_ask(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\n", encoding="utf-8")
    runtime = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"hello.txt","start":1,"end":1}}</tool>',
            "<final>Done.</final>",
        ],
    )

    answer = asyncio.run(QueryEngine(runtime).run_async("Inspect hello.txt"))

    assert answer == "Done."
    assert runtime.current_task_state.status == "completed"
    assert runtime.run_store.report_path(runtime.current_task_state.run_id).exists()


def test_runtime_exposes_only_async_answer_entrypoints(tmp_path):
    runtime = build_agent(tmp_path, ["<final>Facade works.</final>"])

    assert not hasattr(runtime, "ask")
    assert not hasattr(QueryEngine, "run")
    assert asyncio.run(runtime.ask_async("Use facade")) == "Facade works."


def test_query_engine_streams_text_deltas(tmp_path):
    runtime = build_agent(tmp_path, [["<final>", "Streamed ", "answer.</final>"]])

    async def collect_events():
        """收集 QueryEngine 直接产出的事件与最终答案。"""
        events = []
        answer = ""
        async for event in QueryEngine(runtime).stream_async("Respond in chunks"):
            events.append(event)
            if event.type == "final":
                answer = str(event.payload["answer"])
        return answer, events

    answer, events = asyncio.run(collect_events())

    assert answer == "Streamed answer."
    assert [event.payload["text"] for event in events if event.type == "text_delta"] == ["<final>", "Streamed ", "answer.</final>"]


def test_parent_can_start_and_await_a_subagent(tmp_path):
    runtime = build_agent(tmp_path, ["<final>Child investigation complete.</final>"])

    async def run_subagent():
        """启动子 agent 后，通过句柄等待其独立事件流的最终结论。"""
        subagent = await runtime.start_subagent({"task": "Inspect README.md", "max_steps": 2})
        return subagent, await subagent.wait()

    subagent, answer = asyncio.run(run_subagent())

    assert answer == "Child investigation complete."
    assert subagent.runtime.current_task_state.status == "completed"
    assert subagent.task.done() is True
