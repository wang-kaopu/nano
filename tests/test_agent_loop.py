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

    answer = QueryEngine(runtime).run("Inspect hello.txt")

    assert answer == "Done."
    assert runtime.current_task_state.status == "completed"
    assert runtime.run_store.report_path(runtime.current_task_state.run_id).exists()


def test_nano_ask_delegates_to_query_engine(tmp_path):
    runtime = build_agent(tmp_path, ["<final>Facade works.</final>"])

    assert runtime.ask("Use facade") == "Facade works."


def test_query_engine_forwards_text_deltas_to_event_callback(tmp_path):
    runtime = build_agent(tmp_path, [["<final>", "Streamed ", "answer.</final>"]])
    events = []

    answer = QueryEngine(runtime).run("Respond in chunks", event_callback=events.append)

    assert answer == "Streamed answer."
    assert [event.payload["text"] for event in events if event.type == "text_delta"] == ["<final>", "Streamed ", "answer.</final>"]
