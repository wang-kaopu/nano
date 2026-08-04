import asyncio
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import AsyncIterator

from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext, build_runtime
from nano.runtime.agent_loop import QueryEngine
from nano.runtime.query_events import ModelStreamEvent
from nano.tools.tool_executor import ToolExecutionResult


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
    assert runtime.current_task_state is not None
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


def test_rejected_delegate_does_not_consume_the_normal_tool_budget(tmp_path):
    runtime = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"tasks":[{"task":"Read README.md","type":"worker","scope":"."}]}}</tool>',
            "<final>Use explorer for this read-only task.</final>",
        ],
    )

    answer = asyncio.run(QueryEngine(runtime).run_async("Read README.md"))

    assert answer == "Use explorer for this read-only task."
    assert runtime.current_task_state is not None
    assert runtime.current_task_state.tool_steps == 0
    assert runtime.current_task_state.invalid_tool_calls == 1


def test_query_engine_stops_after_three_invalid_tool_calls(tmp_path):
    invalid_worker = '<tool>{"name":"delegate","args":{"tasks":[{"task":"Read README.md","type":"worker","scope":"."}]}}</tool>'
    runtime = build_agent(tmp_path, [invalid_worker, invalid_worker, invalid_worker])

    answer = asyncio.run(QueryEngine(runtime).run_async("Read README.md"))

    assert answer == "Stopped after too many invalid tool calls. Review the latest tool error and choose the corrected action."
    assert runtime.current_task_state is not None
    assert runtime.current_task_state.tool_steps == 0
    assert runtime.current_task_state.invalid_tool_calls == 3
    assert runtime.current_task_state.stop_reason == "invalid_tool_call_limit_reached"


def test_delegate_returns_completed_child_result(tmp_path):
    runtime = build_agent(tmp_path, ["<final>Child investigation complete.</final>"])

    async def launch_subagent():
        """执行批量委派并获取子 agent 的同步结果。"""
        result = json.loads(await runtime.run_delegates([{"task": "Inspect README.md", "type": "explorer", "targets": ["README.md"]}]))
        async_agent_task_id = result["children"][0]["asyncAgentTaskId"]
        task = runtime._async_agent_tasks[async_agent_task_id]
        return result, task

    result, task = asyncio.run(launch_subagent())

    assert task.answer == "Child investigation complete."
    assert task.status == "completed"
    assert task.task.done() is True
    assert result["status"] == "completed"
    assert result["children"][0]["answer"] == "Child investigation complete."


def test_delegate_does_not_reuse_equivalent_task_specs(tmp_path):
    runtime = build_agent(tmp_path, ["<final>Child investigation complete.</final>"])

    async def launch_duplicate_delegates():
        """每次批量委派都创建新的明确子任务集合。"""
        first = json.loads(await runtime.run_delegates([{"task": "Inspect README.md", "type": "explorer", "targets": ["README.md"]}]))
        second = json.loads(await runtime.run_delegates([{"task": "Inspect README.md", "type": "explorer", "targets": ["README.md"]}]))
        return first, second

    first, second = asyncio.run(launch_duplicate_delegates())

    assert first["status"] == second["status"] == "completed"
    assert first["children"][0]["asyncAgentTaskId"] != second["children"][0]["asyncAgentTaskId"]
    assert len(runtime._async_agent_tasks) == 2


def test_parent_cannot_request_next_model_turn_before_delegate_children_finish(tmp_path):
    class ParentAndChildModelClient(FakeModelClient):
        """分别控制父 agent 与子 agent 的模型响应时机。"""

        def __init__(self):
            """初始化父级调用计数与子任务阻塞信号。"""
            super().__init__([])
            self.parent_calls = 0
            self.child_started = asyncio.Event()
            self.release_child = asyncio.Event()

        async def stream(self, prompt: str, max_new_tokens: int, **kwargs) -> AsyncIterator[ModelStreamEvent]:
            """在子 agent 完成前阻塞父 agent 的终态交付。"""
            if "Current user request:\nParent request" in prompt:
                self.parent_calls += 1
                if self.parent_calls == 1:
                    text = '<tool>{"name":"delegate","args":{"tasks":[{"task":"Child request","type":"explorer","targets":["README.md"]}]}}</tool>'
                elif self.parent_calls == 2:
                    text = "<final>Parent finished.</final>"
            else:
                self.child_started.set()
                await self.release_child.wait()
                text = "<final>Child finished.</final>"
            yield ModelStreamEvent("text_delta", text=text)
            yield ModelStreamEvent("completed")

    runtime = build_agent(tmp_path, [])
    model_client = ParentAndChildModelClient()
    runtime.model_client = model_client

    async def run_parent():
        """确认 delegate 未返回前不会发生第二次父级模型请求。"""
        parent_task = asyncio.create_task(runtime.ask_async("Parent request"))
        await model_client.child_started.wait()
        await asyncio.sleep(0)
        assert not parent_task.done()
        assert model_client.parent_calls == 1
        model_client.release_child.set()
        return await parent_task

    assert asyncio.run(run_parent()) == "Parent finished."
    assert model_client.parent_calls == 2
    assert not any(item.get("name") == "async_agent_notification" for item in runtime.session["history"])


def test_worker_delegate_edits_only_its_scoped_worktree(tmp_path):
    source_dir = tmp_path / "nano" / "runtime"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "sample.py"
    source_path.write_text("VALUE = 'before'\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "tests@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Nano Tests"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "test fixture"], cwd=tmp_path, check=True, capture_output=True)
    runtime = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"sample.py","start":1,"end":1}}</tool>',
            '<tool>{"name":"patch_file","args":{"path":"sample.py","old_text":"before","new_text":"after"}}</tool>',
            "<final>Worker change complete.</final>",
        ],
    )

    async def launch_worker():
        """启动 worker 批次并检查隔离工作区。"""
        result = json.loads(await runtime.run_delegates([{"task": "Update the sample", "type": "worker", "scope": "nano/runtime"}]))
        return runtime._async_agent_tasks[result["children"][0]["asyncAgentTaskId"]]

    task = asyncio.run(launch_worker())

    assert task.status == "completed"
    assert task.worktree_path
    assert source_path.read_text(encoding="utf-8") == "VALUE = 'before'\n"
    assert (Path(task.worktree_path) / "nano" / "runtime" / "sample.py").read_text(encoding="utf-8") == "VALUE = 'after'\n"
    subprocess.run(["git", "worktree", "remove", "--force", task.worktree_path], cwd=tmp_path, check=True, capture_output=True)


def test_interrupting_parent_cancels_registered_background_agents(tmp_path):
    class BlockingModelClient(FakeModelClient):
        """保持子 agent 运行，直到父 agent 发出取消请求。"""

        async def stream(self, prompt: str, max_new_tokens: int, **kwargs) -> AsyncIterator[ModelStreamEvent]:
            """等待取消，不产出任何模型事件。"""
            await asyncio.Event().wait()
            if False:
                yield ModelStreamEvent("completed")

    runtime = build_agent(tmp_path, [])
    runtime.model_client = BlockingModelClient([])

    async def launch_and_cancel():
        """创建 explorer 后取消其父级控制器。"""
        task_id = runtime._create_async_agent_task(runtime.resolve_delegate_spec({"task": "Wait", "type": "explorer", "targets": ["README.md"]}))
        task = runtime._async_agent_tasks[task_id]
        await asyncio.sleep(0)
        assert runtime.interrupt_async_agents() == 1
        await asyncio.gather(task.task, return_exceptions=True)
        return task

    task = asyncio.run(launch_and_cancel())

    assert task.status == "stopped"
    assert task.answer == "Interrupted by parent agent."


def test_default_agents_serialize_unsafe_workspace_tools(tmp_path):
    parent = build_agent(tmp_path, [])
    active_count = 0
    max_active_count = 0
    counter_lock = threading.Lock()

    def delayed_execute(name, args):
        """模拟会修改工作区的慢速工具执行。"""
        nonlocal active_count, max_active_count
        with counter_lock:
            active_count += 1
            max_active_count = max(max_active_count, active_count)
        time.sleep(0.05)
        with counter_lock:
            active_count -= 1
        return ToolExecutionResult(content=f"{name} complete", metadata={})

    async def run_default_workers():
        """创建两个 default 子 runtime，并并发执行不可并发工具。"""
        first_definition = await parent._delegate_definition("task-one", parent.resolve_delegate_spec({"task": "one", "type": "default", "requested_max_steps": 2}))
        second_definition = await parent._delegate_definition("task-two", parent.resolve_delegate_spec({"task": "two", "type": "default", "requested_max_steps": 2}))
        first = build_runtime(first_definition)
        second = build_runtime(second_definition)
        first.tool_executor.execute = delayed_execute
        second.tool_executor.execute = delayed_execute
        await asyncio.gather(
            first.tool_executor.execute_async("write_file", {"path": "first.txt", "content": "one"}),
            second.tool_executor.execute_async("write_file", {"path": "second.txt", "content": "two"}),
        )

    asyncio.run(run_default_workers())

    assert max_active_count == 1


def test_explorer_delegate_budget_is_raised_to_cover_target_pages(tmp_path):
    """验证父 agent 的过低建议值会被目标文件分页预算提高。"""
    runtime = build_agent(tmp_path, [])
    (tmp_path / "README.md").write_text("line\n" * 637, encoding="utf-8")

    resolved = runtime.resolve_delegate_spec({"task": "Read README", "type": "explorer", "targets": ["README.md"], "requested_max_steps": 3})

    assert resolved.budget.estimated_reads == 4
    assert resolved.budget.minimum_max_steps == 4
    assert resolved.budget.resolved_max_steps == 4


def test_step_limit_still_requests_one_forced_final_answer(tmp_path):
    """验证 max_steps 只约束工具调用，不占用最后一次总结模型调用。"""
    runtime = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":1}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":2,"end":2}}</tool>',
            '<tool>{"name":"read_file","args":{"path":"README.md","start":3,"end":3}}</tool>',
            '<final>Partial inspection completed; remaining lines were not read.</final>',
        ],
    )
    runtime.max_steps = 3

    answer = asyncio.run(runtime.ask_async("Inspect README"))

    assert answer == "Partial inspection completed; remaining lines were not read."
    assert runtime.current_task_state is not None
    assert runtime.current_task_state.status == "stopped"
    assert runtime.current_task_state.stop_reason == "step_limit_reached"
    assert runtime.current_task_state.tool_steps == 3
    assert runtime.current_task_state.attempts == 4
