"""负责运行、持久化和收尾的外层查询生命周期。"""

import asyncio
import time

from .checkpoint import CHECKPOINT_NONE_STATUS, CHECKPOINT_PARTIAL_STALE_STATUS, CHECKPOINT_WORKSPACE_MISMATCH_STATUS
from .query_loop import QueryLoop
from .task_state import TaskState
from .workspace import clip, now


class QueryEngine:
    """管理请求级异步查询循环之外的运行生命周期。"""

    def __init__(self, runtime):
        """绑定持有会话和运行工件的运行时。"""
        self.runtime = runtime

    def run(self, user_message):
        """供未持有事件循环的同步调用方执行一条查询。"""
        try:
            # 检查当前线程是否已运行 asyncio 事件循环。没有事件循环时会抛出 RuntimeError。
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(user_message))
        raise RuntimeError("Nano.ask() cannot run inside an event loop; await Nano.ask_async() instead")

    async def run_async(self, user_message):
        """执行一条用户请求，并围绕内层 QueryLoop 持久化结果。"""
        runtime = self.runtime
        run_started_at = time.monotonic()
        runtime.memory.set_task_summary(user_message)
        runtime.record({"role": "user", "content": user_message, "created_at": now()})
        runtime.record_conversation({"role": "user", "content": user_message, "created_at": now()})

        task_state = TaskState.create(run_id=runtime.new_run_id(), task_id=runtime.new_task_id(), user_request=user_message)
        task_state.resume_status = runtime.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        runtime.current_task_state = task_state
        runtime.current_run_dir = runtime.run_store.start_run(task_state)
        runtime.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        async for event in QueryLoop(runtime, task_state, user_message).run():
            if event.type == "prompt_built":
                prompt_metadata = event.payload["prompt_metadata"]
                runtime.emit_trace(task_state, "prompt_built", {"prompt_metadata": prompt_metadata})
                if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                    checkpoint = runtime.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                    runtime.run_store.write_task_state(task_state)
                    runtime.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "freshness_mismatch"})
                elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                    runtime.emit_trace(task_state, "runtime_identity_mismatch", {"fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", []))})
                    checkpoint = runtime.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                    runtime.run_store.write_task_state(task_state)
                    runtime.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "workspace_mismatch"})
                if prompt_metadata.get("budget_reductions"):
                    checkpoint = runtime.create_checkpoint(task_state, user_message, trigger="context_reduction")
                    runtime.run_store.write_task_state(task_state)
                    runtime.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "context_reduction"})
            elif event.type == "model_requested":
                runtime.run_store.write_task_state(task_state)
                runtime.emit_trace(task_state, "model_requested", event.payload)
            elif event.type == "tool_completed":
                runtime.run_store.write_task_state(task_state)
                runtime.emit_trace(task_state, "tool_executed", event.payload)
                checkpoint = runtime.create_checkpoint(task_state, user_message, trigger="tool_executed")
                runtime.run_store.write_task_state(task_state)
                runtime.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
            elif event.type == "retry":
                runtime.run_store.write_task_state(task_state)
            elif event.type == "final":
                final = event.payload["answer"]
                runtime.emit_trace(task_state, "model_parsed", {"kind": "final", "completion_metadata": runtime.last_completion_metadata})
                return self._finish_success(task_state, user_message, final, run_started_at)
            elif event.type == "error":
                final = event.payload["message"]
                task_state.stop_model_error(final)
                runtime.record({"role": "assistant", "content": final, "created_at": now()})
                return self._finish_stopped(task_state, user_message, final, run_started_at)
            elif event.type == "stopped":
                if event.payload["reason"] == "retry_limit_reached":
                    final = "Stopped after too many malformed model responses without a valid tool call or final answer."
                    task_state.stop_retry_limit(final)
                else:
                    final = "Stopped after reaching the step limit without a final answer."
                    task_state.stop_step_limit(final)
                runtime.record({"role": "assistant", "content": final, "created_at": now()})
                return self._finish_stopped(task_state, user_message, final, run_started_at)

        raise RuntimeError("QueryLoop ended without a final event")

    def _finish_success(self, task_state, user_message, final, run_started_at):
        """持久化正常完成工件并返回最终答案。"""
        runtime = self.runtime
        task_state.finish_success(final)
        runtime.promote_durable_memory(user_message, final)
        checkpoint = runtime.create_checkpoint(task_state, user_message, trigger="run_finished")
        runtime.run_store.write_task_state(task_state)
        runtime.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "run_finished"})
        runtime.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        runtime.run_store.write_report(task_state, runtime.redact_artifact(runtime.build_report(task_state)))
        return final

    def _finish_stopped(self, task_state, user_message, final, run_started_at):
        """持久化停止或失败工件并返回可见的最终消息。"""
        runtime = self.runtime
        runtime.promote_durable_memory(user_message, final)
        runtime.run_store.write_task_state(task_state)
        checkpoint = runtime.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        runtime.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        runtime.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        runtime.run_store.write_report(task_state, runtime.redact_artifact(runtime.build_report(task_state)))
        return final
