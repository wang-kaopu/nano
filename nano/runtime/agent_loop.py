"""负责运行、持久化和收尾的外层查询生命周期。"""

import asyncio
import time

from nano.runtime.checkpoint import CHECKPOINT_NONE_STATUS, CHECKPOINT_PARTIAL_STALE_STATUS, CHECKPOINT_WORKSPACE_MISMATCH_STATUS
from nano.runtime.query_events import QueryEvent
from nano.runtime.query_loop import QueryLoop
from nano.runtime.runtime import AgentRuntime
from nano.runtime.task_state import TaskState
from nano.utils.text import clip, now


class QueryEngine:
    """管理请求级异步查询循环之外的运行生命周期。"""

    def __init__(self, runtime: AgentRuntime) -> None:
        """绑定持有会话和运行工件的运行时。"""
        self.runtime = runtime

    async def run_async(self, user_message: str) -> str:
        """执行一条用户请求、持久化运行工件并返回最终答案。"""
        final_answer = None
        async for event in self.stream_async(user_message):
            if event.type in {"final", "error", "stopped"}:
                final_answer = str(event.payload["answer"])
        if final_answer is not None:
            return final_answer
        raise RuntimeError("QueryEngine stream ended without a terminal event")

    async def stream_async(self, user_message: str):
        """以事件流方式执行一条请求，并在终止事件产生前完成工件持久化。"""
        runtime = self.runtime
        run_started_at = time.monotonic()
        runtime.memory.set_task_summary(user_message)
        runtime.record({"role": "user", "content": user_message, "created_at": now()})
        runtime.record_conversation({"role": "user", "content": user_message, "created_at": now()})
        memory_prefetch = runtime.start_memory_prefetch(user_message)

        task_state = TaskState.create(run_id=runtime.new_run_id(), task_id=runtime.new_task_id(), user_request=user_message)
        task_state.initial_max_steps = runtime.max_steps
        task_state.resolved_max_steps = runtime.max_steps
        task_state.resume_status = runtime.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        runtime.current_task_state = task_state
        runtime.current_run_dir = runtime.run_store.start_run(task_state)
        runtime._current_query_task = asyncio.current_task()
        runtime.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )
        yield QueryEvent(
            "run_started",
            {
                "task_id": task_state.task_id,
                "run_id": task_state.run_id,
                "use_exact_tools": runtime.use_exact_tools,
                "max_turns": runtime.max_turns,
            },
        )

        try:
            async for event in QueryLoop(runtime, task_state, user_message, memory_prefetch).run():
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
                elif event.type in {"final", "error", "stopped"}:
                    await runtime.wait_for_async_agents()
                    if event.type == "final":
                        final = event.payload["answer"]
                        task_state.evidence_complete = bool(event.payload.get("evidence_complete", runtime.required_targets_complete()))
                        task_state.missing_targets = list(event.payload.get("missing_targets", runtime.missing_required_targets()))
                        task_state.completion_mode = str(event.payload.get("completion_mode", "normal_final"))
                        task_state.provider_finish_reason = str(event.payload.get("provider_finish_reason", ""))
                        task_state.termination_reason = str(event.payload.get("termination_reason", ""))
                        task_state.finalization_error_code = str(event.payload.get("finalization_error_code", ""))
                        runtime.emit_trace(task_state, "model_parsed", {"kind": "final", "completion_metadata": runtime.last_completion_metadata})
                        self._finish_success(task_state, user_message, final, run_started_at)
                        yield event
                        return
                    if event.type == "error":
                        final = event.payload["message"]
                        task_state.stop_model_error(final)
                        runtime.record({"role": "assistant", "content": final, "created_at": now()})
                        self._finish_stopped(task_state, user_message, final, run_started_at)
                        yield QueryEvent("error", {"message": final, "answer": final})
                        return
                    if event.type == "stopped":
                        partial_answer = str(event.payload.get("answer", "")).strip()
                        task_state.evidence_complete = bool(event.payload.get("evidence_complete", False))
                        task_state.missing_targets = list(event.payload.get("missing_targets", []))
                        task_state.completion_mode = str(event.payload.get("completion_mode", ""))
                        task_state.provider_finish_reason = str(event.payload.get("provider_finish_reason", ""))
                        task_state.termination_reason = str(event.payload.get("termination_reason", ""))
                        task_state.finalization_error_code = str(event.payload.get("finalization_error_code", ""))
                        if event.payload["reason"] == "retry_limit_reached":
                            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
                            task_state.stop_retry_limit(final)
                        elif event.payload["reason"] == "invalid_tool_call_limit_reached":
                            final = "Stopped after too many invalid tool calls. Review the latest tool error and choose the corrected action."
                            task_state.stop_invalid_tool_call_limit(final)
                        elif event.payload["reason"] == "turn_limit_reached":
                            final = "Stopped after reaching the turn limit without a final answer."
                            task_state.stop_turn_limit(final)
                        elif event.payload["reason"] == "approval_denied":
                            final = "Operation was not executed because approval was denied."
                            task_state.stop_approval_denied(final)
                        elif event.payload["reason"] == "output_limit_reached":
                            final = partial_answer or "The model output reached its token limit before completing the response."
                            task_state.stop_output_limit(final)
                        elif event.payload["reason"] == "forced_final_invalid":
                            final = partial_answer or "The model attempted to call a tool or returned an invalid response during the final-only phase."
                            task_state.stop_forced_final_invalid(final)
                        else:
                            final = partial_answer or "Stopped after reaching the step limit without a final answer."
                            task_state.stop_step_limit(final)
                        runtime.record({"role": "assistant", "content": final, "created_at": now()})
                        self._finish_stopped(task_state, user_message, final, run_started_at)
                        yield QueryEvent("stopped", {"reason": task_state.stop_reason, "answer": final, "evidence_complete": task_state.evidence_complete, "missing_targets": task_state.missing_targets, "finalization_error_code": str(event.payload.get("finalization_error_code", "")), "completion_mode": task_state.completion_mode, "provider_finish_reason": task_state.provider_finish_reason, "termination_reason": task_state.termination_reason})
                        return
                yield event
        except asyncio.CancelledError:
            runtime.interrupt_async_agents()
            active_tool_tasks = list(runtime._active_tool_tasks)
            if active_tool_tasks:
                await asyncio.shield(asyncio.gather(*active_tool_tasks, return_exceptions=True))
            final = "Interrupted by user."
            task_state.stop_user_interrupted(final)
            runtime.record({"role": "assistant", "content": final, "created_at": now()})
            self._finish_stopped(task_state, user_message, final, run_started_at)
            yield QueryEvent("stopped", {"reason": task_state.stop_reason, "answer": final})
            return
        finally:
            if memory_prefetch is not None and not memory_prefetch.settled:
                memory_prefetch.task.cancel()
            runtime._current_query_task = None

        raise RuntimeError("QueryLoop ended without a final event")

    def _finish_success(self, task_state: TaskState, user_message: str, final: str, run_started_at: float) -> str:
        """持久化正常完成工件并返回最终答案。"""
        runtime = self.runtime
        task_state.finish_success(final)
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

    def _finish_stopped(self, task_state: TaskState, user_message: str, final: str, run_started_at: float) -> str:
        """持久化停止或失败工件并返回可见的最终消息。"""
        runtime = self.runtime
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
