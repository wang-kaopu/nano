"""处理模型流和工具执行的内层异步查询循环。"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

from nano.memory import memory as memorylib
from nano.runtime.query_events import QueryEvent
from nano.runtime.termination import normalize_termination_reason
from nano.utils.text import clip, now


MICROCOMPACT_IDLE_S = 5 * 60
MAX_INVALID_TOOL_CALLS = 3


@dataclass(frozen=True)
class FinalizationResult:
    """描述无工具最终请求的答案有效性和结束状态。"""

    answer: str
    valid: bool
    finish_reason: str
    attempted_tool_call: bool = False
    truncated: bool = False
    error_code: str = ""
    error_message: str = ""
    provider_finish_reason: str = ""


class QueryLoop:
    """执行一条用户请求，直至得到最终答案或达到限制。"""

    def __init__(self, runtime, task_state, user_message, memory_prefetch=None):
        """绑定内层循环执行所需的请求级状态。"""
        self.runtime = runtime
        self.task_state = task_state
        self.user_message = str(user_message)
        self.memory_prefetch = memory_prefetch
        self.injected_memories = []
        self.max_attempts = max(runtime.max_steps * 3, runtime.max_steps + 4)

    def _should_auto_extend(self) -> bool:
        """判断已接近完成的子 agent 是否可获得唯一一次预算扩容。"""
        if self.runtime.depth == 0 or self.task_state.auto_extensions >= 1:
            return False
        if not self.task_state.last_tool_made_progress or self.task_state.duplicate_read_calls >= 2:
            return False
        if self.runtime.max_steps >= 10 or not self.runtime.remaining_file_ranges():
            return False
        if self.runtime.required_targets:
            return True
        return self.runtime.file_coverage_ratio() >= 0.5

    def _apply_auto_extension(self) -> int:
        """为满足条件的子 agent 增加至多两次工具调用额度。"""
        extra_steps = min(2, 10 - self.runtime.max_steps)
        if extra_steps <= 0:
            return 0
        self.runtime.max_steps += extra_steps
        self.task_state.resolved_max_steps = self.runtime.max_steps
        self.task_state.auto_extensions += 1
        return extra_steps

    async def _run_final_only_request(self, instruction: str, native_tool_call_protocol: str, native_input: list[dict[str, Any]], max_tokens: int) -> FinalizationResult:
        """发送一次禁用工具的最终请求，并结构化验证其响应。"""
        self.runtime.record({"role": "system", "content": instruction, "created_at": now()})
        self.runtime.record_conversation({"role": "system", "content": instruction, "created_at": now()})
        self.task_state.record_attempt()
        prompt, _ = self.runtime._build_prompt_and_metadata(self.user_message + "\n\n" + instruction, include_prefix=native_tool_call_protocol != "anthropic", relevant_memories=self.injected_memories)
        stream_kwargs: dict[str, Any] = {}
        if self.runtime.model_client.supports_native_tool_calls:
            content_type = "text" if native_tool_call_protocol == "anthropic" else "input_text"
            final_input = [*native_input, {"role": "user", "content": [{"type": content_type, "text": instruction}]}]
            stream_kwargs["input_items"] = final_input
            if native_tool_call_protocol == "anthropic":
                stream_kwargs["system"] = self.runtime.anthropic_system_blocks()
        raw_parts: list[str] = []
        completion_metadata: dict[str, Any] = {}
        attempted_tool_call = False
        async for event in self.runtime.model_client.stream(prompt, max_tokens, **stream_kwargs):
            if event.type == "text_delta":
                raw_parts.append(event.text)
            elif event.type == "tool_call":
                attempted_tool_call = True
            elif event.type == "completed":
                completion_metadata = dict(event.metadata)
            elif event.type == "error":
                return FinalizationResult(answer="", valid=False, finish_reason="error", error_code="model_error", error_message=str(event.metadata.get("message", "model stream failed")))
        self.runtime.last_completion_metadata = completion_metadata
        termination_reason = normalize_termination_reason(completion_metadata)
        provider_finish_reason = str(completion_metadata.get("provider_finish_reason") or completion_metadata.get("finish_reason") or "")
        raw_answer = "".join(raw_parts).strip()
        kind, payload = self.runtime.parse(raw_answer) if raw_answer else ("final", "")
        answer = str(payload or raw_answer).strip()
        if kind == "tool":
            attempted_tool_call = True
        if attempted_tool_call:
            return FinalizationResult(answer="", valid=False, finish_reason=termination_reason, attempted_tool_call=True, error_code="tool_requested_during_forced_final", provider_finish_reason=provider_finish_reason)
        if not answer:
            return FinalizationResult(answer="", valid=False, finish_reason=termination_reason, error_code="empty_forced_final", provider_finish_reason=provider_finish_reason)
        if termination_reason == "output_limit":
            return FinalizationResult(answer=answer, valid=False, finish_reason=termination_reason, truncated=True, error_code="forced_final_output_limit", provider_finish_reason=provider_finish_reason)
        if termination_reason != "complete":
            return FinalizationResult(answer=answer, valid=False, finish_reason=termination_reason, error_code="invalid_forced_final_termination", provider_finish_reason=provider_finish_reason)
        return FinalizationResult(answer=answer, valid=True, finish_reason=termination_reason, provider_finish_reason=provider_finish_reason)

    async def _force_finalize(self, native_tool_call_protocol: str, native_input: list[dict[str, Any]]) -> FinalizationResult:
        """在工具预算耗尽后，要求模型仅根据已有证据生成最终答案。"""
        instruction = (
            "The tool-call budget is exhausted. Do not request or describe any tool call. "
            "Return the best final answer using only evidence already present in the conversation. "
            "Explicitly report missing required targets."
        )
        return await self._run_final_only_request(instruction, native_tool_call_protocol, native_input, self.runtime.max_final_tokens)

    async def _regenerate_complete_final(self, native_tool_call_protocol: str, native_input: list[dict[str, Any]]) -> FinalizationResult:
        """在普通 Final 被截断后重新生成一份完整且无工具的答案。"""
        instruction = (
            "Your previous final response was truncated by the output-token limit. Regenerate the entire answer from the evidence already present. "
            "Do not continue from the cut-off text. Produce a complete and concise answer. Do not call tools."
        )
        return await self._run_final_only_request(instruction, native_tool_call_protocol, native_input, self.runtime.max_final_tokens)

    def _budget_tool_results_anthropic(self, messages: list[dict[str, Any]]) -> None:
        """按上次请求的上下文利用率裁剪 Anthropic 工具结果副本。"""
        effective_window = self.runtime.effective_window
        utilization = self.runtime.last_input_token_count / effective_window if effective_window else 0
        if utilization < 0.5:
            return
        budget = 15_000 if utilization > 0.7 else 30_000
        for message in messages:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not isinstance(block, dict) or block.get("type") != "tool_result" or not isinstance(block.get("content"), str):
                    continue
                content = block["content"]
                if len(content) <= budget:
                    continue
                artifact_path = self.runtime.persist_tool_result("budgeted_tool_result", content)
                artifact_notice = f"\nFull result persisted: {artifact_path}" if artifact_path else ""
                keep = max(1, (budget - len(artifact_notice) - 96) // 2)
                omitted = len(content) - keep * 2
                block["content"] = f"{content[:keep]}\n\n[... budgeted: {omitted} chars truncated ...]{artifact_notice}\n\n{content[-keep:]}"

    def _anthropic_tool_results(self, messages: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
        """返回按时间排序的 Anthropic 工具结果及其原始工具调用元数据。"""
        tool_uses: dict[str, tuple[str, dict[str, Any]]] = {}
        results: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if message.get("role") == "assistant":
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        input_value = block.get("input")
                        tool_input: dict[str, Any] = input_value if isinstance(input_value, dict) else {}
                        tool_uses[str(block.get("id", ""))] = (
                            str(block.get("name", "")),
                            tool_input,
                        )
                continue
            if message.get("role") != "user":
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result" or not isinstance(block.get("content"), str):
                    continue
                name, args = tool_uses.get(str(block.get("tool_use_id", "")), ("", {}))
                results.append((block, name, args))
        return results

    def _snip_tool_results_anthropic(self, messages: list[dict[str, Any]]) -> None:
        """清理过时的 Anthropic 工具结果，同时保留最近三个结果和所有工具元数据。"""
        results = self._anthropic_tool_results(messages)
        protected_blocks = {id(block) for block, _, _ in results[-3:]}
        latest_read_by_path: dict[str, int] = {}
        for index, (_, name, args) in enumerate(results):
            if name == "read_file" and str(args.get("path", "")).strip():
                latest_read_by_path[str(args["path"])] = index
        for index, (block, name, args) in enumerate(results):
            if id(block) in protected_blocks:
                continue
            if name == "read_file" and str(args.get("path", "")).strip() and latest_read_by_path.get(str(args["path"])) != index:
                block["content"] = "[Result snipped]"
        search_results = [(block, name) for block, name, _ in results if name == "search"]
        for block, _ in search_results[:-3]:
            if id(block) not in protected_blocks:
                block["content"] = "[Result snipped]"

    def _microcompact_anthropic(self, messages: list[dict[str, Any]]) -> None:
        """在 prompt cache 冷启动后清空除最近三个外的所有旧工具结果。"""
        if not self.runtime.last_api_call_time or time.time() - self.runtime.last_api_call_time < MICROCOMPACT_IDLE_S:
            return
        results = self._anthropic_tool_results(messages)
        for block, _, _ in results[:-3]:
            block["content"] = "[Old result cleared]"

    def _prepare_anthropic_tool_results(self, messages: list[dict[str, Any]]) -> None:
        """在 Anthropic API 调用前依次执行预算、Snip 和 Microcompact。"""
        self._budget_tool_results_anthropic(messages)
        utilization = self.runtime.last_input_token_count / self.runtime.effective_window if self.runtime.effective_window else 0
        if utilization > 0.6:
            self._snip_tool_results_anthropic(messages)
        self._microcompact_anthropic(messages)

    async def _auto_compact(self, native_tool_call_protocol: str) -> bool:
        """在上下文接近有效窗口时调用当前 provider 汇总模型侧会话。"""
        if self.runtime.last_input_token_count <= self.runtime.effective_window * 0.85:
            return False
        conversation = self.runtime.session["conversation"]
        minimum_messages = 4 if native_tool_call_protocol == "anthropic" else 5
        if len(conversation) < minimum_messages:
            return False
        print("Context window filling up, compacting conversation...", flush=True)
        last_user_message = conversation[-1] if conversation[-1].get("role") == "user" else None
        messages_to_summarize = conversation[:-1] if last_user_message is not None else conversation
        lines = []
        for message in messages_to_summarize:
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", ""))
            if role == "tool":
                content = f"{message.get('name', 'tool')} {json.dumps(message.get('args', {}), sort_keys=True)}\n{content}"
            lines.append(f"[{role}]\n{content}")
        summary_prompt = (
            "You are a conversation summarizer. Be concise but preserve important details.\n\n"
            "Summarize the conversation so far in a concise paragraph, preserving key decisions, "
            "file paths, tool outcomes, and context needed to continue the work.\n\n"
            "Conversation:\n" + "\n\n".join(lines)
        )
        summary_parts: list[str] = []
        self.runtime.last_api_call_time = time.time()
        async for event in self.runtime.model_client.stream(summary_prompt, 2048, prompt_cache_key=None, prompt_cache_retention=None):
            if event.type == "text_delta":
                summary_parts.append(event.text)
            elif event.type == "error":
                raise RuntimeError(f"autoCompact failed: {event.metadata.get('message', 'model stream failed')}")
        summary_text = "".join(summary_parts).strip() or "No summary available."
        compacted_conversation = [
            {"role": "user", "content": f"[Previous conversation summary]\n{summary_text}", "created_at": now()},
            {"role": "assistant", "content": "Understood. I have the context from our previous conversation. How can I continue helping?", "created_at": now()},
        ]
        if last_user_message is not None:
            compacted_conversation.append(last_user_message)
        self.runtime.session["conversation"] = compacted_conversation
        self.runtime.last_input_token_count = 0
        self.runtime.session["last_input_token_count"] = 0
        self.runtime.session_path = self.runtime.session_store.save(self.runtime.session)
        return True

    async def run(self, native_input_seed: list[dict[str, Any]] | None = None) -> AsyncIterator[QueryEvent]:
        """流式处理模型输出和工具执行，并产出查询进度。"""
        native_tool_call_protocol = self.runtime.model_client.native_tool_call_protocol
        native_tool_calls = self.runtime.model_client.supports_native_tool_calls
        if native_tool_call_protocol == "anthropic":
            native_tools = [
                {
                    "name": tool.name,
                    "description": tool.description(None),
                    "input_schema": tool.input_json_schema,
                }
                for tool in self.runtime.tools.values()
            ]
        else:
            native_tools = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description(None),
                    "parameters": tool.input_json_schema,
                    # 现有带默认值的 Pydantic schema 不满足 strict 模式的全字段必填约束。
                    "strict": False,
                }
                for tool in self.runtime.tools.values()
            ]
        native_input = native_input_seed if native_input_seed is not None else []
        while self.task_state.tool_steps < self.runtime.max_steps and self.task_state.attempts < self.max_attempts and self.task_state.attempts < self.runtime.max_turns:
            try:
                auto_compacted = await self._auto_compact(native_tool_call_protocol)
            except RuntimeError as exc:
                yield QueryEvent("error", {"message": str(exc)})
                return
            if auto_compacted:
                native_input.clear()
            selected_memories = self.runtime.memory.consume_memory_prefetch(self.memory_prefetch)
            if selected_memories:
                self.injected_memories = selected_memories
                if native_tool_calls and native_input:
                    native_input.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": memorylib.format_memories_for_injection(selected_memories)}],
                        }
                    )
            prompt, prompt_metadata = self.runtime._build_prompt_and_metadata(
                self.user_message,
                include_prefix=native_tool_call_protocol != "anthropic",
                relevant_memories=self.injected_memories,
            )
            if native_tool_calls and not native_input:
                content_type = "text" if native_tool_call_protocol == "anthropic" else "input_text"
                native_input.append({"role": "user", "content": [{"type": content_type, "text": prompt}]})
            yield QueryEvent("prompt_built", {"prompt_metadata": prompt_metadata})

            self.task_state.record_attempt()
            yield QueryEvent(
                "model_requested",
                {
                    "attempts": self.task_state.attempts,
                    "tool_steps": self.task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )

            raw_parts: list[str] = []
            tool_calls: list[tuple[dict[str, Any], asyncio.Task | None]] = []
            completion_metadata: dict[str, Any] = {}
            stream_kwargs = {
                "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                "prompt_cache_retention": "in_memory" if self.runtime.model_client.supports_prompt_cache else None,
            }
            if native_tool_calls:
                if native_tool_call_protocol == "anthropic":
                    self._prepare_anthropic_tool_results(native_input)
                    stream_kwargs["system"] = self.runtime.anthropic_system_blocks()
                stream_kwargs.update({"tools": native_tools, "input_items": native_input})
            self.runtime.last_api_call_time = time.time()
            async for event in self.runtime.model_client.stream(prompt, self.runtime.max_new_tokens, **stream_kwargs):
                tool_payload: dict[str, Any] | None = None
                if event.type == "text_delta":
                    raw_parts.append(event.text)
                    yield QueryEvent("text_delta", {"text": event.text})
                    raw = "".join(raw_parts)
                    if not tool_calls and "</tool>" in raw:
                        kind, payload = self.runtime.parse(raw)
                        if kind == "tool":
                            tool_payload = payload
                elif event.type == "tool_call":
                    tool_payload = dict(event.metadata)
                elif event.type == "completed":
                    completion_metadata = dict(event.metadata)
                elif event.type == "error":
                    active_tool_tasks = [task for _, task in tool_calls if task is not None]
                    if active_tool_tasks:
                        # 工具通过安全闸口后不能安全取消，必须等待其完成并保留外部副作用。
                        await asyncio.gather(*active_tool_tasks)
                    yield QueryEvent("error", {"message": event.metadata.get("message", "model stream failed")})
                    return

                if tool_payload is not None:
                    name = str(tool_payload.get("name", ""))
                    args = tool_payload.get("args")
                    if args is None:
                        try:
                            args = json.loads(str(tool_payload.get("arguments", "{}")))
                        except json.JSONDecodeError:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    tool_payload["args"] = args
                    self.runtime.record_conversation(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"name": name, "args": args, "call_id": tool_payload.get("call_id", "")}],
                            "created_at": now(),
                        }
                    )
                    tool = self.runtime.tools.get(name)
                    if tool is None:
                        tool = next((candidate for candidate in self.runtime.tools.values() if name in candidate.aliases), None)
                    try:
                        concurrency_safe = tool is not None and tool.is_concurrency_safe(tool.parse_input(args))
                    except Exception:
                        concurrency_safe = False
                    if concurrency_safe:
                        # 标记为并发安全的工具在参数完整后即可与剩余 SSE 响应抢跑执行。
                        tool_task = asyncio.create_task(self.runtime.tool_executor.execute_async(name, args))
                        self.runtime._active_tool_tasks.add(tool_task)
                        tool_task.add_done_callback(self.runtime._active_tool_tasks.discard)
                        tool_calls.append((tool_payload, tool_task))
                        yield QueryEvent("tool_started", {"name": name, "args": args})
                    else:
                        # 写入、Shell 和委派仍立即登记，但统一在流结束后串行执行。
                        tool_calls.append((tool_payload, None))
                        yield QueryEvent("tool_queued", {"name": name, "args": args})

            self.runtime.last_completion_metadata = completion_metadata
            prompt_metadata.update(completion_metadata)
            self.runtime.last_prompt_metadata = prompt_metadata
            cache_read = int(completion_metadata.get("cache_read_tokens") or 0)
            cache_creation = int(completion_metadata.get("cache_creation_tokens") or 0)
            self.runtime.last_input_token_count = int(completion_metadata.get("input_tokens") or 0) + cache_read + cache_creation + int(completion_metadata.get("output_tokens") or 0)
            self.runtime.session["last_input_token_count"] = self.runtime.last_input_token_count
            self.runtime.session_path = self.runtime.session_store.save(self.runtime.session)
            raw = "".join(raw_parts)

            if tool_calls:
                if native_tool_calls:
                    response_output = completion_metadata.get("response_output", [])
                    if isinstance(response_output, list):
                        native_input.extend(response_output)
                concurrent_tool_tasks = [task for _, task in tool_calls if task is not None]
                concurrent_results = await asyncio.shield(asyncio.gather(*concurrent_tool_tasks)) if concurrent_tool_tasks else []
                concurrent_result_index = 0
                anthropic_tool_results: list[dict[str, str]] = []
                for tool_payload, tool_task in tool_calls:
                    name = str(tool_payload.get("name", ""))
                    args = tool_payload.get("args", {})
                    if tool_task is not None:
                        tool_result = concurrent_results[concurrent_result_index]
                        concurrent_result_index += 1
                    else:
                        tool_task = asyncio.create_task(self.runtime.tool_executor.execute_async(name, args))
                        self.runtime._active_tool_tasks.add(tool_task)
                        tool_task.add_done_callback(self.runtime._active_tool_tasks.discard)
                        yield QueryEvent("tool_started", {"name": name, "args": args})
                        tool_result = await asyncio.shield(tool_task)
                    tool_status = str(tool_result.metadata.get("tool_status", ""))
                    if tool_status == "rejected":
                        self.task_state.record_invalid_tool(name)
                    elif tool_result.metadata.get("counts_as_tool_step", True):
                        self.task_state.record_tool(name)
                    self.task_state.last_tool_made_progress = bool(tool_result.metadata.get("progress_made", True))
                    if name == "list_files" and self.runtime.agent_type == "explorer":
                        self.task_state.explorer_list_files_calls = int(tool_result.metadata.get("explorer_list_files_calls", self.task_state.explorer_list_files_calls))
                    if name == "read_file" and tool_result.metadata.get("duplicate_read"):
                        self.task_state.duplicate_read_calls += 1
                    result = tool_result.content
                    self.runtime.record(
                        {
                            "role": "tool",
                            "name": name,
                            "args": args,
                            "content": result,
                            "created_at": now(),
                        }
                    )
                    self.runtime.record_conversation(
                        {
                            "role": "tool",
                            "name": name,
                            "args": args,
                            "content": result,
                            "created_at": now(),
                        }
                    )
                    if native_tool_calls:
                        if native_tool_call_protocol == "anthropic":
                            anthropic_tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": str(tool_payload.get("call_id", "")),
                                    "content": result,
                                }
                            )
                        else:
                            native_input.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": str(tool_payload.get("call_id", "")),
                                    "output": result,
                                }
                            )
                    yield QueryEvent(
                        "tool_completed",
                        {
                            "name": name,
                            "args": args,
                            "result": clip(result, 500),
                            **dict(tool_result.metadata or {}),
                        },
                    )
                    if tool_result.metadata.get("tool_error_code") == "approval_denied":
                        yield QueryEvent("stopped", {"reason": "approval_denied"})
                        return
                if self.task_state.invalid_tool_calls >= MAX_INVALID_TOOL_CALLS:
                    yield QueryEvent("stopped", {"reason": "invalid_tool_call_limit_reached"})
                    return
                if anthropic_tool_results:
                    native_input.append({"role": "user", "content": anthropic_tool_results})
                yield QueryEvent("next_turn", {"reason": "next_turn"})
                continue

            if native_tool_calls:
                final = raw.strip()
                if not final:
                    yield QueryEvent("error", {"message": "native tool-call response completed without text or a function call"})
                    return
                kind, payload = self.runtime.parse(final)
            else:
                kind, payload = self.runtime.parse(raw)
                if kind == "retry":
                    self.runtime.record({"role": "assistant", "content": payload, "created_at": now()})
                    self.runtime.record_conversation({"role": "assistant", "content": payload, "created_at": now()})
                    yield QueryEvent("retry", {"message": payload})
                    continue
                final = (payload or raw).strip()

            termination_reason = normalize_termination_reason(completion_metadata)
            provider_finish_reason = str(completion_metadata.get("provider_finish_reason") or completion_metadata.get("finish_reason") or "")
            if kind == "tool":
                yield QueryEvent("stopped", {"reason": "forced_final_invalid", "answer": "", "finalization_error_code": "textual_tool_call_in_final", "provider_finish_reason": provider_finish_reason, "termination_reason": termination_reason})
                return
            if termination_reason == "output_limit":
                regeneration = FinalizationResult(answer="", valid=False, finish_reason="output_limit", error_code="final_output_limit", provider_finish_reason=provider_finish_reason)
                for _ in range(self.runtime.max_final_retries):
                    self.task_state.final_regeneration_attempts += 1
                    regeneration = await self._regenerate_complete_final(native_tool_call_protocol, native_input)
                    if regeneration.valid:
                        final = regeneration.answer
                        self.runtime.record({"role": "assistant", "content": final, "created_at": now()})
                        self.runtime.record_conversation({"role": "assistant", "content": final, "created_at": now()})
                        yield QueryEvent("final", {"answer": final, "completion_mode": "output_limit_regenerated", "provider_finish_reason": regeneration.provider_finish_reason, "termination_reason": regeneration.finish_reason})
                        return
                final = regeneration.answer or final
                yield QueryEvent("stopped", {"reason": "output_limit_reached", "answer": final, "finalization_error_code": regeneration.error_code, "provider_finish_reason": regeneration.provider_finish_reason or provider_finish_reason, "termination_reason": regeneration.finish_reason})
                return
            if termination_reason != "complete":
                yield QueryEvent("stopped", {"reason": "forced_final_invalid", "answer": final, "finalization_error_code": "invalid_final_termination", "provider_finish_reason": provider_finish_reason, "termination_reason": termination_reason})
                return
            if self.runtime.required_targets and not self.runtime.required_targets_complete():
                yield QueryEvent("stopped", {"reason": "step_limit_reached", "answer": final, "evidence_complete": False, "missing_targets": self.runtime.missing_required_targets(), "provider_finish_reason": provider_finish_reason, "termination_reason": termination_reason})
                return
            self.runtime.record({"role": "assistant", "content": final, "created_at": now()})
            self.runtime.record_conversation({"role": "assistant", "content": final, "created_at": now()})
            yield QueryEvent("final", {"answer": final, "completion_mode": "normal_final", "provider_finish_reason": provider_finish_reason, "termination_reason": termination_reason})
            return

        if self.task_state.tool_steps >= self.runtime.max_steps:
            if self._should_auto_extend():
                extension = self._apply_auto_extension()
                if extension:
                    notice = f"The child-agent tool budget has been extended by {extension} calls because known unread file ranges remain. Continue from the next unread range; no further extension will be granted."
                    self.runtime.record({"role": "system", "content": notice, "created_at": now()})
                    self.runtime.record_conversation({"role": "system", "content": notice, "created_at": now()})
                    yield QueryEvent("next_turn", {"reason": "auto_extension", "extra_steps": extension})
                    async for event in self.run(native_input):
                        yield event
                    return
            finalization = await self._force_finalize(native_tool_call_protocol, native_input)
            self.task_state.final_answer = finalization.answer
            evidence_complete = self.runtime.required_targets_complete()
            missing_targets = self.runtime.missing_required_targets()
            if evidence_complete and finalization.valid and not finalization.truncated:
                yield QueryEvent("final", {"answer": finalization.answer, "completion_mode": "forced_final", "evidence_complete": True, "provider_finish_reason": finalization.provider_finish_reason, "termination_reason": finalization.finish_reason})
                return
            reason = "output_limit_reached" if finalization.truncated else "step_limit_reached" if not evidence_complete else "forced_final_invalid"
            yield QueryEvent("stopped", {"reason": reason, "answer": finalization.answer, "evidence_complete": evidence_complete, "missing_targets": missing_targets, "finalization_error_code": finalization.error_code, "provider_finish_reason": finalization.provider_finish_reason, "termination_reason": finalization.finish_reason})
        elif self.task_state.attempts >= self.runtime.max_turns:
            yield QueryEvent("stopped", {"reason": "turn_limit_reached"})
        elif self.task_state.attempts >= self.max_attempts and self.task_state.tool_steps < self.runtime.max_steps:
            yield QueryEvent("stopped", {"reason": "retry_limit_reached"})
        else:
            yield QueryEvent("stopped", {"reason": "step_limit_reached"})
