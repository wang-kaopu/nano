"""处理模型流和工具执行的内层异步查询循环。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from nano.runtime.query_events import QueryEvent
from nano.utils import clip, now


MICROCOMPACT_IDLE_S = 5 * 60


class QueryLoop:
    """执行一条用户请求，直至得到最终答案或达到限制。"""

    def __init__(self, runtime, task_state, user_message):
        """绑定内层循环执行所需的请求级状态。"""
        self.runtime = runtime
        self.task_state = task_state
        self.user_message = str(user_message)
        self.max_attempts = max(runtime.max_steps * 3, runtime.max_steps + 4)

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
                        tool_uses[str(block.get("id", ""))] = (
                            str(block.get("name", "")),
                            block.get("input") if isinstance(block.get("input"), dict) else {},
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

    async def run(self) -> AsyncIterator[QueryEvent]:
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
        native_input: list[dict[str, Any]] = []
        while self.task_state.tool_steps < self.runtime.max_steps and self.task_state.attempts < self.max_attempts:
            try:
                auto_compacted = await self._auto_compact(native_tool_call_protocol)
            except RuntimeError as exc:
                yield QueryEvent("error", {"message": str(exc)})
                return
            if auto_compacted:
                native_input.clear()
            prompt, prompt_metadata = self.runtime._build_prompt_and_metadata(self.user_message, include_prefix=native_tool_call_protocol != "anthropic")
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
                    self.task_state.record_tool(name)
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
                if anthropic_tool_results:
                    native_input.append({"role": "user", "content": anthropic_tool_results})
                yield QueryEvent("next_turn", {"reason": "next_turn"})
                continue

            if native_tool_calls:
                final = raw.strip()
                if not final:
                    yield QueryEvent("error", {"message": "native tool-call response completed without text or a function call"})
                    return
                self.runtime.record({"role": "assistant", "content": final, "created_at": now()})
                self.runtime.record_conversation({"role": "assistant", "content": final, "created_at": now()})
                yield QueryEvent("final", {"answer": final})
                return

            kind, payload = self.runtime.parse(raw)
            if kind == "retry":
                self.runtime.record({"role": "assistant", "content": payload, "created_at": now()})
                self.runtime.record_conversation({"role": "assistant", "content": payload, "created_at": now()})
                yield QueryEvent("retry", {"message": payload})
                continue

            final = (payload or raw).strip()
            self.runtime.record({"role": "assistant", "content": final, "created_at": now()})
            self.runtime.record_conversation({"role": "assistant", "content": final, "created_at": now()})
            yield QueryEvent("final", {"answer": final})
            return

        if self.task_state.attempts >= self.max_attempts and self.task_state.tool_steps < self.runtime.max_steps:
            yield QueryEvent("stopped", {"reason": "retry_limit_reached"})
        else:
            yield QueryEvent("stopped", {"reason": "step_limit_reached"})
