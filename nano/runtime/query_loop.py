"""处理模型流和工具执行的内层异步查询循环。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from nano.runtime.query_events import QueryEvent
from nano.utils import clip, now


class QueryLoop:
    """执行一条用户请求，直至得到最终答案或达到限制。"""

    def __init__(self, runtime, task_state, user_message):
        """绑定内层循环执行所需的请求级状态。"""
        self.runtime = runtime
        self.task_state = task_state
        self.user_message = str(user_message)
        self.max_attempts = max(runtime.max_steps * 3, runtime.max_steps + 4)

    async def run(self) -> AsyncIterator[QueryEvent]:
        """流式处理模型输出和工具执行，并产出查询进度。"""
        native_tool_call_protocol = str(getattr(self.runtime.model_client, "native_tool_call_protocol", "openai"))
        native_tool_calls = bool(getattr(self.runtime.model_client, "supports_native_tool_calls", False))
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
            prompt, prompt_metadata = self.runtime._build_prompt_and_metadata(self.user_message)
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
                "prompt_cache_retention": "in_memory" if getattr(self.runtime.model_client, "supports_prompt_cache", False) else None,
            }
            if native_tool_calls:
                stream_kwargs.update({"tools": native_tools, "input_items": native_input})
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
