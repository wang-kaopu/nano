"""模型后端适配层。

runtime 只关心一件事：给我一个 prompt，我拿回一段文本。
不同 provider 在 HTTP 接口、响应结构、是否支持 prompt cache 上都有差异，
这些差异都在这里被抹平成统一的 complete() 接口。
"""

import asyncio
import copy
import json
import random
import sys
import time
from typing import Any, AsyncIterator, Iterable, Mapping

import httpx

from nano.runtime.query_events import ModelStreamEvent
from nano.runtime.termination import normalize_termination_reason
from nano.storage.schemas import AnthropicResponseModel, OpenAIResponseModel

OPENAI_COMPATIBLE_USER_AGENT = "nano/0.1"
MAX_RETRIES = 3


def _completion_termination_metadata(provider_finish_reason: str) -> dict[str, str]:
    """返回 provider 原始结束原因及运行时统一结束原因。"""
    return {
        "provider_finish_reason": provider_finish_reason,
        "termination_reason": normalize_termination_reason({"provider_finish_reason": provider_finish_reason}),
    }


def _openai_provider_finish_reason(response_data: Mapping[str, Any]) -> str:
    """从 Responses 或 Chat Completions 兼容响应中提取结束原因。"""
    status = str(response_data.get("status", "")).strip()
    if status == "incomplete":
        details = response_data.get("incomplete_details")
        if isinstance(details, Mapping):
            return str(details.get("reason", "incomplete"))
        return "incomplete"
    choices = response_data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        return str(choices[0].get("finish_reason", status or "unknown"))
    return status or "completed"


class _ProviderHTTPError(RuntimeError):
    """携带 HTTP 状态码的 provider 响应错误，供重试策略分类。"""

    def __init__(self, status: int) -> None:
        """记录需要按状态码处理的 HTTP 失败。"""
        self.status = status
        self.status_code = status
        super().__init__(f"HTTP {status}")


def is_retryable(error: Any) -> bool:
    """判断错误是否属于限流、临时不可用或可恢复网络故障。"""
    status = error.status if isinstance(error, _ProviderHTTPError) else None
    if status in {429, 503, 529}:
        return True
    if isinstance(error, httpx.TimeoutException):
        return True
    message = str(error).lower()
    return "overloaded" in message or "econnreset" in message or "etimedout" in message


def _retry_delay(attempt: int) -> float:
    """计算带抖动的指数退避时间，避免并发客户端同步重试。"""
    return min(1000 * (2**attempt), 30000) / 1000 + random.random()


def _retry_reason(error: Any) -> str:
    """将可重试错误转换为终端可读的简短原因。"""
    if isinstance(error, _ProviderHTTPError):
        return f"HTTP {error.status}"
    return str(error) or "network error"


def _print_retry(attempt: int, max_retries: int, reason: str) -> None:
    """向终端报告即将进行的 provider 请求重试。"""
    print(f"Retrying provider request ({attempt}/{max_retries}) after {reason}.", file=sys.stderr, flush=True)


def _should_retry(error: Any, attempt: int, max_retries: int) -> bool:
    """在尚有预算且错误可恢复时记录原因并允许下一次请求。"""
    if attempt >= max_retries or not is_retryable(error):
        return False
    _print_retry(attempt + 1, max_retries, _retry_reason(error))
    return True


async def _wait_to_retry(error: Any, attempt: int, max_retries: int) -> bool:
    """在异步请求可重试时等待退避时间，并告知调用方是否继续。"""
    if not _should_retry(error, attempt, max_retries):
        return False
    await asyncio.sleep(_retry_delay(attempt))
    return True


def _validate_json_response(body_text: str, model: type[Any]) -> dict[str, Any]:
    """将 provider JSON 响应校验为指定 Pydantic 模型并返回字典。"""
    try:
        return model.model_validate_json(body_text).model_dump(mode="python")
    except ValueError as exc:
        raise RuntimeError("Provider returned invalid JSON response schema") from exc


def _response_text(response: Any) -> str:
    """读取 HTTPX 响应文本或测试替身的响应字节。"""
    if isinstance(response, httpx.Response):
        return response.text
    return response.read().decode("utf-8")


def _response_status_code(response: Any) -> int:
    """返回 HTTPX 响应状态码，测试替身默认视为成功。"""
    return response.status_code if isinstance(response, httpx.Response) else 200


def _request_with_retries(
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str],
    timeout: float,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response:
    """发送 JSON POST 请求，并对可恢复错误执行带抖动的指数退避重试。"""
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            last_error = exc
            if _should_retry(exc, attempt, max_retries):
                time.sleep(_retry_delay(attempt))
                continue
            raise
        error = _ProviderHTTPError(_response_status_code(response))
        if _should_retry(error, attempt, max_retries):
            time.sleep(_retry_delay(attempt))
            continue
        return response
    raise last_error or RuntimeError("HTTP request failed")


class FakeModelClient:
    """为测试和基准提供可预测的模型客户端。"""

    def __init__(self, outputs: Iterable[str | list[str]]) -> None:
        self.model = "fake"
        self.base_url = ""
        self.outputs = list(outputs)
        self.prompts = []
        self.supports_prompt_cache = False
        self.supports_native_tool_calls = False
        self.native_tool_call_protocol = "openai"
        self.last_completion_metadata = {}

    def complete(self, prompt: str, max_new_tokens: int, **kwargs: Any) -> str | list[str]:
        """为同步调用方返回下一条预设的完整响应。"""
        self.prompts.append(prompt)
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)

    async def stream(self, prompt: str, max_new_tokens: int, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        """将下一条预设响应作为确定性的文本流产出。"""
        raw = self.complete(prompt, max_new_tokens, **kwargs)
        chunks = raw if isinstance(raw, list) else [raw]
        for chunk in chunks:
            yield ModelStreamEvent("text_delta", text=str(chunk))
        metadata = {"provider_finish_reason": "completed", "termination_reason": "complete", **dict(self.last_completion_metadata or {})}
        yield ModelStreamEvent("completed", metadata=metadata)



def _normalize_versioned_base_url(base_url: str) -> str:
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _extract_openai_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return data["output_text"]

    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    return text

    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return text

    return ""


def _extract_openai_response_from_sse(body_text: str) -> tuple[str, dict[str, Any]]:
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            if event.get("type") == "response.completed":
                text = _extract_openai_text(response)
                if text:
                    return text, response
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text, last_response or {}
        else:
            text = _extract_openai_text(event)
            if text:
                return text, event
    if deltas:
        return "".join(deltas), last_response or {}
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response), last_response
    return "", {}


def _extract_usage_cache_details(data: dict[str, Any]) -> dict[str, Any]:
    # 把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，
    # 让 runtime/trace/report 不需要关心 provider 细节。
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_read_tokens": cached_tokens,
        "cache_creation_tokens": 0,
        "cache_hit": cached_tokens > 0,
    }


def _with_anthropic_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为最后一个稳定 Anthropic 内容 block 加缓存断点，不修改原始消息。"""
    prepared_messages = copy.deepcopy(messages)
    for message in reversed(prepared_messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if not isinstance(block, dict) or block.get("type") == "thinking":
                continue
            block["cache_control"] = {"type": "ephemeral"}
            return prepared_messages
    return prepared_messages


class OpenAICompatibleModelClient:
    """通过 OpenAI-compatible Responses API 请求模型。"""

    def __init__(self, model: str, base_url: str, api_key: str, temperature: float | None, timeout: float) -> None:
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        # 当前只在明确支持 prompt cache 语义的后端上启用这条链路，
        # 避免对不支持的后端传一个“看起来统一、其实没意义”的伪参数。
        self.supports_prompt_cache = any(host in self.base_url for host in ("openai.com", "right.codes"))
        self.supports_native_tool_calls = True
        self.native_tool_call_protocol = "openai"
        self.last_completion_metadata = {}

    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        input_items: list[dict[str, Any]] | None = None,
    ) -> str:
        """向 OpenAI-compatible `/responses` 接口发起一次模型调用。

        为什么存在：
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，
        更不应该自己去判断 prompt cache 参数要不要带。这个函数把这些后端
        细节都包起来，对上层暴露统一的 `complete()` 行为。

        输入 / 输出：
        - 输入：完整 prompt、最大输出 token，以及可选的 prompt cache 参数
        - 输出：模型最终文本；同时把 usage / 缓存 token 元数据写进
          `self.last_completion_metadata`

        在 agent 链路里的位置：
        它位于 `AgentRuntime.ask_async()` 的模型调用阶段，是稳定前缀缓存复用链路真正
        落到 provider API 的地方。
        """
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "input": input_items or [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        # runtime 传入会话级缓存键，让同一会话的连续请求复用缓存。
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = _request_with_retries(
                self.base_url + "/responses",
                payload,
                headers,
                self.timeout,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not reach the OpenAI-compatible backend.\n"
                f"Base URL: {self.base_url}\n"
                f"Model: {self.model}"
            ) from exc
        status_code = _response_status_code(response)
        if status_code >= 400:
            raise RuntimeError(f"OpenAI-compatible request failed with HTTP {status_code}: {_response_text(response)}")
        body_text = _response_text(response)
        response_headers = getattr(response, "headers", {})
        content_type = response_headers.get("Content-Type", response_headers.get("content-type", ""))

        # 有些兼容后端返回普通 JSON，有些返回 SSE。
        # 这里两种都接住，并尽量统一抽取文本和 usage/cache 元数据。
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            if isinstance(response_data, dict) and response_data:
                # 这些元数据会一路传回 runtime，进入 trace 和 report，
                # 用来观察 prompt cache 是否真的命中。
                self.last_completion_metadata = {
                    "prompt_cache_supported": self.supports_prompt_cache,
                    "prompt_cache_key": prompt_cache_key,
                    "prompt_cache_retention": prompt_cache_retention,
                    **_completion_termination_metadata(_openai_provider_finish_reason(response_data)),
                    **_extract_usage_cache_details(response_data),
                }
            if text:
                return text
            raise RuntimeError("OpenAI-compatible error: could not extract text from event stream response")

        try:
            data = _validate_json_response(body_text, OpenAIResponseModel)
        except RuntimeError as exc:
            raise RuntimeError(
                "OpenAI-compatible error: backend returned an invalid JSON response"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data['error']}")
        self.last_completion_metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            **_completion_termination_metadata(_openai_provider_finish_reason(data)),
            **_extract_usage_cache_details(data),
        }
        return _extract_openai_text(data)

    async def stream(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        input_items: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """将 OpenAI Responses SSE 事件转换为文本增量和完成元数据。"""
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "input": input_items or [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": max_new_tokens,
            "stream": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": OPENAI_COMPATIBLE_USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(MAX_RETRIES + 1):
            emitted_output = False
            saw_delta = False
            try:
                async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                    async with client.stream("POST", self.base_url + "/responses", json=payload, headers=headers) as response:
                        if response.status_code >= 400:
                            error = _ProviderHTTPError(response.status_code)
                            if await _wait_to_retry(error, attempt, MAX_RETRIES):
                                continue
                            yield ModelStreamEvent("error", metadata={"message": f"OpenAI-compatible request failed with HTTP {response.status_code}: {await response.aread()}"})
                            return
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            body = line[len("data:"):].strip()
                            if not body or body == "[DONE]":
                                continue
                            event = json.loads(body)
                            event_type = event.get("type", "")
                            if event_type == "response.output_text.delta":
                                text = event.get("delta")
                                if isinstance(text, str) and text:
                                    saw_delta = True
                                    emitted_output = True
                                    yield ModelStreamEvent("text_delta", text=text)
                            elif event_type in {"response.completed", "response.incomplete"}:
                                response_data = event.get("response") or {}
                                if not saw_delta:
                                    text = _extract_openai_text(response_data)
                                    if text:
                                        emitted_output = True
                                        yield ModelStreamEvent("text_delta", text=text)
                                metadata = {
                                    "prompt_cache_supported": self.supports_prompt_cache,
                                    "prompt_cache_key": prompt_cache_key,
                                    "prompt_cache_retention": prompt_cache_retention,
                                    "response_output": response_data.get("output", []),
                                    **_completion_termination_metadata(_openai_provider_finish_reason(response_data)),
                                    **_extract_usage_cache_details(response_data),
                                }
                                self.last_completion_metadata = metadata
                                yield ModelStreamEvent("completed", metadata=metadata)
                                return
                            elif event_type == "response.output_item.done":
                                item = event.get("item") or {}
                                if item.get("type") == "function_call":
                                    emitted_output = True
                                    yield ModelStreamEvent(
                                        "tool_call",
                                        metadata={
                                            "call_id": str(item.get("call_id", "")),
                                            "name": str(item.get("name", "")),
                                            "arguments": str(item.get("arguments", "{}")),
                                        },
                                    )
                            elif event_type in {"error", "response.failed"}:
                                error = event.get("error") or event.get("response", {}).get("error") or event
                                yield ModelStreamEvent("error", metadata={"message": str(error)})
                                return
                return
            except (httpx.RequestError, json.JSONDecodeError) as exc:
                if not emitted_output and await _wait_to_retry(exc, attempt, MAX_RETRIES):
                    continue
                yield ModelStreamEvent("error", metadata={"message": f"OpenAI-compatible stream failed: {exc}"})
                return


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    for item in data.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text:
                return text
    return ""


class AnthropicCompatibleModelClient:
    """通过 Anthropic-compatible Messages API 请求模型。"""

    def __init__(self, model: str, base_url: str, api_key: str, temperature: float | None, timeout: float) -> None:
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.supports_prompt_cache = True
        self.supports_native_tool_calls = True
        self.native_tool_call_protocol = "anthropic"
        self.last_completion_metadata = {}

    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        input_items: list[dict[str, Any]] | None = None,
        system: list[dict[str, Any]] | None = None,
    ) -> str:
        """请求一条非流式 Anthropic 兼容完成响应。"""
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "messages": _with_anthropic_cache_breakpoints(input_items) if input_items else [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": False}

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        try:
            response = _request_with_retries(
                self.base_url + "/messages",
                payload,
                headers,
                self.timeout,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not reach the Anthropic-compatible backend.\n"
                f"Base URL: {self.base_url}\n"
                f"Model: {self.model}"
            ) from exc
        status_code = _response_status_code(response)
        if status_code >= 400:
            raise RuntimeError(f"Anthropic-compatible request failed with HTTP {status_code}: {_response_text(response)}")
        body_text = _response_text(response)

        try:
            data = _validate_json_response(body_text, AnthropicResponseModel)
        except RuntimeError as exc:
            raise RuntimeError(
                "Anthropic-compatible error: backend returned an invalid JSON response"
            ) from exc
        if data.get("error"):
            raise RuntimeError(f"Anthropic-compatible error: {data['error']}")
        provider_finish_reason = str(data.get("stop_reason", ""))
        self.last_completion_metadata = _completion_termination_metadata(provider_finish_reason)
        text = _extract_anthropic_text(data)
        if text:
            return text
        raise RuntimeError("Anthropic-compatible error: could not extract text from response")

    async def stream(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        input_items: list[dict[str, Any]] | None = None,
        system: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """将 Anthropic Messages SSE 事件转换为标准化文本增量。"""
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _with_anthropic_cache_breakpoints(input_items) if input_items else [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": max_new_tokens,
            "stream": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": False}
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        for attempt in range(MAX_RETRIES + 1):
            content_blocks: dict[int, dict[str, Any]] = {}
            emitted_output = False
            try:
                async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                    async with client.stream("POST", self.base_url + "/messages", json=payload, headers=headers) as response:
                        if response.status_code >= 400:
                            error = _ProviderHTTPError(response.status_code)
                            if await _wait_to_retry(error, attempt, MAX_RETRIES):
                                continue
                            yield ModelStreamEvent("error", metadata={"message": f"Anthropic-compatible request failed with HTTP {response.status_code}: {await response.aread()}"})
                            return
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            body = line[len("data:"):].strip()
                            if not body:
                                continue
                            event = json.loads(body)
                            event_type = event.get("type", "")
                            if event_type == "message_start":
                                message = event.get("message") or {}
                                usage = message.get("usage") or {} if isinstance(message, dict) else {}
                                cache_read_tokens = int(usage.get("cache_read_input_tokens") or 0)
                                self.last_completion_metadata.update(
                                    {
                                        "input_tokens": usage.get("input_tokens"),
                                        "cache_read_tokens": cache_read_tokens,
                                        "cache_creation_tokens": int(usage.get("cache_creation_input_tokens") or 0),
                                        "cached_tokens": cache_read_tokens,
                                        "cache_hit": cache_read_tokens > 0,
                                    }
                                )
                            elif event_type == "content_block_start":
                                index = int(event.get("index", -1))
                                block = event.get("content_block") or {}
                                if block.get("type") == "text":
                                    content_blocks[index] = {"type": "text", "text": str(block.get("text", ""))}
                                elif block.get("type") == "tool_use":
                                    content_blocks[index] = {
                                        "type": "tool_use",
                                        "id": str(block.get("id", "")),
                                        "name": str(block.get("name", "")),
                                        "input": block.get("input") if isinstance(block.get("input"), dict) else {},
                                        "partial_json": "",
                                    }
                            elif event_type == "content_block_delta":
                                index = int(event.get("index", -1))
                                delta = event.get("delta", {})
                                text = delta.get("text")
                                if isinstance(text, str) and text:
                                    block = content_blocks.get(index)
                                    if block is not None and block.get("type") == "text":
                                        block["text"] = str(block.get("text", "")) + text
                                    emitted_output = True
                                    yield ModelStreamEvent("text_delta", text=text)
                                partial_json = delta.get("partial_json")
                                if isinstance(partial_json, str) and index in content_blocks:
                                    content_blocks[index]["partial_json"] = str(content_blocks[index].get("partial_json", "")) + partial_json
                            elif event_type == "content_block_stop":
                                index = int(event.get("index", -1))
                                block = content_blocks.get(index)
                                if block is not None and block.get("type") == "tool_use":
                                    partial_json = str(block.pop("partial_json", ""))
                                    if partial_json:
                                        try:
                                            block["input"] = json.loads(partial_json)
                                        except json.JSONDecodeError:
                                            block["input"] = {}
                                    emitted_output = True
                                    yield ModelStreamEvent(
                                        "tool_call",
                                        metadata={
                                            "call_id": str(block.get("id", "")),
                                            "name": str(block.get("name", "")),
                                            "arguments": json.dumps(block.get("input", {})),
                                        },
                                    )
                            elif event_type == "message_delta":
                                usage = event.get("usage") or {}
                                provider_finish_reason = str(event.get("delta", {}).get("stop_reason", ""))
                                self.last_completion_metadata.update(
                                    {
                                        "output_tokens": usage.get("output_tokens"),
                                        "finish_reason": provider_finish_reason,
                                        **_completion_termination_metadata(provider_finish_reason),
                                    }
                                )
                            elif event_type == "message_stop":
                                ordered_blocks = [content_blocks[index] for index in sorted(content_blocks)]
                                self.last_completion_metadata["response_output"] = [{"role": "assistant", "content": ordered_blocks}]
                                self.last_completion_metadata.setdefault("provider_finish_reason", "unknown")
                                self.last_completion_metadata.setdefault("termination_reason", "unknown")
                                yield ModelStreamEvent("completed", metadata=dict(self.last_completion_metadata))
                                return
                            elif event_type == "error":
                                yield ModelStreamEvent("error", metadata={"message": str(event.get("error") or event)})
                                return
                return
            except (httpx.RequestError, json.JSONDecodeError) as exc:
                if not emitted_output and await _wait_to_retry(exc, attempt, MAX_RETRIES):
                    continue
                yield ModelStreamEvent("error", metadata={"message": f"Anthropic-compatible stream failed: {exc}"})
                return
