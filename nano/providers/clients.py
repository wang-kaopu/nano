"""模型后端适配层。

runtime 只关心一件事：给我一个 prompt，我拿回一段文本。
不同 provider 在 HTTP 接口、响应结构、是否支持 prompt cache 上都有差异，
这些差异都在这里被抹平成统一的 complete() 接口。
"""

import json
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

import httpx

from ..query_events import ModelStreamEvent
from ..schemas import AnthropicResponseModel, OllamaResponseModel, OpenAIResponseModel
from ..types import JsonObject

OPENAI_COMPATIBLE_USER_AGENT = "nano/0.1"


def _validate_json_response(body_text: str, model: type[Any]) -> JsonObject:
    """将 provider JSON 响应校验为指定 Pydantic 模型并返回字典。"""
    try:
        return model.model_validate_json(body_text).model_dump(mode="python")
    except ValueError as exc:
        raise RuntimeError("Provider returned invalid JSON response schema") from exc


def _response_text(response: Any) -> str:
    """读取 HTTPX 响应文本，并保留测试替身的兼容读取方式。"""
    text = getattr(response, "text", None)
    if text is not None:
        return text
    return response.read().decode("utf-8")


def _response_status_code(response: Any) -> int:
    """返回响应状态码，缺少该属性的轻量测试替身视为成功。"""
    return int(getattr(response, "status_code", 200))


def _request_with_retries(
    url: str,
    payload: JsonObject,
    headers: Mapping[str, str],
    timeout: float,
    attempts: int = 1,
) -> httpx.Response:
    """发送 JSON POST 请求，并对网络错误和 5xx 响应执行有限重试。"""
    last_error = None
    for attempt in range(attempts):
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                response = client.post(url, json=payload, headers=headers)
        except httpx.RequestError as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
        if _response_status_code(response) >= 500 and attempt < attempts - 1:
            time.sleep(0.5 * (attempt + 1))
            continue
        return response
    raise last_error or RuntimeError("HTTP request failed")


class FakeModelClient:
    """为测试和基准提供可预测的模型客户端。"""

    def __init__(self, outputs: Iterable[str | list[str]]) -> None:
        self.outputs = list(outputs)
        self.prompts = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt: str, max_new_tokens: int, **kwargs: Any) -> str | list[str]:
        """为同步调用方返回下一条预设的完整响应。"""
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)

    async def stream(self, prompt: str, max_new_tokens: int, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        """将下一条预设响应作为确定性的文本流产出。"""
        raw = self.complete(prompt, max_new_tokens, **kwargs)
        chunks = raw if isinstance(raw, list) else [raw]
        for chunk in chunks:
            yield ModelStreamEvent("text_delta", text=str(chunk))
        yield ModelStreamEvent("completed", metadata=dict(self.last_completion_metadata or {}))


class OllamaModelClient:
    """通过 Ollama generate API 请求模型。"""

    def __init__(self, model: str, host: str, temperature: float | None, top_p: float | None, timeout: float) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt: str, max_new_tokens: int, **kwargs: Any) -> str:
        """请求一条非流式 Ollama 完成响应。"""
        # Ollama 当前不支持我们这里接入的 prompt cache 语义，
        # 所以 runtime 传下来的缓存参数会被忽略。
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = _request_with_retries(
                self.host + "/api/generate",
                payload,
                headers,
                self.timeout,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc
        if _response_status_code(response) >= 400:
            raise RuntimeError(f"Ollama request failed with HTTP {_response_status_code(response)}: {_response_text(response)}")
        data = _validate_json_response(_response_text(response), OllamaResponseModel)

        if data.get("error"):
            raise RuntimeError(f"Ollama error: {data['error']}")
        return data.get("response", "")

    async def stream(self, prompt: str, max_new_tokens: int, **kwargs: Any) -> AsyncIterator[ModelStreamEvent]:
        """将 Ollama generate 分块转换为标准化模型事件。"""
        del kwargs
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                async with client.stream("POST", self.host + "/api/generate", json=payload, headers={"Content-Type": "application/json"}) as response:
                    if response.status_code >= 400:
                        yield ModelStreamEvent("error", metadata={"message": f"Ollama request failed with HTTP {response.status_code}: {await response.aread()}"})
                        return
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        text = data.get("response")
                        if isinstance(text, str) and text:
                            yield ModelStreamEvent("text_delta", text=text)
                        if data.get("done"):
                            metadata = {
                                "input_tokens": data.get("prompt_eval_count"),
                                "output_tokens": data.get("eval_count"),
                                "finish_reason": data.get("done_reason"),
                            }
                            self.last_completion_metadata = metadata
                            yield ModelStreamEvent("completed", metadata=metadata)
                            return
        except (httpx.RequestError, json.JSONDecodeError) as exc:
            yield ModelStreamEvent("error", metadata={"message": f"Ollama stream failed: {exc}"})


def _normalize_versioned_base_url(base_url: str) -> str:
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _extract_openai_text(data: JsonObject) -> str:
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


def _extract_openai_text_from_sse(body_text: str) -> str:
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
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
            continue
        if event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return text
        part = event.get("part")
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text:
                return text
        item = event.get("item")
        if isinstance(item, dict):
            text = _extract_openai_text({"output": [item]})
            if text:
                return text
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            text = _extract_openai_text(response)
            if text:
                return text
        text = _extract_openai_text(event)
        if text:
            return text
    if deltas:
        return "".join(deltas)
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response)
    return ""


def _extract_openai_response_from_sse(body_text: str) -> tuple[str, JsonObject]:
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


def _extract_usage_cache_details(data: JsonObject) -> JsonObject:
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
        "cache_hit": cached_tokens > 0,
    }


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
        self.last_completion_metadata = {}

    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        tools: list[JsonObject] | None = None,
        input_items: list[JsonObject] | None = None,
    ) -> str:
        """向 OpenAI-compatible `/responses` 接口发起一次模型调用。

        为什么存在：
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，
        更不应该自己去判断 prompt cache 参数要不要带。这个函数把这些后端
        细节都包起来，对上层暴露统一的 `complete()` 行为。

        输入 / 输出：
        - 输入：完整 prompt、最大输出 token，以及可选的 prompt cache 参数
        - 输出：模型最终文本；同时把 usage / cached_tokens 等元数据写进
          `self.last_completion_metadata`

        在 agent 链路里的位置：
        它位于 `Nano.ask()` 的模型调用阶段，是稳定前缀缓存复用链路真正
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
            payload["parallel_tool_calls"] = False
        # runtime 传入的是“稳定前缀”的签名，而不是整段 prompt 的签名。
        # 这样缓存复用针对的是稳定段，不会因为动态 history 每轮变化而失效。
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
                attempts=3,
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
        response_headers = getattr(response, "headers", {}) or {}
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
            **_extract_usage_cache_details(data),
        }
        return _extract_openai_text(data)

    async def stream(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        tools: list[JsonObject] | None = None,
        input_items: list[JsonObject] | None = None,
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
            payload["parallel_tool_calls"] = False
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": OPENAI_COMPATIBLE_USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        saw_delta = False
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                async with client.stream("POST", self.base_url + "/responses", json=payload, headers=headers) as response:
                    if response.status_code >= 400:
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
                                yield ModelStreamEvent("text_delta", text=text)
                        elif event_type == "response.completed":
                            response_data = event.get("response") or {}
                            if not saw_delta:
                                text = _extract_openai_text(response_data)
                                if text:
                                    yield ModelStreamEvent("text_delta", text=text)
                            metadata = {
                                "prompt_cache_supported": self.supports_prompt_cache,
                                "prompt_cache_key": prompt_cache_key,
                                "prompt_cache_retention": prompt_cache_retention,
                                "response_output": response_data.get("output", []),
                                **_extract_usage_cache_details(response_data),
                            }
                            self.last_completion_metadata = metadata
                            yield ModelStreamEvent("completed", metadata=metadata)
                            return
                        elif event_type == "response.output_item.done":
                            item = event.get("item") or {}
                            if item.get("type") == "function_call":
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
        except (httpx.RequestError, json.JSONDecodeError) as exc:
            yield ModelStreamEvent("error", metadata={"message": f"OpenAI-compatible stream failed: {exc}"})


def _extract_anthropic_text(data: JsonObject) -> str:
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
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> str:
        """请求一条非流式 Anthropic 兼容完成响应。"""
        # 为了保持统一接口，runtime 仍然会传缓存参数进来；
        # 这里只是显式丢弃，因为当前 Anthropic-compatible 路径没有接缓存复用。
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

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
                attempts=3,
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
    ) -> AsyncIterator[ModelStreamEvent]:
        """将 Anthropic Messages SSE 事件转换为标准化文本增量。"""
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "max_tokens": max_new_tokens,
            "stream": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                async with client.stream("POST", self.base_url + "/messages", json=payload, headers=headers) as response:
                    if response.status_code >= 400:
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
                        if event_type == "content_block_delta":
                            text = event.get("delta", {}).get("text")
                            if isinstance(text, str) and text:
                                yield ModelStreamEvent("text_delta", text=text)
                        elif event_type == "message_delta":
                            usage = event.get("usage") or {}
                            self.last_completion_metadata.update(
                                {
                                    "output_tokens": usage.get("output_tokens"),
                                    "finish_reason": event.get("delta", {}).get("stop_reason"),
                                }
                            )
                        elif event_type == "message_stop":
                            yield ModelStreamEvent("completed", metadata=dict(self.last_completion_metadata))
                            return
                        elif event_type == "error":
                            yield ModelStreamEvent("error", metadata={"message": str(event.get("error") or event)})
                            return
        except (httpx.RequestError, json.JSONDecodeError) as exc:
            yield ModelStreamEvent("error", metadata={"message": f"Anthropic-compatible stream failed: {exc}"})
