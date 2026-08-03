"""运行时跨模块共享的类型定义。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from nano.runtime.query_events import ModelStreamEvent

class ModelClient(Protocol):
    """定义运行时可调用的统一模型客户端接口。"""

    model: str
    supports_prompt_cache: bool
    supports_native_tool_calls: bool
    native_tool_call_protocol: str
    last_completion_metadata: dict[str, Any]

    def complete(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> str | list[str]:
        """返回一次非流式模型完成结果。"""
        ...

    def stream(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """返回标准化的模型流事件。"""
        ...


class ToolRunner(Protocol):
    """定义工具注册表中可执行函数的统一接口。"""

    def __call__(self, args: Mapping[str, Any]) -> str:
        """执行已校验的工具参数并返回文本结果。"""
        ...
