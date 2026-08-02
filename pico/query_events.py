"""流式模型客户端与查询循环之间交换的标准化事件。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelStreamEvent:
    """描述一条与 provider 无关的模型响应流事件。"""

    type: str
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryEvent:
    """描述查询处理期间可观察到的进度事件。"""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
