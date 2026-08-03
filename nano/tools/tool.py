"""工具抽象及其运行时契约。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel

from nano.workspace.context import MAX_TOOL_OUTPUT


class ToolProgressData(BaseModel):
    """工具执行过程可向调用方报告的进度信息。"""

    stage: str


InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT")
ProgressT = TypeVar("ProgressT", bound=ToolProgressData)


@dataclass(frozen=True)
class PermissionResult:
    """描述工具输入是否通过执行前权限与状态检查。"""

    behavior: Literal["allow", "deny"]
    updated_input: BaseModel | None = None
    message: str = ""

    @classmethod
    def allow(cls, updated_input: BaseModel) -> "PermissionResult":
        """创建允许执行的结果。"""
        return cls(behavior="allow", updated_input=updated_input)

    @classmethod
    def deny(cls, message: str, updated_input: BaseModel | None = None) -> "PermissionResult":
        """创建拒绝执行的结果。"""
        return cls(behavior="deny", updated_input=updated_input, message=message)


@dataclass(frozen=True)
class ToolResult(Generic[OutputT]):
    """承载工具原始输出、模型可见文本和最后进度状态。"""

    output: OutputT
    content: str
    progress: ToolProgressData | None = None


CanUseTool = Callable[["Tool[Any, Any, Any]", BaseModel], bool]
ProgressCallback = Callable[[ToolProgressData], None]


class Tool(Generic[InputT, OutputT, ProgressT]):
    """所有可调用工具的 Python 运行时契约。"""

    name: str
    aliases: tuple[str, ...] = ()
    max_result_size_chars: int = MAX_TOOL_OUTPUT
    input_schema: type[InputT]

    def __init__(
        self,
        *,
        name: str,
        input_schema: type[InputT],
        aliases: tuple[str, ...] = (),
        max_result_size_chars: int = MAX_TOOL_OUTPUT,
    ) -> None:
        """初始化工具的稳定标识、别名和输入 schema。"""
        self.name = name
        self.input_schema = input_schema
        self.aliases = aliases
        self.max_result_size_chars = max_result_size_chars

    @property
    def input_json_schema(self) -> dict[str, Any]:
        """返回可直接发送给模型提供方的 JSON Schema。"""
        return self.input_schema.model_json_schema()

    def parse_input(self, args: Mapping[str, Any] | None) -> InputT:
        """使用 Pydantic 校验并标准化模型提供的输入。"""
        return self.input_schema.model_validate(args or {})

    def description(self, input_value: InputT | None, options: Mapping[str, Any] | None = None) -> str:
        """返回发送给模型 API 的工具能力描述。"""
        raise NotImplementedError("Tool subclasses must implement description()")

    def prompt(self, options: Mapping[str, Any] | None = None) -> str:
        """返回注入 system prompt 的工具使用指南。"""
        raise NotImplementedError("Tool subclasses must implement prompt()")

    def is_concurrency_safe(self, input_value: InputT) -> bool:
        """判断给定输入能否与同类工具调用并发执行。"""
        return False

    def is_read_only(self, input_value: InputT) -> bool:
        """判断给定输入是否不会修改工作区。"""
        return False

    def is_destructive(self, input_value: InputT) -> bool:
        """判断给定输入是否可能产生不可逆副作用。"""
        return False

    def requires_approval(self, input_value: InputT) -> bool:
        """判断本次工具调用是否需要遵循运行时审批策略。"""
        return not self.is_read_only(input_value)

    def check_permissions(self, input_value: InputT, context: Any) -> PermissionResult:
        """执行工具专属的路径、状态和权限检查。"""
        return PermissionResult.allow(input_value)

    def call(
        self,
        args: InputT,
        context: Any,
        can_use_tool: CanUseTool,
        parent_message: str | None,
        on_progress: ProgressCallback | None = None,
    ) -> ToolResult[OutputT]:
        """执行已校验工具调用并返回统一结果。"""
        raise NotImplementedError("Tool subclasses must implement call()")

    def render_tool_use_message(self, input_value: InputT, options: Mapping[str, Any] | None = None) -> str:
        """返回 CLI 中展示工具调用的文本。"""
        return f"{self.name}({input_value.model_dump(mode='python')})"

    def render_tool_result_message(
        self,
        content: str,
        progress: ToolProgressData | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> str:
        """返回 CLI 中展示工具结果的文本。"""
        return content
