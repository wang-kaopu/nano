"""对象化的 agent 启动 API。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypeAlias

from nano.runtime.agent_loop import QueryEngine
from nano.runtime.query_events import QueryEvent
from nano.runtime.runtime import AgentRuntime
from nano.storage.run_store import RunStore
from nano.storage.session_store import SessionStore
from nano.tools.tool_context import ToolContext
from nano.types import ModelClient
from nano.workspace.context import WorkspaceContext


PromptMessage: TypeAlias = str | Mapping[str, Any]


@dataclass(frozen=True)
class AgentDefinition:
    """描述一次 agent 启动所需的模型、提示词和工具能力配置。"""

    model_client: ModelClient
    workspace: WorkspaceContext
    session_store: SessionStore
    tools: Iterable[str] | None = None
    instructions: str = ""
    session: dict[str, Any] | None = None
    run_store: RunStore | None = None
    approval_policy: str = "ask"
    max_steps: int = 12
    max_new_tokens: int = 512
    depth: int = 0
    max_depth: int = 1
    read_only: bool = False
    shell_env_allowlist: Iterable[str] | None = None
    secret_env_names: Iterable[str] | None = None
    feature_flags: Mapping[str, bool] | None = None
    workspace_mutation_lock: asyncio.Lock | None = None


def _normalize_prompt_messages(prompt_messages: PromptMessage | Iterable[PromptMessage]) -> str:
    """校验并合并本次运行的初始用户消息。"""
    messages = (prompt_messages,) if isinstance(prompt_messages, (str, Mapping)) else tuple(prompt_messages)
    if not messages:
        raise ValueError("prompt_messages must contain at least one user message")
    contents: list[str] = []
    for message in messages:
        if isinstance(message, str):
            content = message
        elif isinstance(message, Mapping):
            role = str(message.get("role", "user"))
            if role != "user":
                raise ValueError("prompt_messages only accepts user messages")
            content = str(message.get("content", ""))
        else:
            raise TypeError("prompt_messages entries must be strings or user-message mappings")
        if not content.strip():
            raise ValueError("prompt_messages entries must not be empty")
        contents.append(content)
    return "\n\n".join(contents)


def build_runtime(
    agent_definition: AgentDefinition,
    *,
    tool_use_context: ToolContext | None = None,
    use_exact_tools: bool = False,
    max_turns: int | None = None,
) -> AgentRuntime:
    """根据对象化配置创建本次运行专属的 AgentRuntime。"""
    return AgentRuntime(
        model_client=agent_definition.model_client,
        workspace=agent_definition.workspace,
        session_store=agent_definition.session_store,
        session=agent_definition.session,
        run_store=agent_definition.run_store,
        approval_policy=agent_definition.approval_policy,
        max_steps=agent_definition.max_steps,
        max_new_tokens=agent_definition.max_new_tokens,
        depth=agent_definition.depth,
        max_depth=agent_definition.max_depth,
        read_only=agent_definition.read_only,
        shell_env_allowlist=agent_definition.shell_env_allowlist,
        secret_env_names=agent_definition.secret_env_names,
        feature_flags=agent_definition.feature_flags,
        allowed_tools=agent_definition.tools,
        tool_use_context=tool_use_context,
        use_exact_tools=use_exact_tools,
        max_turns=max_turns,
        agent_instructions=agent_definition.instructions,
        workspace_mutation_lock=agent_definition.workspace_mutation_lock,
    )


async def run_agent(
    *,
    agent_definition: AgentDefinition,
    prompt_messages: PromptMessage | Iterable[PromptMessage],
    tool_use_context: ToolContext | None = None,
    use_exact_tools: bool = False,
    max_turns: int | None = None,
):
    """启动 agent 并持续产出 QueryEvent 事件。

    运行循环始终在独立 asyncio Task 中推进，事件通过队列交付给宿主。
    `use_exact_tools` 会严格应用 AgentDefinition 中声明的工具白名单。
    `max_turns` 限制模型循环次数；未设置时保留现有 runtime 的格式重试容量。
    """
    user_message = _normalize_prompt_messages(prompt_messages)
    runtime = build_runtime(
        agent_definition,
        tool_use_context=tool_use_context,
        use_exact_tools=use_exact_tools,
        max_turns=max_turns,
    )
    events: asyncio.Queue[QueryEvent | Exception | None] = asyncio.Queue()

    async def produce() -> None:
        """在后台推进运行循环，并将结果事件发送给异步消费者。"""
        try:
            async for event in QueryEngine(runtime).stream_async(user_message):
                await events.put(event)
        except Exception as exc:
            await events.put(exc)
        finally:
            await events.put(None)

    producer = asyncio.create_task(produce())
    try:
        while True:
            event = await events.get()
            if event is None:
                break
            if isinstance(event, Exception):
                raise event
            yield event
    finally:
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except asyncio.CancelledError:
            pass
