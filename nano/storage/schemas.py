"""Pydantic schemas for persisted Nano state and provider payloads."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CheckpointModel(BaseModel):
    """描述一个可恢复的任务 checkpoint。"""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str = ""
    parent_checkpoint_id: str = ""
    schema_version: str = "phase1-v1"
    created_at: str = ""
    current_goal: str = ""
    completed: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    current_blocker: str = ""
    next_step: str = ""
    key_files: list[dict[str, Any]] = Field(default_factory=list)
    freshness: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    runtime_identity: dict[str, Any] = Field(default_factory=dict)


class CheckpointStateModel(BaseModel):
    """描述会话中当前 checkpoint 及其历史记录。"""

    model_config = ConfigDict(extra="forbid")

    current_id: str = ""
    items: dict[str, CheckpointModel] = Field(default_factory=dict)


class ResumeStateModel(BaseModel):
    """描述恢复检查结果。"""

    model_config = ConfigDict(extra="forbid")

    status: str = "no-checkpoint"
    stale_paths: list[str] = Field(default_factory=list)
    runtime_identity_mismatch_fields: list[str] = Field(default_factory=list)
    stale_summary_invalidations: int = Field(default=0, ge=0)


class SessionModel(BaseModel):
    """描述写入 `.nano/sessions` 的会话文档。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: str = ""
    workspace_root: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    memory: dict[str, Any] = Field(default_factory=dict)
    last_input_token_count: int = 0
    checkpoints: CheckpointStateModel = Field(default_factory=CheckpointStateModel)
    runtime_identity: dict[str, Any] = Field(default_factory=dict)
    resume_state: ResumeStateModel = Field(default_factory=ResumeStateModel)


class ProviderErrorModel(BaseModel):
    """统一 provider 返回中的错误字段。"""

    message: str = ""
    type: str = ""
    code: str | int | None = None


class ProviderEnvelopeModel(BaseModel):
    """校验 provider JSON 响应的通用外层结构。"""

    model_config = ConfigDict(extra="allow")

    error: ProviderErrorModel | str | None = None


class OllamaResponseModel(ProviderEnvelopeModel):
    """校验 Ollama 生成响应。"""

    response: str = ""


class OpenAIResponseModel(ProviderEnvelopeModel):
    """校验 OpenAI-compatible JSON 响应。"""

    output_text: str | None = None
    output: list[dict[str, Any]] = Field(default_factory=list)
    choices: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class AnthropicResponseModel(ProviderEnvelopeModel):
    """校验 Anthropic-compatible 消息响应。"""

    content: list[dict[str, Any]] = Field(default_factory=list)
