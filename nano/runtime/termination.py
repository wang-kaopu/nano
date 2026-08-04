"""统一 Provider 完成事件的终止原因。"""

from __future__ import annotations

from typing import Any, Mapping


def normalize_termination_reason(metadata: Mapping[str, Any]) -> str:
    """将不同 Provider 的结束字段映射为运行时统一语义。"""
    raw_reason = str(metadata.get("termination_reason") or metadata.get("provider_finish_reason") or metadata.get("finish_reason") or "").lower()
    if raw_reason in {"max_tokens", "length", "max_output_tokens", "output_limit"}:
        return "output_limit"
    if raw_reason in {"tool_use", "tool_calls", "function_call"}:
        return "tool_call"
    if raw_reason in {"end_turn", "stop", "stop_sequence", "completed", "complete"}:
        return "complete"
    if raw_reason in {"content_filter", "safety"}:
        return "content_filter"
    return "unknown"
