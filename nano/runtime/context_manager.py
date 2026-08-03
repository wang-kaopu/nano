"""Prompt 组装。

这个模块负责将稳定前缀、工作记忆、相关笔记、短对话和当前请求原样送入模型。
"""

from __future__ import annotations

import json
from typing import Any


SECTION_ORDER = ("prefix", "memory", "relevant_memory", "history", "current_request")
CURRENT_REQUEST_SECTION = "current_request"
RELEVANT_MEMORY_LIMIT = 5


class ContextManager:
    """组装不按 section 裁剪的模型输入。"""

    def __init__(self, agent: Any) -> None:
        """绑定提供 prefix、memory 和会话状态的运行时。"""
        self.agent = agent

    def build(self, user_message: str, include_prefix: bool = True, relevant_memories: list[Any] | None = None) -> tuple[str, dict[str, Any]]:
        """组装一轮完整 prompt，并记录各 section 的原始长度。"""
        user_message = str(user_message)
        memory_enabled = self.agent.feature_enabled("memory")
        relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
        prefix = self.agent.prefix if include_prefix else ""
        checkpoint_text = self.agent.render_checkpoint_text().strip()
        if checkpoint_text and include_prefix:
            prefix = prefix + "\n\n" + checkpoint_text
        memory = "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text())
        selected_memories = relevant_memories or []
        relevant_memory = (
            "\n\n".join(["Relevant memories:", *[f"<system-reminder>\n{memory.header}\n\n{memory.content}\n</system-reminder>" for memory in selected_memories]])
            if selected_memories
            else "Relevant memories:\n- none"
        )
        history = self._raw_history_text(self._model_history())
        current_request = f"Current user request:\n{user_message}"
        if not include_prefix:
            reminder = self.agent.anthropic_user_system_reminder()
            if reminder:
                current_request = f"<system-reminder>\n{reminder}\n</system-reminder>\n\n{current_request}"
        sections = {
            "prefix": prefix,
            "memory": memory,
            "relevant_memory": relevant_memory,
            "history": history,
            CURRENT_REQUEST_SECTION: current_request,
        }
        prompt = "\n\n".join(sections[section] for section in SECTION_ORDER if sections[section]).strip()
        return prompt, self._metadata(prompt, sections, selected_memories, user_message)

    def _model_history(self) -> list[dict[str, Any]]:
        """优先返回流式循环维护的短对话窗口。"""
        session = self.agent.session
        conversation = session.get("conversation")
        if isinstance(conversation, list) and conversation:
            return list(conversation)
        return list(session.get("history", []))

    def _raw_history_text(self, history: list[dict[str, Any]]) -> str:
        """将模型侧历史完整渲染为 Transcript section。"""
        if not history:
            return "Transcript:\n- empty"
        lines = []
        for item in history:
            if item["role"] == "tool":
                lines.append(f"[tool:{item['name']}] {json.dumps(item.get('args', {}), sort_keys=True)}")
                lines.append(str(item["content"]))
            else:
                lines.append(f"[{item['role']}] {item['content']}")
        return "\n".join(["Transcript:", *lines])

    def _metadata(
        self,
        prompt: str,
        sections: dict[str, str],
        selected_memories: list[Any],
        user_message: str,
    ) -> dict[str, Any]:
        """生成不含预算决策的 prompt trace 元数据。"""
        section_metadata = {
            section: {
                "raw_chars": len(sections[section]),
                "rendered_chars": len(sections[section]),
            }
            for section in SECTION_ORDER
        }
        return {
            "prompt_chars": len(prompt),
            "section_order": list(SECTION_ORDER),
            "sections": section_metadata,
            "relevant_memory": {
                "limit": RELEVANT_MEMORY_LIMIT,
                "selected_count": len(selected_memories),
                "selected_filenames": [memory.filename for memory in selected_memories],
                "selected_paths": [str(memory.path) for memory in selected_memories],
                "raw_chars": len(sections["relevant_memory"]),
                "rendered_chars": len(sections["relevant_memory"]),
                "rendered_count": len(selected_memories),
            },
            "history": {
                "raw_chars": len(sections["history"]),
                "rendered_chars": len(sections["history"]),
                "older_entries_count": 0,
                "collapsed_duplicate_reads": 0,
                "reused_file_summary_count": 0,
                "summarized_tool_count": 0,
            },
            "current_request": {
                "text": user_message,
                "raw_chars": len(user_message),
                "rendered_chars": len(user_message),
                "section_chars": len(sections[CURRENT_REQUEST_SECTION]),
            },
        }
