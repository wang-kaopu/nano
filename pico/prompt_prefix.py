"""Stable prompt prefix construction."""

import hashlib
import json
import textwrap
from dataclasses import dataclass

from .workspace import now


@dataclass
class PromptPrefix:
    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    hash: str
    workspace_fingerprint: str
    tool_signature: str
    built_at: str


def tool_signature(tools):
    payload = []
    for name in sorted(tools):
        tool = tools[name]
        payload.append(
            {
                "name": tool.name,
                "aliases": tool.aliases,
                "max_result_size_chars": tool.max_result_size_chars,
                "input_json_schema": tool.input_json_schema,
                "description": tool.description(None),
                "prompt": tool.prompt(),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_prompt_prefix(workspace, tools, built_at=None, native_tool_calls=False):
    tool_lines = []
    for name, tool in tools.items():
        schema = tool.input_json_schema
        required = set(schema.get("required", []))
        fields = []
        for field_name, field in schema.get("properties", {}).items():
            default = "" if field_name in required else f"={field.get('default')!r}"
            fields.append(f"{field_name}: {field.get('type', 'value')}{default}")
        risk = "approval required" if not tool.is_read_only(tool.input_schema.model_construct()) else "safe"
        tool_lines.append(f"- {name}({', '.join(fields)}) [{risk}] {tool.description(None)}\n  {tool.prompt()}")
    tool_text = "\n".join(tool_lines)
    tool_protocol_rules = (
        "- Use the provided function tools for workspace actions; never serialize a <tool> tag yourself.\n"
        "- Return the final answer as normal text after tool calls complete."
        if native_tool_calls
        else "- Return exactly one <tool>...</tool> or one <final>...</final>.\n"
        "- Tool calls must look like:\n"
        "  <tool>{{\"name\":\"tool_name\",\"args\":{{...}}}}</tool>\n"
        "- For write_file and patch_file with multi-line text, prefer XML style:\n"
        "  <tool name=\"write_file\" path=\"file.py\"><content>...</content></tool>\n"
        "- Final answers must look like:\n"
        "  <final>your answer</final>"
    )
    # prefix 可以理解成 agent 的“工作手册”：
    # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
    text = textwrap.dedent(
        f"""\
        You are pico, a small local coding agent working inside a local repository.

        Rules:
        - Use tools instead of guessing about the workspace.
        {tool_protocol_rules}
        - Never invent tool results.
        - Keep answers concise and concrete.
        - If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.
        - Before writing tests for existing code, read the implementation first.
        - When writing tests, match the current implementation unless the user explicitly asked you to change the code.
        - New files should be complete and runnable, including obvious imports.
        - Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or return a final answer.
        - Required tool arguments must not be empty. Do not call read_file, write_file, patch_file, run_shell, or delegate with args={{}}.

        Tools:
        {tool_text}

        {workspace.text()}
        """
    ).strip()
    signature = tool_signature(tools)
    return PromptPrefix(
        text=text,
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=workspace.fingerprint(),
        tool_signature=signature,
        built_at=built_at or now(),
    )
