"""Stable prompt prefix construction."""

import hashlib
import json
import textwrap
from dataclasses import dataclass

from nano.utils import now


@dataclass
class PromptPrefix:
    # prefix 除了文本本身，还带一小份元数据，
    # 这样 runtime 才能明确判断 prefix 是否可以复用。
    text: str
    static_system: str
    dynamic_system: str
    user_system_reminder: str
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


def build_dynamic_system_context(workspace) -> str:
    """构建随工作区变化而刷新的 system prompt 上下文。"""
    return "\n\n".join(
        (
            "You are operating in a local repository. Treat the following Git state and project instructions as authoritative context.",
            "Paths supplied to workspace tools are relative to the repository root unless the tool says otherwise.",
            workspace.system_text(),
        )
    )


def build_user_system_reminder(workspace) -> str:
    """构建放入首条 user 消息的项目级系统提醒。"""
    return workspace.user_system_reminder_text()


def build_prompt_prefix(workspace, tools, memory_prompt_section="", built_at=None, native_tool_calls=False):
    """构建包含工作规范、工具协议和工作区元数据的稳定提示词前缀。"""
    dynamic_system_context = build_dynamic_system_context(workspace)
    # prefix 可以理解成 agent 的“工作手册”：
    # 它是谁、工具怎么调用、当前仓库是什么状态，都写在这里。
    identity = "# 1. Identity\nYou are nano, an interactive agent that helps with software engineering tasks using the tools available to you."
    doing_tasks = textwrap.dedent(
        """\
        # 3. Doing Tasks
         - Do not propose changes to code you haven't read. Read files first.
         - Do not create files unless necessary. Prefer editing existing files.
         - Avoid over-engineering. Only make changes that were requested.
        - Do not expand scope: fixing a bug does not justify refactoring surrounding code.
        - Do not add defensive programming for impossible scenarios: avoid speculative try-catch blocks and validation.
        - Do not abstract prematurely: "Three similar lines of code is better than a premature abstraction."
        """
    ).strip()
    actions = textwrap.dedent(
        """\
        # 4. Actions
        Prefer reversible actions. For risky or destructive ones (rm -rf, git push, dropping tables), confirm with the user before proceeding.
        - Reversibility: determine whether the action can be safely undone.
        - Blast radius: determine whether the action affects only this local workspace or shared people, data, infrastructure, or history.
        - High risk combines irreversible changes with a shared blast radius, such as force-pushing, deleting cloud resources, or dropping shared tables. Confirm with the user before proceeding.
        - Low risk is reversible and local, such as editing a local file. Proceed when it is within the requested task.
        """
    ).strip()
    using_tools = textwrap.dedent(
        """\
        # 5. Using Tools
        - Use read_file / edit_file / list_files / grep_search instead of shell cat,
          sed, ls, grep. Reserve run_shell for actual shell operations.
        - If several tool calls are independent, make them in parallel.
        """
    ).strip()
    tone_and_style = textwrap.dedent(
        """\
        # 6. Tone & Style
         - Keep responses short and concise. Lead with the answer.
         - Reference code as file_path:line_number.`;
        """
    ).strip()
    output_efficiency = textwrap.dedent(
        """\
        # 7. Output Efficiency
        - Do not restate the request, narrate routine tool use, or add unrelated background.
        - For completed work, report the changed files and verification in the fewest useful lines.
        """
    ).strip()
    static_system = "\n\n".join((identity, doing_tasks, actions, using_tools, tone_and_style, output_efficiency))
    dynamic_system = "# 2. System\n" + dynamic_system_context
    if memory_prompt_section:
        dynamic_system += "\n\n" + memory_prompt_section
    text = "\n\n".join((static_system, dynamic_system))
    signature = tool_signature(tools)
    return PromptPrefix(
        text=text,
        static_system=static_system,
        dynamic_system=dynamic_system,
        user_system_reminder=build_user_system_reminder(workspace),
        hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        workspace_fingerprint=workspace.fingerprint(),
        tool_signature=signature,
        built_at=built_at or now(),
    )
