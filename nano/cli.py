"""命令行入口。

这个模块负责把“用户怎么启动 nano”翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。
"""

import argparse
import asyncio
import os
import signal
import shutil
import sys
import textwrap
import time
from typing import Callable, Sequence

from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import has_completions
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.shortcuts import CompleteStyle, PromptSession
from prompt_toolkit.widgets import Dialog, Label, RadioList

from nano.config import load_project_env, provider_env
from nano.providers.clients import AnthropicCompatibleModelClient, OpenAICompatibleModelClient
from nano.runtime.agent_loop import QueryEngine
from nano.runtime.query_events import QueryEvent
from nano.runtime.runtime import AgentRuntime
from nano.skills import get_skill_by_name, resolve_skill_prompt, discover_skills
from nano.storage.session_store import SessionStore
from nano.utils.text import middle
from nano.workspace.context import WorkspaceContext

DEFAULT_SECRET_ENV_NAMES = (
    "NANO_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "NANO_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "NANO_DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY",
    "NANO_RIGHT_CODES_API_KEY",
    "RIGHT_CODES_API_KEY",
    "GITHUB_PAT",
    "GH_PAT",
)

WELCOME_ART = ("")
WELCOME_NAME = "nano"
WELCOME_SUBTITLE = "local coding agent"
WELCOME_STATUS = "calm shell, ready for work"
SLASH_COMMANDS = ("/help", "/memory", "/session", "/resume", "/reset", "/exit", "/quit")
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /memory  List persistent project memories.
    /session Show the path to the saved session file.
    /resume  Select and resume a saved session.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.

    While a request is running, press Ctrl-C twice within 2 seconds to interrupt it.
    """
).strip()

INTERRUPT_CONFIRMATION_SECONDS = 2.0


DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_BASE_URL = "https://www.right.codes/claude/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
SECRET_ENV_NAMES_VAR = "NANO_SECRET_ENV_NAMES"


class _DoubleCtrlCInterruptHandler:
    """要求连续两次 Ctrl-C 才取消当前请求，避免误触中断。"""

    def __init__(self, runtime: AgentRuntime, clock: Callable[[], float] = time.monotonic) -> None:
        """绑定当前 agent 和用于判定连续按键间隔的时钟。"""
        self.runtime = runtime
        self.clock = clock
        self.first_interrupt_at = None

    def __call__(self, signum, frame) -> None:
        """处理 SIGINT：首次确认意图，第二次取消正在运行的请求。"""
        del signum, frame
        interrupt_at = self.clock()
        if self.first_interrupt_at is not None and interrupt_at - self.first_interrupt_at <= INTERRUPT_CONFIRMATION_SECONDS:
            if self.runtime.interrupt_current_request():
                print("\nInterrupting current request...", flush=True)
            self.first_interrupt_at = None
            return
        self.first_interrupt_at = interrupt_at
        print("\nPress Ctrl-C again within 2 seconds to interrupt the current request.", flush=True)


class _LiveResponsePrinter:
    """将模型文本增量实时写入终端，并过滤协议标签和工具调用。"""

    def __init__(self) -> None:
        """初始化当前模型回合的增量缓冲状态。"""
        self._mode = "undecided"
        self._pending = ""
        self.has_output = False

    def __call__(self, event: QueryEvent) -> None:
        """消费查询事件并在文本增量到达时输出用户可见内容。"""
        if event.type == "model_requested":
            self._mode = "undecided"
            self._pending = ""
            return
        if event.type == "text_delta":
            self._consume(str(event.payload.get("text", "")))
            return
        if event.type == "final" and self._mode == "final":
            self._write(self._pending)
            self._pending = ""

    def _consume(self, text: str) -> None:
        """识别当前增量所属的协议类型，并输出可安全展示的文本。"""
        self._pending += text
        if self._mode == "undecided":
            if self._pending.startswith("<final>"):
                self._mode = "final"
                self._pending = self._pending[len("<final>"):]
            elif self._pending.startswith("<tool"):
                self._mode = "tool"
                self._pending = ""
            elif "<final>".startswith(self._pending) or "<tool".startswith(self._pending):
                return
            else:
                self._mode = "text"
        if self._mode == "final":
            self._write_final_text()
        elif self._mode == "text":
            self._write(self._pending)
            self._pending = ""

    def _write_final_text(self) -> None:
        """输出最终回答正文，同时保留可能被拆分的结束标签。"""
        closing_tag = "</final>"
        closing_index = self._pending.find(closing_tag)
        if closing_index >= 0:
            self._write(self._pending[:closing_index])
            self._pending = ""
            return
        keep_length = 0
        for length in range(1, min(len(self._pending), len(closing_tag) - 1) + 1):
            if self._pending.endswith(closing_tag[:length]):
                keep_length = length
        if keep_length:
            self._write(self._pending[:-keep_length])
            self._pending = self._pending[-keep_length:]
            return
        self._write(self._pending)
        self._pending = ""

    def _write(self, text: str) -> None:
        """立即写入非空文本，确保终端在流尚未结束时刷新。"""
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()
        self.has_output = True


async def _consume_streamed_response(runtime: AgentRuntime, user_message: str) -> tuple[str, bool]:
    """消费运行事件流，并返回最终答案及是否已实时输出正文。"""
    printer = _LiveResponsePrinter()
    answer = ""
    async for event in QueryEngine(runtime).stream_async(user_message):
        printer(event)
        if event.type in {"final", "error", "stopped"}:
            answer = str(event.payload["answer"])
    return answer, printer.has_output


def _print_streamed_response(runtime: AgentRuntime, user_message: str) -> None:
    """执行请求并实时打印模型回答，流结束后补齐终端换行。"""
    answer, has_output = asyncio.run(_consume_streamed_response(runtime, user_message))
    if has_output:
        print()
        return
    print(answer)


def _resolve_user_skill_command(runtime: AgentRuntime, user_input: str) -> tuple[str, str] | None:
    """识别用户 `/skill 参数` 输入，并返回已展开的 Skill 提示词。"""
    if not user_input.startswith("/"):
        return None
    space_index = user_input.find(" ")
    command_name = user_input[1:space_index] if space_index > 0 else user_input[1:]
    command_args = user_input[space_index + 1 :] if space_index > 0 else ""
    skill = get_skill_by_name(command_name, runtime.root)
    if skill is None or not skill.user_invocable:
        return None
    return skill.name, resolve_skill_prompt(skill, command_args)


SLASH_COMMAND_KEY_BINDINGS = KeyBindings()
SESSION_SELECTOR_KEY_BINDINGS = KeyBindings()


@SLASH_COMMAND_KEY_BINDINGS.add("enter", filter=has_completions)
def _accept_slash_command_completion(event) -> None:
    """将当前候选命令写入输入缓冲区，并立即提交该命令。"""
    completion_state = event.current_buffer.complete_state
    if completion_state is not None:
        completion = completion_state.current_completion or completion_state.completions[0]
        event.current_buffer.apply_completion(completion)
    event.current_buffer.validate_and_handle()


@SESSION_SELECTOR_KEY_BINDINGS.add("escape", eager=True)
def _cancel_session_selector(event) -> None:
    """取消会话选择弹窗并回到主 REPL 输入框。"""
    event.app.exit(result=None)


def _effective_model(args, provider):
    # 模型选择优先级：
    # 1. 用户显式传入 --model
    # 2. provider 对应的环境变量
    # 3. 代码里的默认值
    explicit_model = args.model
    if explicit_model:
        return explicit_model
    if provider == "openai":
        model = provider_env("NANO_OPENAI_MODEL", ("OPENAI_MODEL",))
        if model:
            return model
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        model = provider_env("NANO_ANTHROPIC_MODEL", ("ANTHROPIC_MODEL",))
        if model:
            return model
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "deepseek":
        model = provider_env("NANO_DEEPSEEK_MODEL", ("DEEPSEEK_MODEL",))
        if model:
            return model
        return DEFAULT_DEEPSEEK_MODEL
    raise ValueError(f"Unsupported provider: {provider}")


def _configured_secret_names(args):
    configured_secret_names: set[str] = set(DEFAULT_SECRET_ENV_NAMES)
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    return sorted(configured_secret_names)


def _build_model_client(args):
    provider = args.provider
    # CLI 只负责把 provider 选择翻译成具体 client。
    # 真正的提示词格式、缓存支持、HTTP 协议差异，都封装在 models.py 里。
    if provider == "openai":
        model = _effective_model(args, provider)
        base_url = args.base_url or provider_env("NANO_OPENAI_API_BASE", ("OPENAI_API_BASE",), DEFAULT_OPENAI_BASE_URL)
        api_key = provider_env(
            "NANO_OPENAI_API_KEY",
            ("OPENAI_API_KEY", "NANO_RIGHT_CODES_API_KEY", "RIGHT_CODES_API_KEY", "NANO_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        )
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=args.openai_timeout,
        )
    if provider == "anthropic":
        model = _effective_model(args, provider)
        base_url = args.base_url or provider_env("NANO_ANTHROPIC_API_BASE", ("ANTHROPIC_API_BASE",), DEFAULT_ANTHROPIC_BASE_URL)
        api_key = provider_env(
            "NANO_ANTHROPIC_API_KEY",
            ("ANTHROPIC_API_KEY", "NANO_RIGHT_CODES_API_KEY", "RIGHT_CODES_API_KEY", "NANO_OPENAI_API_KEY", "OPENAI_API_KEY"),
        )
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=args.openai_timeout,
        )
    if provider == "deepseek":
        model = _effective_model(args, provider)
        base_url = args.base_url or provider_env("NANO_DEEPSEEK_API_BASE", ("DEEPSEEK_API_BASE",), DEFAULT_DEEPSEEK_BASE_URL)
        api_key = provider_env("NANO_DEEPSEEK_API_KEY", ("DEEPSEEK_API_KEY",))
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=args.openai_timeout,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def build_welcome(runtime: AgentRuntime, model: str, host: str) -> str:
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = middle(text, width - 4)
        return f"| {body.ljust(width - 4)} |"

    def divider(char="-"):
        return "+" + char * (width - 2) + "+"

    def center(text):
        body = middle(text, inner)
        return f"| {body.center(inner)} |"

    def cell(label, value, size):
        body = middle(f"{label:<9} {value}", size)
        return body.ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return f"| {left}{' ' * gap}{right} |"

    line = divider("=")
    rows = [center(text) for text in WELCOME_ART]
    rows.extend(
        [
            center(WELCOME_NAME),
            center(WELCOME_SUBTITLE),
            center(WELCOME_STATUS),
            divider("-"),
            row(""),
            row("WORKSPACE  " + middle(runtime.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", runtime.workspace.branch),
            pair("APPROVAL", runtime.approval_policy, "SESSION", runtime.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, line])


def build_agent(args: argparse.Namespace) -> AgentRuntime:
    """根据 CLI 参数装配出一个可运行的 AgentRuntime 实例。

    为什么存在：
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：
    model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把“启动参数”翻译成“agent 运行现场”。

    输入 / 输出：
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `AgentRuntime`，或一个从旧 session 恢复出来的 `AgentRuntime`

    在 agent 链路里的位置：
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，
    得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会消费异步事件流。
    """
    # 这里是 CLI 到 runtime 的装配点：
    # 先采集工作区快照和加载项目级环境，再整理 secret 名单、模型后端和 session。
    workspace = WorkspaceContext.build(args.cwd)
    load_project_env(workspace.repo_root)
    configured_secret_names = _configured_secret_names(args)
    store = SessionStore(workspace.repo_root + "/.nano/sessions", secret_env_names=configured_secret_names)
    model = _build_model_client(args)
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return AgentRuntime.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=configured_secret_names,
        )
    return AgentRuntime(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for DeepSeek, OpenAI-compatible, or Anthropic-compatible models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--provider", choices=("openai", "anthropic", "deepseek"), default="deepseek", help="Model backend to use.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to NANO_OPENAI_MODEL for openai, NANO_ANTHROPIC_MODEL for anthropic, and NANO_DEEPSEEK_MODEL for deepseek when set.",
    )
    parser.add_argument("--base-url", default=None, help="Provider API base URL for deepseek, openai, or anthropic.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum successful tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to the provider.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runtime = build_agent(args)

    model = runtime.model_client.model
    host = runtime.model_client.base_url
    print(build_welcome(runtime, model=model, host=host))

    if args.prompt:
        # one-shot 模式：只跑一次 ask，不进入 REPL 循环。
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                previous_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, _DoubleCtrlCInterruptHandler(runtime))
                try:
                    _print_streamed_response(runtime, prompt)
                finally:
                    signal.signal(signal.SIGINT, previous_handler)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        return 0

    prompt_session = PromptSession(
        completer=WordCompleter(
            [*SLASH_COMMANDS, *(f"/{skill.name}" for skill in discover_skills(runtime.root) if skill.user_invocable)],
            sentence=True,
        ),
        complete_while_typing=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        key_bindings=SLASH_COMMAND_KEY_BINDINGS,
    )
    while True:
        # 交互模式：每次读取一条用户输入，交给同一个 agent，
        # 因此 session history 和 working memory 会跨轮延续。
        try:
            user_input = prompt_session.prompt("\nnano> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            memories = runtime.memory.list_file_memories()
            if not memories:
                print("No memories saved yet.")
            else:
                print(f"{len(memories)} memories:")
                for memory in memories:
                    print(f"    [{memory.type}] {memory.name} - {memory.description}")
            continue
        if user_input == "/session":
            print(runtime.session_path)
            continue
        if user_input == "/resume":
            sessions = runtime.session_store.list_summaries()
            if not sessions:
                print("No saved sessions.")
                continue
            session_selector = RadioList(
                values=[(item["id"], f"{item['title']}\n  Updated: {item['updated_at']}") for item in sessions],
                select_on_focus=True,
            )
            session_confirm_key_bindings = KeyBindings()

            @session_confirm_key_bindings.add("enter", eager=True)
            def _confirm_session_selector(event) -> None:
                """确认当前带有星号标记的会话并退出选择界面。"""
                event.app.exit(result=session_selector.current_value)

            session_dialog = Dialog(
                title="Resume session",
                body=HSplit([Label("Use ↑/↓ to select, Enter to confirm, or Esc to cancel."), session_selector], padding=1),
                with_background=True,
            )
            session_app = Application(
                layout=Layout(session_dialog, focused_element=session_selector.control),
                key_bindings=merge_key_bindings([SESSION_SELECTOR_KEY_BINDINGS, session_confirm_key_bindings]),
                full_screen=True,
                mouse_support=True,
            )
            session_id = session_app.run()
            if not session_id:
                continue
            runtime = AgentRuntime.from_session(
                model_client=runtime.model_client,
                workspace=runtime.workspace,
                session_store=runtime.session_store,
                session_id=session_id,
                run_store=runtime.run_store,
                approval_policy=runtime.approval_policy,
                max_steps=runtime.max_steps,
                max_new_tokens=runtime.max_new_tokens,
                depth=runtime.depth,
                max_depth=runtime.max_depth,
                read_only=runtime.read_only,
                shell_env_allowlist=runtime.shell_env_allowlist,
                secret_env_names=runtime.secret_env_names,
                feature_flags=runtime.feature_flags,
                allowed_tools=runtime.allowed_tools,
            )
            print(f"Resumed session {session_id}.")
            continue
        if user_input == "/reset":
            runtime.reset()
            print("session reset")
            continue

        skill_command = _resolve_user_skill_command(runtime, user_input)
        if skill_command is not None:
            skill_name, resolved_prompt = skill_command
            print(f"Invoking skill: {skill_name}")
            user_input = resolved_prompt

        print()
        try:
            previous_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _DoubleCtrlCInterruptHandler(runtime))
            try:
                _print_streamed_response(runtime, user_input)
            finally:
                signal.signal(signal.SIGINT, previous_handler)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
