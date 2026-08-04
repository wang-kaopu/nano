"""Agent 运行时核心逻辑。

AgentRuntime 就是包在模型外面的控制循环：负责组 prompt、解析模型输出、
校验并执行工具、写 trace、更新工作记忆，以及在合适的时候停下来。
"""

import asyncio
import json
import hashlib
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import nano.tools.security as securitylib
import nano.tools.tools as toolkit
from nano.permissions import load_project_permissions
from nano.memory import memory as memorylib
from nano.runtime.prompt_prefix import PromptPrefix, build_prompt_prefix, tool_signature
from nano.runtime.task_state import TaskState
from nano.skills import build_skill_descriptions
from nano.storage.run_store import RunStore
from nano.storage.session_store import SessionStore
from nano.tools.tool import Tool, ToolProgressData
from nano.tools.tool_context import ToolContext
from nano.types import ModelClient
from nano.utils.text import clip, now
from nano.workspace.context import IGNORED_PATH_NAMES, MAX_HISTORY, WorkspaceContext

DEFAULT_SHELL_ENV_ALLOWLIST = ("HOME", "LANG", "LC_ALL", "LC_CTYPE", "LOGNAME", "PATH", "PWD", "SHELL", "TERM", "TMPDIR", "TMP", "TEMP", "USER")
DEFAULT_FEATURE_FLAGS = {
    "memory": True,
    "relevant_memory": True,
    "prompt_cache": True,
}
MODEL_CONTEXT_WINDOW = 1_000_000
CONTEXT_WINDOW_RESERVE = 20_000


class SubagentHandle:
    """持有已启动子 agent 的运行时，并提供等待最终结论的入口。"""

    def __init__(self, runtime, task) -> None:
        """绑定子 agent runtime 与其后台运行任务。"""
        self.runtime = runtime
        self.task = task

    async def wait(self) -> str:
        """等待子 agent 完成当前任务并返回最终结论。"""
        return await self.task


class AgentRuntime:
    def __init__(
        self,
        model_client: ModelClient,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session: dict[str, Any] | None = None,
        run_store: RunStore | None = None,
        approval_policy: str = "ask",
        max_steps: int = 6,
        max_new_tokens: int = 512,
        depth: int = 0,
        max_depth: int = 1,
        read_only: bool = False,
        shell_env_allowlist: Iterable[str] | None = None,
        secret_env_names: Iterable[str] | None = None,
        feature_flags: Mapping[str, bool] | None = None,
        allowed_tools: Iterable[str] | None = None,
        tool_use_context: ToolContext | None = None,
        use_exact_tools: bool = False,
        max_turns: int | None = None,
        agent_instructions: str = "",
    ) -> None:
        from nano.runtime.context_manager import ContextManager
        from nano.tools.tool_executor import ToolExecutor

        self.model_client = model_client
        self.workspace = workspace
        self.root = Path(workspace.repo_root)
        self.permissions = load_project_permissions(self.root)
        self.read_file_state: dict[str, int] = {}
        self.session_store = session_store
        self.approval_policy = approval_policy
        self.max_steps = max_steps
        # 未显式配置 turn 上限时，保留既有的“允许若干次格式重试”行为。
        self.max_turns = max(max_steps * 3, max_steps + 4) if max_turns is None else int(max_turns)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.max_new_tokens = max_new_tokens
        self.effective_window = MODEL_CONTEXT_WINDOW - CONTEXT_WINDOW_RESERVE
        self.depth = depth
        self.max_depth = max_depth
        self.read_only = read_only
        self.shell_env_allowlist = tuple(shell_env_allowlist or DEFAULT_SHELL_ENV_ALLOWLIST)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.feature_flags = dict(DEFAULT_FEATURE_FLAGS)
        if feature_flags:
            self.feature_flags.update({str(key): bool(value) for key, value in feature_flags.items()})
        self.allowed_tools = self._normalize_allowed_tools(allowed_tools)
        self.tool_use_context_override = tool_use_context
        if tool_use_context is not None and tool_use_context.root.resolve() != self.root.resolve():
            raise ValueError("tool_use_context.root must match the runtime workspace root")
        self.use_exact_tools = bool(use_exact_tools)
        self.agent_instructions = str(agent_instructions).strip()
        self.run_store = run_store or RunStore(Path(workspace.repo_root) / ".nano" / "runs")
        self.session = session or {
            "id": datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
            "created_at": now(),
            "workspace_root": workspace.repo_root,
            "history": [],
            "memory": memorylib.default_memory_state(),
        }
        self._ensure_session_shape()
        self.memory = memorylib.LayeredMemory(
            self.session.setdefault("memory", memorylib.default_memory_state()),
            workspace_root=self.root,
        )
        self.session["memory"] = self.memory.to_dict()
        self.tools = self._apply_tool_allowlist(self.build_tools())
        self.tool_executor = ToolExecutor(self)
        self._skill_prompt_hash = ""
        self.prefix_state = self.build_prefix()
        self.prefix = self.prefix_state.text
        self.context_manager = ContextManager(self)
        self.resume_state = self.evaluate_resume_state()
        self.session_path = self.session_store.save(self.session)
        self.current_task_state: TaskState | None = None
        self.current_run_dir: Path | None = None
        self._current_query_task: asyncio.Task[Any] | None = None
        self._active_tool_tasks = set()
        self.last_input_token_count = int(self.session.get("last_input_token_count", 0))
        self.last_api_call_time = 0.0
        self.last_prompt_metadata = {}
        self.last_completion_metadata = {}
        self._memory_prompt_hash = ""
        self._last_tool_result_metadata = {}
        self._last_prefix_refresh = {
            "workspace_changed": False,
            "prefix_changed": False,
        }

    @classmethod
    def from_session(
        cls,
        model_client: ModelClient,
        workspace: WorkspaceContext,
        session_store: SessionStore,
        session_id: str,
        **kwargs: Any,
    ) -> "AgentRuntime":
        return cls(
            model_client=model_client,
            workspace=workspace,
            session_store=session_store,
            session=session_store.load(session_id),
            **kwargs,
        )

    def _ensure_session_shape(self) -> None:
        self.session.setdefault("history", [])
        self.session.setdefault("memory", memorylib.default_memory_state())
        checkpoints = self.session.setdefault("checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            self.session["checkpoints"] = checkpoints
        checkpoints.setdefault("current_id", "")
        checkpoints.setdefault("items", {})
        runtime_identity = self.session.setdefault("runtime_identity", {})
        if not isinstance(runtime_identity, dict):
            self.session["runtime_identity"] = {}
        resume_state = self.session.setdefault("resume_state", {})
        if not isinstance(resume_state, dict):
            self.session["resume_state"] = {}
        self.session.setdefault("last_input_token_count", 0)

    def current_runtime_identity(self) -> dict[str, Any]:
        from nano.runtime import checkpoint as checkpointlib

        return checkpointlib.current_runtime_identity(self)

    def checkpoint_state(self) -> dict[str, Any]:
        from nano.runtime import checkpoint as checkpointlib

        return checkpointlib.checkpoint_state(self)

    def current_checkpoint(self) -> dict[str, Any] | None:
        from nano.runtime import checkpoint as checkpointlib

        return checkpointlib.current_checkpoint(self)

    def invalidate_stale_memory(self) -> list[str]:
        invalidated = self.memory.invalidate_stale_file_summaries()
        self.session["memory"] = self.memory.to_dict()
        return invalidated

    def evaluate_resume_state(self) -> dict[str, Any]:
        from nano.runtime import checkpoint as checkpointlib

        return checkpointlib.evaluate_resume_state(self)

    def render_checkpoint_text(self) -> str:
        from nano.runtime import checkpoint as checkpointlib

        return checkpointlib.render_checkpoint_text(self)

    @staticmethod
    def remember(bucket: list[Any], item: Any, limit: int) -> None:
        if not item:
            return
        if item in bucket:
            bucket.remove(item)
        bucket.append(item)
        del bucket[:-limit]

    def build_tools(self) -> dict[str, Tool[Any, str, ToolProgressData]]:
        return toolkit.build_tool_registry(self.tool_context())

    @staticmethod
    def _normalize_allowed_tools(allowed_tools: Iterable[str] | None) -> tuple[str, ...] | None:
        if allowed_tools is None:
            return None
        normalized = tuple(str(name).strip() for name in allowed_tools)
        if not normalized or any(not name for name in normalized):
            raise ValueError("allowed_tools must be a non-empty sequence of tool names")
        return normalized

    def _apply_tool_allowlist(self, tools: dict[str, Tool[Any, str, ToolProgressData]]) -> dict[str, Tool[Any, str, ToolProgressData]]:
        if self.allowed_tools is None:
            return tools
        legal_names = toolkit.legal_tool_names()
        unknown = [name for name in self.allowed_tools if name not in legal_names]
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        allowed = set(self.allowed_tools)
        return {
            name: tool
            for name, tool in tools.items()
            if name in allowed
        }

    def tool_signature(self) -> str:
        return tool_signature(self.tools)

    def build_prefix(self) -> PromptPrefix:
        memory_prompt_section = self.memory.memory_prompt_section() if self.feature_enabled("memory") else ""
        self._memory_prompt_hash = hashlib.sha256(memory_prompt_section.encode("utf-8")).hexdigest()
        skill_descriptions = build_skill_descriptions(self.root)
        self._skill_prompt_hash = hashlib.sha256(skill_descriptions.encode("utf-8")).hexdigest()
        return build_prompt_prefix(
            workspace=self.workspace,
            tools=self.tools,
            memory_prompt_section=memory_prompt_section,
            skill_descriptions=skill_descriptions,
            agent_instructions=self.agent_instructions,
            native_tool_calls=self.model_client.supports_native_tool_calls,
        )

    def _apply_prefix_state(self, prefix_state: PromptPrefix) -> None:
        self.prefix_state = prefix_state
        self.prefix = prefix_state.text

    def refresh_prefix(self, force: bool = False) -> dict[str, bool]:
        previous_hash = self.prefix_state.hash
        previous_workspace_fingerprint = self.prefix_state.workspace_fingerprint
        previous_memory_prompt_hash = self._memory_prompt_hash

        # 工作区事实相对稳定，所以这里按整体刷新；
        # 只有这些事实真的变化了，才重建完整 prefix。
        refreshed_workspace = WorkspaceContext.build(self.root)
        refreshed_workspace_fingerprint = refreshed_workspace.fingerprint()
        workspace_changed = force or refreshed_workspace_fingerprint != previous_workspace_fingerprint
        if workspace_changed:
            self.workspace = refreshed_workspace

        current_memory_prompt_hash = hashlib.sha256(
            (self.memory.memory_prompt_section() if self.feature_enabled("memory") else "").encode("utf-8")
        ).hexdigest()
        current_skill_prompt_hash = hashlib.sha256(build_skill_descriptions(self.root).encode("utf-8")).hexdigest()
        memory_changed = current_memory_prompt_hash != previous_memory_prompt_hash
        skills_changed = current_skill_prompt_hash != self._skill_prompt_hash
        prefix_state = self.build_prefix() if workspace_changed or memory_changed or skills_changed or force or previous_hash is None else self.prefix_state
        prefix_changed = force or previous_hash != prefix_state.hash
        if prefix_changed:
            self._apply_prefix_state(prefix_state)

        self._last_prefix_refresh = {
            "workspace_changed": workspace_changed,
            "prefix_changed": prefix_changed,
        }
        return dict(self._last_prefix_refresh)

    def memory_text(self) -> str:
        return self.memory.render_memory_text()

    def side_query(self, system_prompt: str, user_prompt: str) -> str:
        """复用当前配置模型执行低成本语义记忆选择。"""
        return str(self.model_client.complete(f"{system_prompt}\n\n{user_prompt}", 512, prompt_cache_key=None, prompt_cache_retention=None))

    def start_memory_prefetch(self, user_message: str):
        """为顶层请求异步启动语义记忆召回。"""
        if self.depth > 0 or not self.feature_enabled("memory") or not self.feature_enabled("relevant_memory"):
            return None
        return self.memory.start_memory_prefetch(user_message, self.side_query)

    def history_text(self) -> str:
        history = self.session["history"]
        if not history:
            return "- empty"

        lines = []
        seen_reads = set()
        recent_start = max(0, len(history) - 6)
        for index, item in enumerate(history):
            recent = index >= recent_start
            if item["role"] == "tool" and item["name"] == "read_file" and not recent:
                path = str(item["args"].get("path", ""))
                if path in seen_reads:
                    continue
                seen_reads.add(path)

            if item["role"] == "tool":
                limit = 900 if recent else 180
                lines.append(f"[tool:{item['name']}] {json.dumps(item['args'], sort_keys=True)}")
                lines.append(clip(item["content"], limit))
            else:
                limit = 900 if recent else 220
                lines.append(f"[{item['role']}] {clip(item['content'], limit)}")

        return clip("\n".join(lines), MAX_HISTORY)

    def feature_enabled(self, name: str) -> bool:
        return bool(self.feature_flags.get(str(name), False))

    def prompt(self, user_message: str) -> str:
        prompt, _ = self._build_prompt_and_metadata(user_message)
        return prompt

    def record(self, item: dict[str, Any]) -> None:
        self.session["history"].append(item)
        self.session_path = self.session_store.save(self.session)

    def record_conversation(self, item: dict[str, Any]) -> None:
        """追加一条模型侧消息，供 autoCompact 在上下文接近上限时汇总。"""
        conversation = self.session.setdefault("conversation", [])
        conversation.append(item)
        self.session_path = self.session_store.save(self.session)

    @staticmethod
    def looks_sensitive_env_name(name):
        return securitylib.looks_sensitive_env_name(name)

    def is_secret_env_name(self, name):
        return securitylib.is_secret_env_name(name, secret_env_names=self.secret_env_names)

    def configured_secret_env_items(self):
        return securitylib.configured_secret_env_items(secret_env_names=self.secret_env_names)

    def detected_secret_env_items(self):
        return securitylib.detected_secret_env_items(secret_env_names=self.secret_env_names)

    def secret_env_summary(self):
        return securitylib.secret_env_summary(secret_env_names=self.secret_env_names)

    def detected_secret_env_summary(self):
        return securitylib.detected_secret_env_summary(secret_env_names=self.secret_env_names)

    def redact_text(self, text):
        return securitylib.redact_text(text, secret_env_names=self.secret_env_names)

    def redact_artifact(self, value, key=None):
        return securitylib.redact_artifact(value, key=key, secret_env_names=self.secret_env_names)

    def shell_env(self):
        return securitylib.shell_env(allowlist=self.shell_env_allowlist, root=self.root)

    def prompt_metadata(self, user_message, prompt):
        _, metadata = self._build_prompt_and_metadata(user_message)
        return metadata

    def anthropic_system_blocks(self) -> list[dict[str, Any]]:
        """构建 Anthropic 显式缓存的静态 system block 与未缓存动态尾巴。"""
        blocks: list[dict[str, Any]] = [{"type": "text", "text": self.prefix_state.static_system, "cache_control": {"type": "ephemeral"}}]
        dynamic_text = self.prefix_state.dynamic_system
        checkpoint_text = self.render_checkpoint_text().strip()
        if checkpoint_text:
            dynamic_text = dynamic_text + "\n\n" + checkpoint_text
        if dynamic_text:
            blocks.append({"type": "text", "text": dynamic_text})
        return blocks

    def anthropic_user_system_reminder(self) -> str:
        """返回注入 Anthropic 首条 user 消息的项目级系统提醒。"""
        return self.prefix_state.user_system_reminder

    def _build_prompt_and_metadata(self, user_message, include_prefix: bool = True, relevant_memories=None):
        from nano.runtime.checkpoint import CHECKPOINT_NONE_STATUS

        refresh = self.refresh_prefix()
        self.resume_state = self.evaluate_resume_state()
        prompt, metadata = self.context_manager.build(
            user_message,
            include_prefix=include_prefix,
            relevant_memories=relevant_memories,
        )
        # 这里把“这轮 prompt 是怎么拼出来的”连同缓存相关状态一起记下来，
        # 后面 trace/report 才能解释清楚：为什么这一轮 prefix 变了、缓存有没有命中。
        metadata.update(
            {
                "prefix_chars": len(self.prefix),
                "workspace_chars": len(self.workspace.text()),
                "memory_chars": len(self.memory_text()),
                "history_chars": len(self.history_text()),
                "request_chars": len(user_message),
                "tool_count": len(self.tools),
                "workspace_docs": len(self.workspace.project_docs),
                "recent_commits": len(self.workspace.recent_commits),
                "prefix_hash": self.prefix_state.hash,
                "prompt_cache_key": f"session:{self.session['id']}",
                "workspace_fingerprint": self.prefix_state.workspace_fingerprint,
                "tool_signature": self.prefix_state.tool_signature,
                "workspace_changed": refresh["workspace_changed"],
                "prefix_changed": refresh["prefix_changed"],
                "prompt_cache_supported": self.model_client.supports_prompt_cache,
                "resume_status": self.resume_state.get("status", CHECKPOINT_NONE_STATUS),
                "stale_summary_invalidations": int(self.resume_state.get("stale_summary_invalidations", 0)),
                "stale_paths": list(self.resume_state.get("stale_paths", [])),
                "runtime_identity_mismatch_fields": list(self.resume_state.get("runtime_identity_mismatch_fields", [])),
            }
        )
        metadata.update(self.detected_secret_env_summary())
        return prompt, metadata

    def emit_trace(self, task_state, event, payload=None):
        payload = self.redact_artifact(payload or {})
        payload["event"] = event
        payload["created_at"] = now()
        # trace 是运行中的逐事件时间线，适合回答“这一轮 agent 到底做了什么”。
        self.run_store.append_trace(task_state, payload)
        return payload

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts):
                continue
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        all_paths = sorted(set(before) | set(after))
        for path in all_paths:
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        from nano.runtime import checkpoint as checkpointlib

        return checkpointlib.create_checkpoint(self, task_state, user_message, trigger)

    def infer_next_step(self, task_state):
        from nano.runtime import checkpoint as checkpointlib

        return checkpointlib.infer_next_step(task_state)

    def update_memory_after_tool(self, name, args, result):
        """把少量高价值工具结果沉淀到 working memory。

        为什么存在：
        并不是每个工具结果都值得长期带进下一轮 prompt。完整结果已经进了
        `history`，这里只挑少量“下一轮大概率还会用到”的事实做提纯，
        例如最近读写过哪些文件、某个文件读出来的短摘要。

        输入 / 输出：
        - 输入：工具名 `name`、参数 `args`、执行结果 `result`
        - 输出：无显式返回值，副作用是更新 `self.memory`

        在 agent 链路里的位置：
        它发生在 `run_tool()` 真正执行完工具之后、下一轮 prompt 组装之前。
        也就是说：工具结果先进入完整历史，再由这个函数择优沉淀成轻量记忆。
        """
        if not self.feature_enabled("memory"):
            return
        path = args.get("path")
        if not path:
            return

        if name in {"write_file", "patch_file"} and self.memory.memory_dir is not None:
            resolved_path = self.path(path)
            if resolved_path.parent == self.memory.memory_dir and resolved_path.suffix == ".md" and resolved_path.name != "MEMORY.md":
                # Agent 通过通用写文件工具保存记忆后，由运行时保持索引与文件一致。
                memorylib.update_memory_index(self.memory.memory_dir)

        canonical_path = self.memory.canonical_path(path)
        # 不是所有工具结果都进入工作记忆。
        # 读文件会生成摘要；写文件/patch 会让旧摘要失效，因为它们可能过期了。
        if name in {"read_file", "write_file", "patch_file"}:
            self.memory.remember_file(canonical_path)
        if name == "read_file":
            summary = memorylib.summarize_read_result(result)
            self.memory.set_file_summary(canonical_path, summary)
            self.memory.append_note(summary, tags=(canonical_path,), source=canonical_path)
        elif name in {"write_file", "patch_file"}:
            self.memory.invalidate_file_summary(canonical_path)

    def note_tool(self, name, args, result):
        self.update_memory_after_tool(name, args, result)

    def record_process_note_for_tool(self, name, metadata):
        status = str(metadata.get("tool_status", "")).strip()
        if status not in {"partial_success", "error", "rejected"}:
            return
        affected_paths = [str(path).strip() for path in metadata.get("affected_paths", []) if str(path).strip()]
        path_text = ", ".join(affected_paths) or "workspace"
        if status == "partial_success":
            text = f"{name} partial_success on {path_text}; inspect diff before retry"
        elif status == "error":
            text = f"{name} error on {path_text}; check the failure before retry"
        else:
            text = f"{name} rejected; choose a different action before retry"
        tags = ["process", status, *affected_paths]
        self.memory.append_note(text, tags=tuple(tags), source=name, kind="process")
        self.session["memory"] = self.memory.to_dict()

    async def ask_async(self, user_message):
        """异步执行一条请求并返回最终答案。"""
        from nano.runtime.agent_loop import QueryEngine

        return await QueryEngine(self).run_async(user_message)

    def interrupt_current_request(self):
        """请求取消当前正在运行的模型查询。"""
        task = self._current_query_task
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def execute_tool(self, name, args):
        result = self.tool_executor.execute(name, args)
        self._last_tool_result_metadata = dict(result.metadata)
        return result

    def persist_tool_result(self, tool_name, content):
        """将当前运行中超长工具输出保存为可审计工件。"""
        if self.current_task_state is None or self.current_run_dir is None:
            return ""
        path = self.run_store.write_tool_result(
            self.current_task_state,
            self.current_task_state.tool_steps + 1,
            str(tool_name),
            self.redact_text(content),
        )
        return str(path.relative_to(self.root))

    def run_tool(self, name, args):
        """执行一次工具调用，并在执行前后套上完整护栏。

        为什么存在：
        在 agent 系统里，真正危险的不是“模型会不会想调用工具”，而是
        “平台有没有在执行前把边界守住”。这个函数就是工具层的总闸口：
        所有工具调用都必须先经过它，不能让模型直接碰到底层函数。

        输入 / 输出：
        - 输入：工具名 `name`，参数字典 `args`
        - 输出：字符串结果。无论是成功结果还是错误信息，都会统一返回文本，
          这样模型下一轮都能继续消费这份反馈。

        在 agent 链路里的位置：
        它位于 `ask_async()` 的“模型决定要调用工具”之后，是控制循环里真正把模型
        意图落到外部世界的一步。因此这里串起了几乎所有安全与可控设计：
        工具是否存在、参数是否合法、是否重复、是否需要审批、执行结果是否裁剪、
        是否需要回写记忆。
        """
        return self.execute_tool(name, args).content

    def repeated_tool_call(self, name, args):
        # agent 很常见的一种坏循环，是在没有新信息的情况下反复发起同一调用。
        # 这里提前挡掉最简单的这种循环。
        tool_events = [item for item in self.session["history"] if item["role"] == "tool"]
        if len(tool_events) < 2:
            return False
        normalized_args = toolkit.validate_tool_arguments(name, args)
        recent = tool_events[-2:]
        return all(
            item["name"] == name
            and toolkit.validate_tool_arguments(name, item.get("args", {})) == normalized_args
            for item in recent
        )

    @staticmethod
    def new_task_id():
        return "task_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    @staticmethod
    def new_run_id():
        return "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def build_report(self, task_state):
        # report 是一次运行的最终摘要；
        # 和 trace 的区别在于，trace 关注过程，report 关注结果与关键指标。
        return {
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": task_state.final_answer,
            "tool_steps": task_state.tool_steps,
            "attempts": task_state.attempts,
            "checkpoint_id": task_state.checkpoint_id,
            "resume_status": task_state.resume_status,
            "task_state": task_state.to_dict(),
            "prompt_metadata": self.last_prompt_metadata,
            "redacted_env": self.detected_secret_env_summary(),
        }

    def tool_example(self, name):
        return toolkit.tool_example(name)

    def validate_tool(self, name, args):
        """把通用工具校验和 runtime 级额外约束串起来。"""
        return toolkit.validate_tool(self.tool_context(), name, args)

    def tool_context(self):
        if self.tool_use_context_override is not None:
            return self.tool_use_context_override
        return ToolContext(
            root=self.root,
            path_resolver=self.path,
            shell_env_provider=self.shell_env,
            depth=self.depth,
            max_depth=self.max_depth,
            spawn_delegate=self.spawn_delegate,
            read_file_state=self.read_file_state,
            permissions=self.permissions,
        )

    async def _consume_subagent_stream(self, task: str) -> str:
        """由子 agent 自行消费自身 QueryLoop 事件并返回终止答案。"""
        from nano.runtime.agent_loop import QueryEngine

        async for event in QueryEngine(self).stream_async(task):
            if event.type in {"final", "error", "stopped"}:
                return str(event.payload["answer"])
        raise RuntimeError("subagent stream ended without a terminal event")

    async def start_subagent(self, args) -> SubagentHandle:
        """异步创建并启动只读子 agent，立即返回可等待的运行句柄。"""
        task = str(args.get("task", "")).strip()
        child = AgentRuntime(
            model_client=self.model_client,
            workspace=self.workspace,
            session_store=self.session_store,
            run_store=self.run_store,
            approval_policy="never",
            max_steps=int(args.get("max_steps", 3)),
            max_new_tokens=self.max_new_tokens,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            read_only=True,
            secret_env_names=self.secret_env_names,
            shell_env_allowlist=self.shell_env_allowlist,
            allowed_tools=self.allowed_tools if self.use_exact_tools else None,
            use_exact_tools=self.use_exact_tools,
        )
        # 委派的目标是“调查”，不是“放权执行”。
        # 子 agent 以只读方式运行、步数更少，最后只把结论文本返回给父 agent。
        child.memory.set_task_summary(task)
        child.memory.append_note(clip(self.history_text(), 300), source="parent_history")
        child.session["memory"] = child.memory.to_dict()
        return SubagentHandle(child, asyncio.create_task(child._consume_subagent_stream(task)))

    async def spawn_delegate(self, args) -> str:
        """启动子 agent 并等待其调查结论，供异步 delegate 工具调用。"""
        subagent = await self.start_subagent(args)
        return "delegate_result:\n" + await subagent.wait()

    def tool_list_files(self, args):
        return toolkit.tool_list_files(self.tool_context(), args)

    def tool_read_file(self, args):
        return toolkit.tool_read_file(self.tool_context(), args)

    def tool_search(self, args):
        return toolkit.tool_search(self.tool_context(), args)

    def tool_run_shell(self, args):
        return toolkit.tool_run_shell(self.tool_context(), args)

    def tool_write_file(self, args):
        return toolkit.tool_write_file(self.tool_context(), args)

    def tool_patch_file(self, args):
        return toolkit.tool_patch_file(self.tool_context(), args)

    async def tool_delegate(self, args):
        """异步执行 delegate 工具。"""
        return await toolkit.tool_delegate(self.tool_context(), args)

    def approve(self, name, args):
        if self.read_only:
            return False
        if self.approval_policy == "auto":
            return True
        if self.approval_policy == "never":
            return False
        try:
            answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    @staticmethod
    def parse(raw):
        """把模型原始输出解析成 runtime 可执行的动作或最终答案。

        为什么存在：
        模型输出首先是自然语言文本，而 runtime 需要的是结构化决策：
        “这是工具调用”还是“这是最终答案”。如果没有这层解析，后面的工具校验、
        审批和执行链路就没法可靠工作。

        输入 / 输出：
        - 输入：模型返回的原始文本 `raw`
        - 输出：`(kind, payload)`，其中 `kind` 可能是 `tool`、`final`、`retry`

        在 agent 链路里的位置：
        它位于 `model_client.complete()` 之后、`run_tool()` 之前，是模型输出
        进入平台控制流的第一道结构化关口。
        """
        raw = str(raw)
        # 这里支持两种工具格式：
        # 1. <tool>...</tool> 里包 JSON，适合简短调用
        # 2. XML 风格属性/子标签，适合写文件这类多行内容
        if "<tool>" in raw and ("<final>" not in raw or raw.find("<tool>") < raw.find("<final>")):
            body = AgentRuntime.extract(raw, "tool")
            try:
                payload = json.loads(body)
            except Exception:
                return "retry", AgentRuntime.retry_notice("model returned malformed tool JSON")
            if not isinstance(payload, dict):
                return "retry", AgentRuntime.retry_notice("tool payload must be a JSON object")
            if not str(payload.get("name", "")).strip():
                return "retry", AgentRuntime.retry_notice("tool payload is missing a tool name")
            args = payload.get("args", {})
            if args is None:
                payload["args"] = {}
            elif not isinstance(args, dict):
                return "retry", AgentRuntime.retry_notice()
            return "tool", payload
        if "<tool" in raw and ("<final>" not in raw or raw.find("<tool") < raw.find("<final>")):
            payload = AgentRuntime.parse_xml_tool(raw)
            if payload is not None:
                return "tool", payload
            return "retry", AgentRuntime.retry_notice()
        if "<final>" in raw:
            final = AgentRuntime.extract(raw, "final").strip()
            if final:
                return "final", final
            return "retry", AgentRuntime.retry_notice("model returned an empty <final> answer")
        raw = raw.strip()
        if raw:
            return "final", raw
        return "retry", AgentRuntime.retry_notice("model returned an empty response")

    @staticmethod
    def retry_notice(problem=None):
        prefix = "Runtime notice"
        if problem:
            prefix += f": {problem}"
        else:
            prefix += ": model returned malformed tool output"
        return (
            f"{prefix}. Reply with a valid <tool> call or a non-empty <final> answer. "
            'For multi-line files, prefer <tool name="write_file" path="file.py"><content>...</content></tool>.'
        )

    @staticmethod
    def parse_xml_tool(raw):
        match = re.search(r"<tool(?P<attrs>[^>]*)>(?P<body>.*?)</tool>", raw, re.S)
        if not match:
            return None
        attrs = AgentRuntime.parse_attrs(match.group("attrs"))
        name = str(attrs.pop("name", "")).strip()
        if not name:
            return None

        body = match.group("body")
        args = dict(attrs)
        for key in ("content", "old_text", "new_text", "command", "task", "pattern", "path"):
            if f"<{key}>" in body:
                args[key] = AgentRuntime.extract_raw(body, key)

        body_text = body.strip("\n")
        if name == "write_file" and "content" not in args and body_text:
            args["content"] = body_text
        if name == "delegate" and "task" not in args and body_text:
            args["task"] = body_text.strip()
        return {"name": name, "args": args}

    @staticmethod
    def parse_attrs(text):
        attrs = {}
        for match in re.finditer(r"""([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""", text):
            attrs[match.group(1)] = match.group(2) if match.group(2) is not None else match.group(3)
        return attrs

    @staticmethod
    def extract(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:].strip()
        return text[start:end].strip()

    @staticmethod
    def extract_raw(text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start = text.find(start_tag)
        if start == -1:
            return text
        start += len(start_tag)
        end = text.find(end_tag, start)
        if end == -1:
            return text[start:]
        return text[start:end]

    def reset(self):
        """清空持久化历史、短对话和工作记忆，以开启新会话。"""
        self.session["history"] = []
        self.session["conversation"] = []
        self.session["memory"].clear()
        self.session["memory"].update(memorylib.default_memory_state())
        self.memory = memorylib.LayeredMemory(self.session["memory"], workspace_root=self.root)
        self.session_store.save(self.session)

    def path(self, raw_path):
        path = Path(raw_path)
        path = path if path.is_absolute() else self.root / path
        resolved = path.resolve()
        # 所有文件类工具都被锚定在 workspace root 之下。
        # 这样既能防住 "../" 逃逸，也能防住符号链接解析后跳出仓库。
        if os.path.commonpath([str(self.root), str(resolved)]) != str(self.root):
            raise ValueError(f"path escapes workspace: {raw_path}")
        return resolved
