"""Checkpoint and resume-state helpers."""

import hashlib
import uuid
from typing import Any

from nano.memory import memory as memorylib
from nano.runtime.runtime import AgentRuntime
from nano.runtime.task_state import TaskState
from nano.storage.schemas import CheckpointStateModel
from nano.utils.text import clip, now

CHECKPOINT_SCHEMA_VERSION = "phase1-v1"
CHECKPOINT_NONE_STATUS = "no-checkpoint"
CHECKPOINT_FULL_VALID_STATUS = "full-valid"
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"
CHECKPOINT_SCHEMA_MISMATCH_STATUS = "schema-mismatch"

RUNTIME_IDENTITY_KEYS = (
    "cwd",
    "model",
    "model_client",
    "approval_policy",
    "read_only",
    "max_steps",
    "max_turns",
    "agent_instructions_hash",
    "use_exact_tools",
    "max_new_tokens",
    "max_final_tokens",
    "max_final_retries",
    "feature_flags",
    "shell_env_allowlist",
    "tool_signature",
)


def current_runtime_identity(runtime: AgentRuntime) -> dict[str, Any]:
    return {
        "session_id": runtime.session.get("id", ""),
        "cwd": str(runtime.root),
        "model": runtime.model_client.model,
        "model_client": runtime.model_client.__class__.__name__,
        "approval_policy": runtime.approval_policy,
        "read_only": bool(runtime.read_only),
        "max_steps": int(runtime.max_steps),
        "max_turns": int(runtime.max_turns),
        "agent_instructions_hash": hashlib.sha256(runtime.agent_instructions.encode("utf-8")).hexdigest(),
        "use_exact_tools": bool(runtime.use_exact_tools),
        "max_new_tokens": int(runtime.max_new_tokens),
        "max_final_tokens": int(runtime.max_final_tokens),
        "max_final_retries": int(runtime.max_final_retries),
        "feature_flags": dict(runtime.feature_flags),
        "shell_env_allowlist": list(runtime.shell_env_allowlist),
        "workspace_fingerprint": runtime.prefix_state.workspace_fingerprint,
        "tool_signature": runtime.tool_signature(),
    }


def checkpoint_state(runtime: AgentRuntime) -> dict[str, Any]:
    """校验并返回当前会话的 checkpoint 字典。"""
    runtime._ensure_session_shape()
    model = CheckpointStateModel.model_validate(runtime.session["checkpoints"])
    runtime.session["checkpoints"] = model.model_dump(mode="python")
    return runtime.session["checkpoints"]


def current_checkpoint(runtime: AgentRuntime) -> dict[str, Any] | None:
    state = checkpoint_state(runtime)
    checkpoint_id = str(state.get("current_id", "")).strip()
    if not checkpoint_id:
        return None
    return state.get("items", {}).get(checkpoint_id)


def evaluate_resume_state(runtime: AgentRuntime) -> dict[str, Any]:
    previous_resume_state = dict(runtime.session.get("resume_state", {}) or {})
    invalidated = runtime.invalidate_stale_memory()
    checkpoint = current_checkpoint(runtime)
    status = CHECKPOINT_NONE_STATUS
    stale_paths = list(invalidated)
    mismatch_fields = []
    if checkpoint:
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            status = CHECKPOINT_SCHEMA_MISMATCH_STATUS
        else:
            for item in checkpoint.get("key_files", []):
                path = str(item.get("path", "")).strip()
                if not path:
                    continue
                expected = item.get("freshness")
                current = memorylib.file_freshness(path, runtime.root)
                if expected != current and path not in stale_paths:
                    stale_paths.append(path)
            saved_identity = dict(checkpoint.get("runtime_identity", {}) or runtime.session.get("runtime_identity", {}) or {})
            current_identity = current_runtime_identity(runtime)
            for key in RUNTIME_IDENTITY_KEYS:
                if key not in saved_identity:
                    continue
                if saved_identity.get(key) != current_identity.get(key):
                    mismatch_fields.append(key)
            mismatch_fields.sort()
            if stale_paths:
                status = CHECKPOINT_PARTIAL_STALE_STATUS
            elif mismatch_fields:
                status = CHECKPOINT_WORKSPACE_MISMATCH_STATUS
            else:
                status = CHECKPOINT_FULL_VALID_STATUS

    resume_state = {
        "status": status,
        "stale_paths": stale_paths,
        "runtime_identity_mismatch_fields": mismatch_fields,
        "stale_summary_invalidations": max(
            len(invalidated),
            int(previous_resume_state.get("stale_summary_invalidations", 0))
            if status == CHECKPOINT_PARTIAL_STALE_STATUS
            else 0,
        ),
    }
    runtime.session["resume_state"] = resume_state
    runtime.session["runtime_identity"] = current_runtime_identity(runtime)
    return resume_state


def render_checkpoint_text(runtime: AgentRuntime) -> str:
    checkpoint = current_checkpoint(runtime)
    if not checkpoint:
        return ""
    lines = [
        "Task checkpoint:",
        f"- Resume status: {runtime.resume_state.get('status', CHECKPOINT_NONE_STATUS)}",
        f"- Current goal: {checkpoint.get('current_goal', '-') or '-'}",
        f"- Current blocker: {checkpoint.get('current_blocker', '-') or '-'}",
        f"- Next step: {checkpoint.get('next_step', '-') or '-'}",
    ]
    key_files = [str(item.get("path", "")).strip() for item in checkpoint.get("key_files", []) if str(item.get("path", "")).strip()]
    lines.append(f"- Key files: {', '.join(key_files) or '-'}")
    if checkpoint.get("completed"):
        lines.append("- Completed: " + " | ".join(str(item) for item in checkpoint.get("completed", [])))
    if checkpoint.get("excluded"):
        lines.append("- Excluded: " + " | ".join(str(item) for item in checkpoint.get("excluded", [])))
    if runtime.resume_state.get("stale_paths"):
        lines.append("- Stale paths: " + ", ".join(runtime.resume_state["stale_paths"]))
    summary = str(checkpoint.get("summary", "")).strip()
    if summary:
        lines.append(f"- Summary: {summary}")
    return "\n".join(lines)


def infer_next_step(task_state):
    if task_state.status == "completed":
        return "No next step recorded."
    if task_state.stop_reason == "step_limit_reached":
        return "Resume from the latest checkpoint and continue the task."
    if task_state.last_tool:
        return f"Decide the next action after {task_state.last_tool}."
    return "Continue the task from the latest checkpoint."


def create_checkpoint(runtime: AgentRuntime, task_state: TaskState, user_message: str, trigger: str) -> dict[str, Any]:
    current = current_checkpoint(runtime)
    state = checkpoint_state(runtime)
    checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
    key_files = []
    freshness = {}
    for path in runtime.memory.to_dict()["working"]["recent_files"]:
        file_freshness = memorylib.file_freshness(path, runtime.root)
        freshness[path] = file_freshness
        key_files.append({"path": path, "freshness": file_freshness})
    checkpoint = {
        "checkpoint_id": checkpoint_id,
        "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": now(),
        "current_goal": str(user_message),
        "completed": [task_state.final_answer] if task_state.final_answer else [],
        "excluded": [],
        "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
        "next_step": infer_next_step(task_state),
        "key_files": key_files,
        "freshness": freshness,
        "summary": f"{trigger}: {clip(str(user_message), 120)}",
        "runtime_identity": current_runtime_identity(runtime),
    }
    state["items"][checkpoint_id] = checkpoint
    state["current_id"] = checkpoint_id
    task_state.checkpoint_id = checkpoint_id
    runtime.session["runtime_identity"] = checkpoint["runtime_identity"]
    runtime.session_path = runtime.session_store.save(runtime.session)
    return checkpoint
