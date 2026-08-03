"""会话工作记忆和按项目隔离的文件记忆。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from nano.memory.frontmatter import format_frontmatter, parse_frontmatter
from nano.utils import clip, now

WORKING_FILE_LIMIT = 8
EPISODIC_NOTE_LIMIT = 12
FILE_SUMMARY_LIMIT = 6
MEMORY_TYPES = ("user", "feedback", "project", "reference")
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000
MAX_MEMORY_BYTES_PER_FILE = 4_000
MAX_SURFACED_MEMORY_BYTES = 60_000
MAX_RELEVANT_MEMORIES = 5

SELECT_MEMORIES_PROMPT = """You are selecting memories that will be useful to an AI coding assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected_memories" array of filenames for the memories that will clearly be useful (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful, do not include it.
- If no memories would clearly be useful, return an empty array."""


@dataclass(frozen=True)
class MemoryHeader:
    """索引和语义选择所需的记忆文件头信息。"""

    name: str
    description: str
    type: str
    filename: str
    file_path: Path
    mtime_ms: float


@dataclass(frozen=True)
class RelevantMemory:
    """已读取并可注入当前模型请求的记忆。"""

    path: Path
    filename: str
    content: str
    mtime_ms: float
    header: str


def memory_project_hash(cwd: Path | None = None) -> str:
    """返回当前进程工作目录的稳定 SHA-256 前缀。"""
    project_path = (cwd or Path.cwd()).resolve()
    return hashlib.sha256(str(project_path).encode("utf-8")).hexdigest()[:16]


def get_memory_dir(workspace_root: Path | str | None = None) -> Path:
    """返回当前进程目录对应的项目文件记忆目录。"""
    return Path(workspace_root or Path.cwd()) / ".nano" / "projects" / memory_project_hash() / "memory"


def _get_index_path(memory_dir: Path) -> Path:
    """返回指定记忆目录内的索引路径。"""
    return memory_dir / "MEMORY.md"


def _slugify(name: str) -> str:
    """生成不会逃逸目录的稳定文件名片段。"""
    normalized = unicodedata.normalize("NFKC", str(name)).strip().lower()
    slug = re.sub(r"[^\w]+", "-", normalized, flags=re.UNICODE).strip("-_")
    return slug or "memory"


def list_memories(memory_dir: Path | None = None) -> list[MemoryHeader]:
    """读取记忆目录内全部有效记忆的 frontmatter 头。"""
    memory_dir = memory_dir or get_memory_dir()
    if not memory_dir.exists():
        return []
    memories = []
    for path in sorted(memory_dir.glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        parsed = parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        memory_type = parsed.meta.get("type", "").strip()
        name = parsed.meta.get("name", "").strip()
        description = parsed.meta.get("description", "").strip()
        if memory_type not in MEMORY_TYPES or not name or not description:
            continue
        memories.append(
            MemoryHeader(
                name=name,
                description=description,
                type=memory_type,
                filename=path.name,
                file_path=path,
                mtime_ms=path.stat().st_mtime * 1000,
            )
        )
    return memories


def save_memory(name: str, description: str, type: str, content: str, memory_dir: Path | None = None) -> str:
    """保存一条文件记忆，并重建其 Markdown 索引。"""
    if type not in MEMORY_TYPES:
        raise ValueError(f"unsupported memory type: {type}")
    memory_dir = memory_dir or get_memory_dir()
    memory_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{type}_{_slugify(name)}.md"
    text = format_frontmatter(
        {"name": name, "description": description, "type": type},
        content,
    )
    (memory_dir / filename).write_text(text, encoding="utf-8")
    update_memory_index(memory_dir)
    return filename


def update_memory_index(memory_dir: Path | None = None) -> None:
    """用当前记忆文件头重建 MEMORY.md 索引。"""
    memory_dir = memory_dir or get_memory_dir()
    memory_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Memory Index", ""]
    for memory in list_memories(memory_dir):
        lines.append(f"- **[{memory.name}]({memory.filename})** ({memory.type}) - {memory.description}")
    _get_index_path(memory_dir).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_memory_index(memory_dir: Path | None = None) -> str:
    """读取索引，并分别以行数和字节数防止异常上下文占用。"""
    memory_dir = memory_dir or get_memory_dir()
    index_path = _get_index_path(memory_dir)
    if not index_path.exists():
        return ""
    content = index_path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")
    if len(lines) > MAX_INDEX_LINES:
        content = "\n".join(lines[:MAX_INDEX_LINES]) + "\n\n[... truncated, too many memory entries ...]"
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_INDEX_BYTES:
        content = encoded[:MAX_INDEX_BYTES].decode("utf-8", errors="ignore") + "\n\n[... truncated, index too large ...]"
    return content


def build_memory_prompt_section(memory_dir: Path | None = None) -> str:
    """构建说明文件记忆边界、格式和当前索引的系统提示词片段。"""
    memory_dir = memory_dir or get_memory_dir()
    index = load_memory_index(memory_dir)
    return f"""# Memory System

You have a persistent, file-based memory system at `{memory_dir}`.

## Memory Types
- **user**: User's role, preferences, knowledge level
- **feedback**: Corrections and guidance from the user
- **project**: Ongoing work, goals, deadlines, decisions
- **reference**: Pointers to external resources

## How to Save Memories
Use the write_file tool to create a memory file in `{memory_dir}`. Name files `{{type}}_{{slugified_name}}.md` and use this format:
---
name: Short memory name
description: A concise retrieval description
type: feedback
---
Memory body with the fact, its context, and how to apply it.

## What NOT to Save
- Code patterns or architecture: read the code instead
- Git history: use git log
- Anything already in AGENTS.md or project instructions
- Ephemeral task details

## Current Memory Index
{index if index else "(No memories saved yet.)"}"""


def scan_memory_headers(memory_dir: Path | None = None) -> list[MemoryHeader]:
    """返回语义召回需要的文件名和描述清单。"""
    return list_memories(memory_dir)


def format_memory_manifest(headers: list[MemoryHeader]) -> str:
    """将候选记忆压缩成供 side query 判断的清单。"""
    return "\n".join(f"- {header.filename}: {header.name} - {header.description}" for header in headers)


def memory_age(mtime_ms: float) -> str:
    """返回适合注入提示词的简短记忆保存时间。"""
    age_seconds = max(0, int(datetime.now(timezone.utc).timestamp() - mtime_ms / 1000))
    if age_seconds < 3600:
        return "less than one hour ago"
    if age_seconds < 86_400:
        return f"{age_seconds // 3600} hours ago"
    return f"{age_seconds // 86_400} days ago"


def memory_freshness_warning(mtime_ms: float) -> str:
    """为超过三十天未修改的记忆标记可能过期。"""
    if datetime.now(timezone.utc).timestamp() - mtime_ms / 1000 > 30 * 86_400:
        return f"Warning: this memory was last updated {memory_age(mtime_ms)} and may be stale."
    return ""


def _truncate_memory_content(content: str, limit: int) -> str:
    """按 UTF-8 字节数截断单条召回记忆。"""
    encoded = content.encode("utf-8")
    if len(encoded) <= limit:
        return content
    notice = "\n\n[... truncated, memory file too large ...]"
    allowed = max(0, limit - len(notice.encode("utf-8")))
    return encoded[:allowed].decode("utf-8", errors="ignore") + notice


def select_relevant_memories(
    query: str,
    side_query: Callable[[str, str], str],
    already_surfaced: set[str],
    memory_dir: Path,
    remaining_bytes: int = MAX_SURFACED_MEMORY_BYTES,
) -> list[RelevantMemory]:
    """用 side query 从未展示记忆中选择最多五条与请求相关的内容。"""
    headers = scan_memory_headers(memory_dir)
    candidates = [header for header in headers if str(header.file_path) not in already_surfaced]
    if not candidates or remaining_bytes <= 0:
        return []
    try:
        text = side_query(SELECT_MEMORIES_PROMPT, f"Query: {query}\n\nAvailable memories:\n{format_memory_manifest(candidates)}")
        match = re.search(r"\{[\s\S]*\}", str(text))
        if not match:
            return []
        selected_filenames = json.loads(match.group(0)).get("selected_memories", [])
        if not isinstance(selected_filenames, list):
            return []
        filename_set = {str(filename) for filename in selected_filenames}
    except Exception:
        # 语义召回不能阻塞主请求，side query 失败时静默跳过。
        return []

    selected = []
    consumed_bytes = 0
    for header in candidates:
        if header.filename not in filename_set or len(selected) >= MAX_RELEVANT_MEMORIES:
            continue
        allowed_bytes = min(MAX_MEMORY_BYTES_PER_FILE, remaining_bytes - consumed_bytes)
        if allowed_bytes <= 0:
            break
        content = _truncate_memory_content(header.file_path.read_text(encoding="utf-8", errors="replace"), allowed_bytes)
        freshness = memory_freshness_warning(header.mtime_ms)
        selected.append(
            RelevantMemory(
                path=header.file_path,
                filename=header.filename,
                content=content,
                mtime_ms=header.mtime_ms,
                header=f"{freshness}\n\nMemory: {header.file_path}:" if freshness else f"Memory (saved {memory_age(header.mtime_ms)}): {header.file_path}:",
            )
        )
        consumed_bytes += len(content.encode("utf-8"))
    return selected


def default_memory_state():
    """创建持久化到 session 的轻量工作记忆状态。"""
    return {
        "working": {"task_summary": "", "recent_files": []},
        "episodic_notes": [],
        "file_summaries": {},
        "next_note_index": 0,
        "surfaced_memory_paths": [],
        "surfaced_memory_bytes": 0,
    }


def _ensure_list(value):
    """将可选标量标准化为列表。"""
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [] if value in (None, "") else [value]


def _dedupe_preserve_order(items):
    """去重并保留最后一次出现的相对顺序。"""
    result = []
    for item in items:
        if item in result:
            result.remove(item)
        result.append(item)
    return result


def resolve_workspace_path(raw_path, workspace_root=None):
    """将工作区相对路径解析为受根目录约束的绝对路径。"""
    path = Path(str(raw_path))
    if workspace_root is None:
        return path
    root = Path(workspace_root).resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def canonicalize_path(raw_path, workspace_root=None):
    """将路径转换为工作区内的稳定相对表示。"""
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or workspace_root is None:
        return Path(str(raw_path)).as_posix()
    return resolved.relative_to(Path(workspace_root).resolve()).as_posix()


def file_freshness(raw_path, workspace_root=None):
    """返回当前文件内容哈希，不存在或越界时返回空值。"""
    resolved = resolve_workspace_path(raw_path, workspace_root)
    if resolved is None or not resolved.is_file():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _normalize_note(data, index):
    """校验当前 session 格式的短笔记。"""
    return {
        "text": clip(str(data.get("text", "")).strip(), 500),
        "tags": _dedupe_preserve_order([str(tag).strip() for tag in _ensure_list(data.get("tags")) if str(tag).strip()]),
        "source": str(data.get("source", "")).strip(),
        "created_at": str(data.get("created_at", "")).strip() or now(),
        "note_index": int(data.get("note_index", index)),
        "kind": str(data.get("kind", "episodic")).strip() or "episodic",
    }


def normalize_memory_state(state: dict[str, Any] | None, workspace_root=None) -> dict[str, Any]:
    """将 session 中可能旧版本的工作记忆标准化。"""
    if state is None:
        state = default_memory_state()
    if not isinstance(state, dict):
        raise TypeError("memory state must be a mapping")
    normalized_state: dict[str, Any] = state
    working_value = normalized_state.get("working")
    working: dict[str, Any] = working_value if isinstance(working_value, dict) else {}
    working["task_summary"] = clip(str(working.get("task_summary", "")).strip(), 300)
    working["recent_files"] = _dedupe_preserve_order(
        [canonicalize_path(path, workspace_root) for path in _ensure_list(working.get("recent_files")) if str(path).strip()]
    )[-WORKING_FILE_LIMIT:]
    normalized_state["working"] = working
    raw_notes = normalized_state.get("episodic_notes")
    if not isinstance(raw_notes, list):
        raw_notes = []
    normalized_state["episodic_notes"] = [note for index, item in enumerate(raw_notes) if isinstance(item, dict) and (note := _normalize_note(item, index))["text"]][-EPISODIC_NOTE_LIMIT:]
    summaries_value = normalized_state.get("file_summaries")
    summaries: dict[str, Any] = summaries_value if isinstance(summaries_value, dict) else {}
    normalized_state["file_summaries"] = {
        canonicalize_path(path, workspace_root): {
            "summary": clip(str(value.get("summary", "") if isinstance(value, dict) else value).strip(), 500),
            "created_at": str(value.get("created_at", "") if isinstance(value, dict) else "") or now(),
            "freshness": value.get("freshness") if isinstance(value, dict) else None,
        }
        for path, value in summaries.items()
        if str(value.get("summary", "") if isinstance(value, dict) else value).strip()
    }
    normalized_state["next_note_index"] = max([note["note_index"] for note in normalized_state["episodic_notes"]], default=-1) + 1
    normalized_state["surfaced_memory_paths"] = _dedupe_preserve_order([str(path) for path in _ensure_list(normalized_state.get("surfaced_memory_paths")) if str(path).strip()])
    normalized_state["surfaced_memory_bytes"] = max(0, int(normalized_state.get("surfaced_memory_bytes", 0)))
    return normalized_state


def set_task_summary(state, summary, workspace_root=None):
    """更新当前任务摘要。"""
    state = normalize_memory_state(state, workspace_root)
    state["working"]["task_summary"] = clip(str(summary).strip(), 300)
    return state


def remember_file(state, path, workspace_root=None):
    """记录最近访问的工作区文件。"""
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    state["working"]["recent_files"] = _dedupe_preserve_order([*state["working"]["recent_files"], path])[-WORKING_FILE_LIMIT:] if path else state["working"]["recent_files"]
    return state


def append_note(state, text, tags=(), source="", created_at=None, workspace_root=None, kind="episodic"):
    """追加一条会话内短笔记，重复正文以最新版本为准。"""
    state = normalize_memory_state(state, workspace_root)
    text = clip(str(text).strip(), 500)
    if not text:
        return state
    note = {
        "text": text,
        "tags": _dedupe_preserve_order([str(tag).strip() for tag in _ensure_list(tags) if str(tag).strip()]),
        "source": str(source).strip(),
        "created_at": str(created_at).strip() if created_at else now(),
        "note_index": state["next_note_index"],
        "kind": str(kind).strip() or "episodic",
    }
    state["next_note_index"] += 1
    state["episodic_notes"] = [item for item in state["episodic_notes"] if item["text"] != text][-EPISODIC_NOTE_LIMIT + 1 :] + [note]
    return state


def set_file_summary(state, path, summary, workspace_root=None):
    """保存带内容新鲜度哈希的文件短摘要。"""
    state = normalize_memory_state(state, workspace_root)
    path = canonicalize_path(path, workspace_root).strip()
    summary = clip(str(summary).strip(), 500)
    if path and summary:
        state["file_summaries"][path] = {"summary": summary, "created_at": now(), "freshness": file_freshness(path, workspace_root)}
    return state


def invalidate_file_summary(state, path, workspace_root=None):
    """移除指定文件的短摘要。"""
    state = normalize_memory_state(state, workspace_root)
    state["file_summaries"].pop(canonicalize_path(path, workspace_root), None)
    return state


def invalidate_stale_file_summaries(state, workspace_root=None):
    """删除与磁盘内容不再一致的文件摘要。"""
    state = normalize_memory_state(state, workspace_root)
    invalidated = [path for path, summary in state["file_summaries"].items() if summary["freshness"] != file_freshness(path, workspace_root)]
    for path in invalidated:
        state["file_summaries"].pop(path, None)
    return state, invalidated


def summarize_read_result(result, limit=180):
    """从读文件结果提取下一轮足够使用的短摘要。"""
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return clip(" | ".join(lines[:3]) if lines else "(empty)", limit)


def render_memory_text(state, workspace_root=None):
    """渲染会话工作记忆仪表盘，不展开持久记忆正文。"""
    state = normalize_memory_state(state, workspace_root)
    lines = ["Memory:", f"- task: {state['working']['task_summary'] or '-'}", f"- recent_files: {', '.join(state['working']['recent_files']) or '-'}"]
    summaries = [
        f"- {path}: {summary['summary']}"
        for path in state["working"]["recent_files"][:FILE_SUMMARY_LIMIT]
        if (summary := state["file_summaries"].get(path)) and summary["freshness"] == file_freshness(path, workspace_root)
    ]
    lines.extend(["- file_summaries:", *[f"  {line}" for line in summaries]] if summaries else ["- file_summaries: -"])
    lines.append(f"- episodic_notes: {len(state['episodic_notes'])}")
    lines.append(f"- surfaced_file_memories: {len(state['surfaced_memory_paths'])}")
    return "\n".join(lines)


def is_effectively_empty(state, workspace_root=None):
    """判断 session 工作记忆是否尚未记录任何有效上下文。"""
    state = normalize_memory_state(state, workspace_root)
    return not (
        state["working"]["task_summary"]
        or state["working"]["recent_files"]
        or state["episodic_notes"]
        or state["file_summaries"]
    )


class LayeredMemory:
    """组合会话内工作记忆与项目文件记忆的运行时接口。"""

    def __init__(self, state=None, workspace_root=None):
        """绑定 session 状态和项目根目录。"""
        self.workspace_root = Path(workspace_root) if workspace_root is not None else None
        self.state = normalize_memory_state(state, self.workspace_root)
        self.memory_dir = get_memory_dir(self.workspace_root) if self.workspace_root is not None else None

    def to_dict(self):
        """返回可持久化的 session 工作记忆。"""
        self.state = normalize_memory_state(self.state, self.workspace_root)
        return self.state

    def canonical_path(self, path):
        """返回工作区内路径的稳定相对形式。"""
        return canonicalize_path(path, self.workspace_root)

    def set_task_summary(self, summary):
        """更新工作记忆中的任务摘要。"""
        self.state = set_task_summary(self.state, summary, self.workspace_root)
        return self

    def remember_file(self, path):
        """记录最近访问文件。"""
        self.state = remember_file(self.state, path, self.workspace_root)
        return self

    def append_note(self, text, tags=(), source="", created_at=None, kind="episodic"):
        """追加会话短笔记。"""
        self.state = append_note(self.state, text, tags, source, created_at, self.workspace_root, kind)
        return self

    def set_file_summary(self, path, summary):
        """写入文件摘要。"""
        self.state = set_file_summary(self.state, path, summary, self.workspace_root)
        return self

    def invalidate_file_summary(self, path):
        """使文件摘要失效。"""
        self.state = invalidate_file_summary(self.state, path, self.workspace_root)
        return self

    def invalidate_stale_file_summaries(self):
        """使磁盘已变化文件的摘要失效。"""
        self.state, invalidated = invalidate_stale_file_summaries(self.state, self.workspace_root)
        return invalidated

    def list_file_memories(self) -> list[MemoryHeader]:
        """列出当前项目的持久文件记忆。"""
        return list_memories(self.memory_dir) if self.memory_dir is not None else []

    def memory_prompt_section(self) -> str:
        """生成文件记忆系统的提示词说明。"""
        return build_memory_prompt_section(self.memory_dir) if self.memory_dir is not None else ""

    def select_relevant_memories(self, query: str, side_query: Callable[[str, str], str]) -> list[RelevantMemory]:
        """语义选择未展示文件记忆，并计入本会话预算。"""
        if self.memory_dir is None:
            return []
        selected = select_relevant_memories(
            query,
            side_query,
            set(self.state["surfaced_memory_paths"]),
            self.memory_dir,
            MAX_SURFACED_MEMORY_BYTES - self.state["surfaced_memory_bytes"],
        )
        self.state["surfaced_memory_paths"].extend(str(memory.path) for memory in selected)
        self.state["surfaced_memory_bytes"] += sum(len(memory.content.encode("utf-8")) for memory in selected)
        return selected

    def render_memory_text(self):
        """渲染工作记忆仪表盘。"""
        return render_memory_text(self.state, self.workspace_root)
