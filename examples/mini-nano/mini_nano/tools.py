import shutil
import subprocess
from dataclasses import dataclass

from .workspace import IGNORED_NAMES


@dataclass(frozen=True)
class ToolSpec:
    description: str
    risky: bool
    schema: str


TOOL_SPECS = {
    "list_files": ToolSpec("List files in the workspace.", False, '{"path": "str=."}'),
    "read_file": ToolSpec("Read a UTF-8 file by line range.", False, '{"path": "str", "start": "int=1", "end": "int=120"}'),
    "search": ToolSpec("Search text in the workspace.", False, '{"pattern": "str", "path": "str=."}'),
    "write_file": ToolSpec("Write a new text file or overwrite a previously read file.", True, '{"path": "str", "content": "str"}'),
    "patch_file": ToolSpec("Replace one exact text block in a previously read file.", True, '{"path": "str", "old_text": "str", "new_text": "str"}'),
}


def tool_signature():
    lines = []
    for name, spec in TOOL_SPECS.items():
        risk = "risky" if spec.risky else "read-only"
        lines.append(f"- {name} ({risk}) {spec.schema}: {spec.description}")
    return "\n".join(lines)


def validate_tool(workspace, name, args, read_file_state=None):
    if name not in TOOL_SPECS:
        raise ValueError(f"unknown tool: {name}")
    args = args or {}
    if name == "list_files":
        path = workspace.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return
    if name == "read_file":
        path = workspace.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 120))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return
    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        workspace.path(args.get("path", "."))
        return
    if name == "write_file":
        path = workspace.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        if read_file_state is not None and path.exists():
            recorded_mtime = read_file_state.get(str(path))
            if recorded_mtime is None:
                raise ValueError("you must read this file before modifying it; use read_file first")
            if path.stat().st_mtime_ns // 1_000_000 != recorded_mtime:
                raise ValueError("warning: file was modified externally; use read_file again")
        return
    if name == "patch_file":
        path = workspace.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        if read_file_state is not None:
            recorded_mtime = read_file_state.get(str(path))
            if recorded_mtime is None:
                raise ValueError("you must read this file before modifying it; use read_file first")
            if path.stat().st_mtime_ns // 1_000_000 != recorded_mtime:
                raise ValueError("warning: file was modified externally; use read_file again")
        count = path.read_text(encoding="utf-8", errors="replace").count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")


def run_tool(workspace, name, args, read_file_state=None):
    validate_tool(workspace, name, args, read_file_state)
    if name == "list_files":
        return _list_files(workspace, args)
    if name == "read_file":
        return _read_file(workspace, args, read_file_state)
    if name == "search":
        return _search(workspace, args)
    if name == "write_file":
        return _write_file(workspace, args, read_file_state)
    if name == "patch_file":
        return _patch_file(workspace, args, read_file_state)
    raise ValueError(f"unknown tool: {name}")


def _list_files(workspace, args):
    path = workspace.path(args.get("path", "."))
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_NAMES
    ]
    lines = []
    for entry in entries[:120]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {workspace.relative(entry)}")
    return "\n".join(lines) or "(empty)"


def _read_file(workspace, args, read_file_state=None):
    path = workspace.path(args["path"])
    start = int(args.get("start", 1))
    end = int(args.get("end", 120))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    if read_file_state is not None:
        try:
            read_file_state[str(path)] = path.stat().st_mtime_ns // 1_000_000
        except OSError:
            pass
    return f"# {workspace.relative(path)}\n{body}"


def _search(workspace, args):
    pattern = str(args["pattern"])
    path = workspace.path(args.get("path", "."))
    if shutil.which("rg"):
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "80", pattern, str(path)],
            cwd=workspace.root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "(no matches)"
    matches = []
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    for file_path in files:
        if any(part in IGNORED_NAMES for part in file_path.relative_to(workspace.root).parts):
            continue
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{workspace.relative(file_path)}:{number}:{line}")
                if len(matches) >= 80:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def _write_file(workspace, args, read_file_state=None):
    path = workspace.path(args["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(args["content"]), encoding="utf-8")
    if read_file_state is not None:
        try:
            read_file_state[str(path)] = path.stat().st_mtime_ns // 1_000_000
        except OSError:
            pass
    return f"wrote {workspace.relative(path)}"


def _patch_file(workspace, args, read_file_state=None):
    path = workspace.path(args["path"])
    text = path.read_text(encoding="utf-8")
    old_text = str(args["old_text"])
    new_text = str(args["new_text"])
    path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    if read_file_state is not None:
        try:
            read_file_state[str(path)] = path.stat().st_mtime_ns // 1_000_000
        except OSError:
            pass
    return f"patched {workspace.relative(path)}"
