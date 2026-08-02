"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import shutil
import subprocess
import textwrap
from functools import partial
from typing import Annotated, Any, Type

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from .workspace import IGNORED_PATH_NAMES
from .tool_context import ToolContext
from .types import ToolArguments as ToolArgumentsPayload


class ToolArguments(BaseModel):
    """所有模型生成工具参数的共同校验配置。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class ListFilesArguments(ToolArguments):
    """列出目录内容所需的参数。"""

    path: str = "."


class ReadFileArguments(ToolArguments):
    """读取文件所需的参数。"""

    path: NonEmptyText
    start: int = Field(default=1, ge=1)
    end: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> "ReadFileArguments":
        """确保结束行不早于起始行。"""
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self


class SearchArguments(ToolArguments):
    """搜索文本所需的参数。"""

    pattern: NonEmptyText
    path: str = "."


class RunShellArguments(ToolArguments):
    """执行 shell 命令所需的参数。"""

    command: NonEmptyText
    timeout: int = Field(default=20, ge=1, le=120)


class WriteFileArguments(ToolArguments):
    """写入文件所需的参数。"""

    path: NonEmptyText
    content: str


class PatchFileArguments(ToolArguments):
    """精确替换文件内容所需的参数。"""

    path: NonEmptyText
    old_text: NonEmptyText
    new_text: str


class DelegateArguments(ToolArguments):
    """委派只读子任务所需的参数。"""

    task: NonEmptyText
    max_steps: int = Field(default=3, ge=1)


TOOL_ARGUMENT_MODELS: dict[str, Type[ToolArguments]] = {
    "list_files": ListFilesArguments,
    "read_file": ReadFileArguments,
    "search": SearchArguments,
    "run_shell": RunShellArguments,
    "write_file": WriteFileArguments,
    "patch_file": PatchFileArguments,
    "delegate": DelegateArguments,
}


def tool_arguments_model(name: str) -> Type[ToolArguments]:
    """返回工具名称对应的 Pydantic 参数模型。"""
    try:
        return TOOL_ARGUMENT_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"unknown tool: {name}") from exc


def tool_json_schema(name: str) -> dict[str, Any]:
    """返回供模型调用协议使用的 JSON Schema。"""
    return tool_arguments_model(name).model_json_schema()


def validate_tool_arguments(name: str, args: ToolArgumentsPayload | None) -> dict[str, Any]:
    """校验工具参数并返回规范化后的普通字典。"""
    try:
        model = tool_arguments_model(name).model_validate(args or {})
    except ValidationError:
        raise
    return model.model_dump(mode="python")


def _human_schema(model: Type[ToolArguments]) -> dict[str, str]:
    """把 Pydantic 字段转换为 prompt 中使用的简短字段描述。"""
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    result = {}
    for name, value in properties.items():
        field_type = value.get("type", "value")
        default = value.get("default")
        suffix = "" if name in required else f"={default!r}"
        result[name] = f"{field_type}{suffix}"
    return result

BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
}

for _tool_name, _tool_spec in BASE_TOOL_SPECS.items():
    _tool_spec["args_model"] = TOOL_ARGUMENT_MODELS[_tool_name]
    _tool_spec["json_schema"] = tool_json_schema(_tool_name)
    _tool_spec["schema"] = _human_schema(_tool_spec["args_model"])

DELEGATE_TOOL_SPEC = {
    "schema": {"task": "str", "max_steps": "int=3"},
    "risky": False,
    "description": "Ask a bounded read-only child agent to investigate.",
}
DELEGATE_TOOL_SPEC["args_model"] = TOOL_ARGUMENT_MODELS["delegate"]
DELEGATE_TOOL_SPEC["json_schema"] = tool_json_schema("delegate")
DELEGATE_TOOL_SPEC["schema"] = _human_schema(DELEGATE_TOOL_SPEC["args_model"])


def legal_tool_names() -> set[str]:
    return set(BASE_TOOL_SPECS) | {"delegate"}

TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    "delegate": '<tool>{"name":"delegate","args":{"task":"inspect README.md","max_steps":3}}</tool>',
}


def build_tool_registry(context: ToolContext) -> dict[str, dict[str, Any]]:
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: {**spec, "run": partial(_TOOL_RUNNERS[name], context)}
        for name, spec in BASE_TOOL_SPECS.items()
    }
    # 子 agent 是刻意做成受限能力的：一旦深度耗尽，
    # 就连 delegate 这个工具都不再暴露给模型。
    if context.depth < context.max_depth:
        tools["delegate"] = {**DELEGATE_TOOL_SPEC, "run": partial(tool_delegate, context)}
    return tools


def tool_example(name: str) -> str:
    return TOOL_EXAMPLES.get(name, "")


def validate_tool(context: ToolContext, name: str, args: ToolArgumentsPayload | None) -> dict[str, Any]:
    args = validate_tool_arguments(name, args)

    if name == "list_files":
        path = context.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")
        return args

    if name == "read_file":
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        start = int(args.get("start", 1))
        end = int(args.get("end", 200))
        if start < 1 or end < start:
            raise ValueError("invalid line range")
        return args

    if name == "search":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        context.path(args.get("path", "."))
        return args

    if name == "run_shell":
        command = str(args.get("command", "")).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = int(args.get("timeout", 20))
        if timeout < 1 or timeout > 120:
            raise ValueError("timeout must be in [1, 120]")
        return args

    if name == "write_file":
        path = context.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")
        if "content" not in args:
            raise ValueError("missing content")
        return args

    if name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = context.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        old_text = str(args.get("old_text", ""))
        if not old_text:
            raise ValueError("old_text must not be empty")
        if "new_text" not in args:
            raise ValueError("missing new_text")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")
        return args

    if name == "delegate":
        task = str(args.get("task", "")).strip()
        if not task:
            raise ValueError("task must not be empty")
        if context.depth >= context.max_depth:
            raise ValueError("delegate depth exceeded")
        return args

    raise ValueError(f"unknown tool: {name}")


def tool_list_files(context: ToolContext, args: ToolArgumentsPayload) -> str:
    path = context.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = [
        item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        if item.name not in IGNORED_PATH_NAMES
    ]
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(context.root)}")
    return "\n".join(lines) or "(empty)"


def tool_read_file(context: ToolContext, args: ToolArgumentsPayload) -> str:
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(f"{number:>4}: {line}" for number, line in enumerate(lines[start - 1:end], start=start))
    return f"# {path.relative_to(context.root)}\n{body}"


def tool_search(context: ToolContext, args: ToolArgumentsPayload) -> str:
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = context.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=context.root,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = [path] if path.is_file() else [
        item for item in path.rglob("*")
        if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(context.root).parts)
    ]
    for file_path in files:
        for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(context.root)}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_run_shell(context: ToolContext, args: ToolArgumentsPayload) -> str:
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    result = subprocess.run(
        command,
        cwd=context.root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
        # 目的是减少敏感信息被意外带进命令执行环境的风险。
        env=context.shell_env(),
    )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


def tool_write_file(context: ToolContext, args: ToolArgumentsPayload) -> str:
    path = context.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(context.root)} ({len(content)} chars)"


def tool_patch_file(context: ToolContext, args: ToolArgumentsPayload) -> str:
    path = context.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(context.root)}"


def tool_delegate(context: ToolContext, args: ToolArgumentsPayload) -> str:
    if context.depth >= context.max_depth:
        raise ValueError("delegate depth exceeded")
    task = str(args.get("task", "")).strip()
    if not task:
        raise ValueError("task must not be empty")
    return context.spawn_delegate(args)


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
}
