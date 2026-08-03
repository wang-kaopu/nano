"""基于 Bash AST 的 shell 命令安全白名单。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import bashlex
from bashlex import ast
from bashlex.errors import ParsingError


class ShellCommandParseError(ValueError):
    """表示命令无法被 Bash AST 解析，因而不能安全执行。"""


@dataclass(frozen=True)
class ShellRisk:
    """描述一条尚未被证明安全、因此需要人工审批的 shell 操作。"""

    command: str
    reason: str


_SAFE_INSPECTION_COMMANDS = {
    "basename",
    "cat",
    "cut",
    "date",
    "dirname",
    "du",
    "echo",
    "false",
    "file",
    "grep",
    "head",
    "hostname",
    "id",
    "ls",
    "md5",
    "md5sum",
    "paste",
    "printf",
    "pwd",
    "realpath",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "tee",
    "test",
    "tr",
    "true",
    "uname",
    "uniq",
    "wc",
    "which",
    "whoami",
}
_SAFE_TEST_COMMANDS = {"eslint", "jest", "mypy", "pyright", "pytest", "ruff", "tsc", "vitest"}
_SAFE_BUILD_SUBCOMMANDS = {"build", "check", "lint", "test", "typecheck"}
_SAFE_GIT_SUBCOMMANDS = {"diff", "log", "ls-files", "ls-tree", "rev-parse", "show", "shortlog", "status"}
_SHELL_INTERPRETERS = {"bash", "dash", "sh", "zsh"}
_GIT_OPTIONS_WITH_VALUE = {"-C", "-c", "--config-env", "--git-dir", "--work-tree"}


def analyze_shell_command(command: str) -> tuple[ShellRisk, ...]:
    """解析命令；仅 AST 能证明安全的操作不会出现在风险列表中。"""
    try:
        trees = bashlex.parse(command)
    except ParsingError as exc:
        raise ShellCommandParseError(f"shell command cannot be parsed: {exc}") from exc

    visitor = _ShellRiskVisitor()
    for tree in trees:
        visitor.visit(tree)
    return tuple(visitor.risks)


def requires_shell_approval(command: str) -> bool:
    """判断 shell 命令是否包含未知或有副作用的操作。"""
    return bool(analyze_shell_command(command))


def _command_words(node: Any) -> list[str]:
    """提取 command 节点中的静态单词。"""
    return [part.word for part in node.parts if part.kind == "word"]


def _effective_words(words: list[str]) -> list[str]:
    """移除环境变量赋值与无副作用包装器，返回实际执行的命令及参数。"""
    index = 0
    while index < len(words):
        if "=" in words[index] and not words[index].startswith("="):
            index += 1
            continue
        if words[index] in {"command", "env", "nohup"}:
            index += 1
            while index < len(words) and words[index].startswith("-"):
                index += 1
            continue
        return words[index:]
    return []


def _command_name(words: list[str]) -> str:
    """从实际执行命令的第一个单词中提取可执行文件名。"""
    return PurePosixPath(words[0].lstrip("\\")).name if words else ""


def _git_subcommand(words: list[str]) -> str:
    """跳过 Git 全局选项后提取子命令。"""
    index = 1
    while index < len(words):
        word = words[index]
        if word in _GIT_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if word.startswith("--git-dir=") or word.startswith("--work-tree=") or word.startswith("--config-env="):
            index += 1
            continue
        if word.startswith("-"):
            index += 1
            continue
        return word
    return ""


def _is_safe_inspection_command(command: str, words: list[str]) -> bool:
    """判断本地检查命令是否没有已知的写入或执行语义。"""
    if command not in _SAFE_INSPECTION_COMMANDS:
        return False
    if command == "sed":
        return not any(word == "-i" or word.startswith("-i") or word == "--in-place" for word in words[1:])
    if command == "tee":
        return len(words) == 1
    return True


def _is_safe_git_command(words: list[str]) -> bool:
    """判断 Git 调用是否属于严格只读的子命令集合。"""
    return _git_subcommand(words) in _SAFE_GIT_SUBCOMMANDS


def _is_safe_build_or_test_command(command: str, words: list[str]) -> bool:
    """判断项目中常用的检查、构建或测试命令是否可自动执行。"""
    if command in _SAFE_TEST_COMMANDS:
        return True
    if command in {"cargo", "go", "make"}:
        return len(words) > 1 and words[1] in _SAFE_BUILD_SUBCOMMANDS
    if command in {"npm", "pnpm", "yarn"}:
        return len(words) > 1 and (words[1] == "test" or (words[1] == "run" and len(words) > 2 and words[2] in _SAFE_BUILD_SUBCOMMANDS))
    if command == "uv" and len(words) > 2 and words[1] == "run":
        return _is_safe_build_or_test_command(_command_name(words[2:]), words[2:])
    if command == "python":
        return len(words) == 2 and words[1] in {"-V", "--version"}
    return False


class _ShellRiskVisitor(ast.nodevisitor):
    """遍历 Bash AST，将非白名单操作记录为需要审批的风险。"""

    def __init__(self) -> None:
        """初始化去重后的风险收集器。"""
        self.risks: list[ShellRisk] = []
        self._seen: set[tuple[str, str]] = set()

    def visitcommand(self, node: Any, parts: list[Any]) -> None:
        """仅允许显式声明为安全的命令和子命令。"""
        words = _effective_words(_command_words(node))
        command = _command_name(words)
        if not command:
            self._add("shell command", "cannot determine the executable")
        elif command in _SHELL_INTERPRETERS or command in {"eval", "exec", "source", ".", "xargs"}:
            self._add(command, "executes dynamically supplied shell code")
        elif command == "git":
            if not _is_safe_git_command(words):
                subcommand = _git_subcommand(words) or "command"
                self._add(f"git {subcommand}", "is not a read-only Git command")
        elif _is_safe_inspection_command(command, words) or _is_safe_build_or_test_command(command, words):
            return
        else:
            self._add(command, "is not on the shell safe allowlist")

    def visitredirect(self, node: Any, input_value: Any, redirect_type: Any, output: Any, heredoc: Any) -> None:
        """将输出重定向视为文件写入，避免安全命令经重定向绕过审批。"""
        if ">" in str(redirect_type):
            self._add("shell redirection", "writes command output to a file")

    def visitfunction(self, node: Any, name: Any, body: Any, parts: list[Any]) -> None:
        """函数定义可延后执行任意 shell 代码，必须获得审批。"""
        self._add("shell function", "defines executable shell code")

    def _add(self, command: str, reason: str) -> None:
        """保存一条唯一的风险记录。"""
        key = (command, reason)
        if key not in self._seen:
            self._seen.add(key)
            self.risks.append(ShellRisk(command=command, reason=reason))
