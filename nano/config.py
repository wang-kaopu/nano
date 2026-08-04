"""Project-local configuration helpers."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class RuntimeLimits:
    """集中保存 agent 回合、步骤和子任务预算配置。"""

    max_steps: int | None
    max_new_tokens: int
    max_final_tokens: int
    max_turns: int | None
    max_final_retries: int
    max_protocol_retries: int
    max_invalid_tool_calls: int
    max_auto_extensions: int
    auto_extension_steps: int
    auto_extension_max_steps: int
    duplicate_read_limit: int
    explorer_list_files_limit: int
    min_explorer_steps: int
    max_initial_explorer_steps: int
    initial_worker_steps: int
    max_initial_worker_steps: int
    initial_default_steps: int
    max_agent_depth: int
    provider_max_retries: int


def _integer_env(name: str, default: int, minimum: int = 1) -> int:
    """读取并校验一个整数环境变量，避免错误配置静默改变运行预算。"""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        comparison = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparison}")
    return value


def _optional_integer_env(name: str, minimum: int = 1) -> int | None:
    """读取可留空的整数环境变量，用于启用或关闭固定上限。"""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        comparison = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparison}")
    return value


def _integer_or_unlimited_env(name: str, default: int, minimum: int = 1) -> int | None:
    """读取可留空的整数上限；未设置时使用兼容默认值，留空时表示无限。"""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    if raw_value.strip() == "":
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        comparison = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparison}")
    return value


def runtime_limits_from_env() -> RuntimeLimits:
    """从已加载的环境变量构建 agent 的全部回合与步骤预算。"""
    limits = RuntimeLimits(
        max_steps=_integer_or_unlimited_env("NANO_MAX_STEPS", 12),
        max_new_tokens=_integer_env("NANO_MAX_NEW_TOKENS", 4096),
        max_final_tokens=_integer_env("NANO_MAX_FINAL_TOKENS", 2048),
        max_turns=_optional_integer_env("NANO_MAX_TURNS"),
        max_final_retries=_integer_env("NANO_MAX_FINAL_RETRIES", 1, minimum=0),
        max_protocol_retries=_integer_env("NANO_MAX_PROTOCOL_RETRIES", 3, minimum=0),
        max_invalid_tool_calls=_integer_env("NANO_MAX_INVALID_TOOL_CALLS", 3),
        max_auto_extensions=_integer_env("NANO_MAX_AUTO_EXTENSIONS", 1, minimum=0),
        auto_extension_steps=_integer_env("NANO_AUTO_EXTENSION_STEPS", 2),
        auto_extension_max_steps=_integer_env("NANO_AUTO_EXTENSION_MAX_STEPS", 10),
        duplicate_read_limit=_integer_env("NANO_DUPLICATE_READ_LIMIT", 2),
        explorer_list_files_limit=_integer_env("NANO_EXPLORER_LIST_FILES_LIMIT", 5),
        min_explorer_steps=_integer_env("NANO_MIN_EXPLORER_STEPS", 2),
        max_initial_explorer_steps=_integer_env("NANO_MAX_INITIAL_EXPLORER_STEPS", 8),
        initial_worker_steps=_integer_env("NANO_INITIAL_WORKER_STEPS", 6),
        max_initial_worker_steps=_integer_env("NANO_MAX_INITIAL_WORKER_STEPS", 10),
        initial_default_steps=_integer_env("NANO_INITIAL_DEFAULT_STEPS", 3),
        max_agent_depth=_integer_env("NANO_MAX_AGENT_DEPTH", 1),
        provider_max_retries=_integer_env("NANO_PROVIDER_MAX_RETRIES", 3, minimum=0),
    )
    if limits.min_explorer_steps > limits.max_initial_explorer_steps:
        raise ValueError("NANO_MIN_EXPLORER_STEPS must not exceed NANO_MAX_INITIAL_EXPLORER_STEPS")
    if limits.initial_worker_steps > limits.max_initial_worker_steps:
        raise ValueError("NANO_INITIAL_WORKER_STEPS must not exceed NANO_MAX_INITIAL_WORKER_STEPS")
    return limits


def deepseek_web_search_max_uses_from_env() -> int:
    """读取 DeepSeek 原生网页搜索的单次请求调用上限；设为 0 可关闭搜索。"""
    return _integer_env("NANO_DEEPSEEK_WEB_SEARCH_MAX_USES", 5, minimum=0)


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        raise ValueError(f"invalid .env line: {line}")
    name, value = line.split("=", 1)
    name = name.strip()
    if not ENV_KEY_PATTERN.match(name):
        raise ValueError(f"invalid .env variable name: {name}")
    return name, _strip_quotes(value)


def find_project_env(start: str | Path) -> Path | None:
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for path in (current, *current.parents):
        env_path = path / ".env"
        if env_path.exists():
            return env_path
    return None


def load_project_env(start: str | Path, override: bool = True) -> dict[str, str]:
    env_path = find_project_env(start)
    if env_path is None:
        return {}
    loaded = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        name, value = parsed
        loaded[name] = value
        if override or name not in os.environ:
            os.environ[name] = value
    return loaded


def provider_env(name: str, legacy_names: tuple[str, ...] = (), default: str = "") -> str:
    for env_name in (name, *legacy_names):
        value = os.environ.get(env_name)
        if value:
            return value
    return default
