"""Security and redaction helpers for runtime artifacts."""

import os
from collections.abc import Iterable
from pathlib import Path

from .types import Environment, JsonValue

SENSITIVE_ENV_NAME_MARKERS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
REDACTED_VALUE = "<redacted>"


def _normalized_secret_names(secret_env_names: Iterable[str] | None) -> set[str]:
    return {str(name).upper() for name in (secret_env_names or ())}


def looks_sensitive_env_name(name: str) -> bool:
    upper = str(name).upper()
    return any(upper == marker or upper.endswith(marker) or upper.endswith(f"_{marker}") for marker in SENSITIVE_ENV_NAME_MARKERS)


def is_secret_env_name(name: str, secret_env_names: Iterable[str] | None = None) -> bool:
    upper = str(name).upper()
    return upper in _normalized_secret_names(secret_env_names) or looks_sensitive_env_name(upper)


def configured_secret_env_items(
    env: Environment | None = None, secret_env_names: Iterable[str] | None = None
) -> list[tuple[str, str]]:
    env = os.environ if env is None else env
    configured_names = _normalized_secret_names(secret_env_names)
    items = [
        (name, value)
        for name, value in env.items()
        if str(name).upper() in configured_names and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def detected_secret_env_items(
    env: Environment | None = None, secret_env_names: Iterable[str] | None = None
) -> list[tuple[str, str]]:
    env = os.environ if env is None else env
    items = [
        (name, value)
        for name, value in env.items()
        if is_secret_env_name(name, secret_env_names=secret_env_names) and value
    ]
    items.sort(key=lambda item: item[0])
    return items


def secret_env_summary(
    env: Environment | None = None, secret_env_names: Iterable[str] | None = None
) -> dict[str, int | list[str]]:
    names = [name for name, _ in configured_secret_env_items(env=env, secret_env_names=secret_env_names)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def detected_secret_env_summary(
    env: Environment | None = None, secret_env_names: Iterable[str] | None = None
) -> dict[str, int | list[str]]:
    names = [name for name, _ in detected_secret_env_items(env=env, secret_env_names=secret_env_names)]
    return {
        "secret_env_count": len(names),
        "secret_env_names": names,
    }


def redact_text(text: object, env: Environment | None = None, secret_env_names: Iterable[str] | None = None) -> str:
    text = str(text)
    for _, value in sorted(
        detected_secret_env_items(env=env, secret_env_names=secret_env_names),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        text = text.replace(value, REDACTED_VALUE)
    return text


def redact_artifact(
    value: JsonValue,
    key: str | None = None,
    env: Environment | None = None,
    secret_env_names: Iterable[str] | None = None,
) -> JsonValue:
    if key and is_secret_env_name(key, secret_env_names=secret_env_names):
        return REDACTED_VALUE
    if isinstance(value, dict):
        return {
            str(item_key): redact_artifact(item_value, key=item_key, env=env, secret_env_names=secret_env_names)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]
    if isinstance(value, tuple):
        return [redact_artifact(item, key=key, env=env, secret_env_names=secret_env_names) for item in value]
    if isinstance(value, str):
        return redact_text(value, env=env, secret_env_names=secret_env_names)
    return value


def shell_env(
    env: Environment | None = None,
    allowlist: Iterable[str] = (),
    root: str | Path = ".",
) -> dict[str, str]:
    env = os.environ if env is None else env
    filtered = {
        name: env[name]
        for name in allowlist
        if name in env
    }
    filtered["PWD"] = str(root)
    if "PATH" not in filtered and env.get("PATH"):
        filtered["PATH"] = env["PATH"]
    return filtered
