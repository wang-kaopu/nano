"""Session JSON persistence."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nano.storage.schemas import SessionModel
from nano.tools.security import redact_artifact


class SessionStore:
    """负责会话文档的 Pydantic 校验和 JSON 持久化。"""

    def __init__(self, root: str | Path, secret_env_names: Iterable[str] | None = None) -> None:
        """初始化会话存储目录，并配置需要额外脱敏的环境变量名。"""
        self.root = Path(root)
        self.secret_env_names = {str(name).upper() for name in (secret_env_names or ())}
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        """返回指定会话的 JSON 路径。"""
        return self.root / f"{session_id}.json"

    def save(self, session: SessionModel | dict[str, Any]) -> Path:
        """校验、脱敏并保存会话，返回写入路径。"""
        model = session if isinstance(session, SessionModel) else SessionModel.model_validate(session)
        model = SessionModel.model_validate(redact_artifact(model.model_dump(mode="python"), secret_env_names=self.secret_env_names))
        path = self.path(model.id)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, session_id: str) -> dict[str, Any]:
        """读取并校验会话 JSON，返回运行时使用的普通字典。"""
        return SessionModel.model_validate_json(self.path(session_id).read_text(encoding="utf-8")).model_dump(mode="python")

    def latest(self) -> str | None:
        """返回最近写入的会话 ID；没有会话时返回 None。"""
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None
