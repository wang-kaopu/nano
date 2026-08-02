"""Session JSON persistence."""

from pathlib import Path

from .schemas import SessionModel


class SessionStore:
    """负责会话文档的 Pydantic 校验和 JSON 持久化。"""

    def __init__(self, root):
        """初始化会话存储目录。"""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id):
        """返回指定会话的 JSON 路径。"""
        return self.root / f"{session_id}.json"

    def save(self, session):
        """校验并保存会话，返回写入路径。"""
        model = session if isinstance(session, SessionModel) else SessionModel.model_validate(session)
        path = self.path(model.id)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load(self, session_id):
        """读取并校验会话 JSON，返回运行时使用的普通字典。"""
        return SessionModel.model_validate_json(self.path(session_id).read_text(encoding="utf-8")).model_dump(mode="python")

    def latest(self):
        """返回最近写入的会话 ID；没有会话时返回 None。"""
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        return files[-1].stem if files else None
