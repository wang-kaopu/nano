from pathlib import Path

IGNORED_NAMES = {".git", ".mini-nano", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()

    @classmethod
    def build(cls, cwd="."):
        return cls(Path(cwd).resolve())

    def path(self, value):
        candidate = (self.root / str(value)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes workspace")
        return candidate

    def relative(self, path):
        return str(Path(path).resolve().relative_to(self.root))

    def snapshot_text(self):
        readme = self.root / "README.md"
        readme_text = ""
        if readme.exists():
            readme_text = readme.read_text(encoding="utf-8", errors="replace")[:1200]
        entries = []
        for item in sorted(self.root.iterdir(), key=lambda path: (path.is_file(), path.name.lower())):
            if item.name in IGNORED_NAMES:
                continue
            entries.append(("[D] " if item.is_dir() else "[F] ") + item.name)
        listing = "\n".join(entries[:80]) or "(empty)"
        return f"Workspace root: {self.root}\nFiles:\n{listing}\nREADME excerpt:\n{readme_text}"


def clip(text, limit=5000):
    """保留过长文本的首尾内容，避免丢失工具输出结论。"""
    text = str(text)
    if len(text) < limit:
        return text
    preserved = (limit - 60) // 2
    return text[:preserved] + "(...)" + text[-preserved:]
