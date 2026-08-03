"""跨模块复用的轻量文本与时间工具。"""

from datetime import datetime, timezone


def now() -> str:
    """返回统一使用的 UTC ISO 8601 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def clip(text: object, limit: int = 5000) -> str:
    """保留超长文本的首尾内容，避免丢失工具输出结论。"""
    text = str(text)
    if len(text) < limit:
        return text
    preserved = (limit - 60) // 2
    return text[:preserved] + "(...)" + text[-preserved:]


def middle(text: object, limit: int) -> str:
    """截断文本中段，使终端展示保留首尾信息。"""
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]
