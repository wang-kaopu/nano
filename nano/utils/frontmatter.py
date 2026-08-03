"""记忆文件 YAML frontmatter 的最小解析与格式化实现。"""

from dataclasses import dataclass, field


@dataclass
class FrontmatterResult:
    """保存 frontmatter 元数据与正文。"""

    meta: dict[str, str] = field(default_factory=dict)
    body: str = ""


def parse_frontmatter(content: str) -> FrontmatterResult:
    """解析 Markdown 开头的简单 YAML frontmatter。"""
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        return FrontmatterResult(body=content)

    end_index = -1
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index == -1:
        return FrontmatterResult(body=content)

    meta: dict[str, str] = {}
    for line in lines[1:end_index]:
        colon_index = line.find(":")
        if colon_index == -1:
            continue
        key = line[:colon_index].strip()
        value = line[colon_index + 1 :].strip()
        if key:
            meta[key] = value

    return FrontmatterResult(meta=meta, body="\n".join(lines[end_index + 1 :]).strip())


def format_frontmatter(meta: dict[str, str], content: str) -> str:
    """将单行元数据和正文格式化为记忆 Markdown 文件。"""
    lines = ["---"]
    for key, value in meta.items():
        lines.append(f"{key}: {str(value).replace(chr(10), ' ').strip()}")
    lines.extend(("---", "", str(content).strip(), ""))
    return "\n".join(lines)
