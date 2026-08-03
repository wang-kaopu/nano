"""项目级 Skill 的发现、解析和提示词展开。"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from nano.utils.frontmatter import parse_frontmatter

SKILL_DIRECTORY = ".nano/skills"


@dataclass(frozen=True)
class SkillDefinition:
    """描述一个可由用户或模型调用的项目级 Skill。"""

    name: str
    description: str
    when_to_use: str | None
    allowed_tools: list[str] | None
    user_invocable: bool
    context: str
    prompt_template: str
    source: str
    skill_dir: str


def _parse_skill_file(file_path: Path, source: str, skill_dir: str) -> SkillDefinition | None:
    """解析单个 SKILL.md，格式错误时忽略该 Skill。"""
    try:
        raw = file_path.read_text(encoding="utf-8")
        result = parse_frontmatter(raw)
        meta = result.meta

        name = meta.get("name") or file_path.parent.name or "unknown"
        user_invocable = meta.get("user-invocable", "true") != "false"
        context = "fork" if meta.get("context") == "fork" else "inline"

        allowed_tools: list[str] | None = None
        if "allowed-tools" in meta:
            raw_tools = meta["allowed-tools"]
            if raw_tools.startswith("["):
                try:
                    parsed_tools = json.loads(raw_tools)
                    if isinstance(parsed_tools, list):
                        allowed_tools = [str(tool).strip() for tool in parsed_tools]
                    else:
                        allowed_tools = [str(parsed_tools).strip()]
                except Exception:
                    allowed_tools = [tool.strip() for tool in raw_tools.strip("[]").split(",")]
            else:
                allowed_tools = [tool.strip() for tool in raw_tools.split(",")]
            allowed_tools = [tool for tool in allowed_tools if tool]

        return SkillDefinition(
            name=name,
            description=meta.get("description", ""),
            when_to_use=meta.get("when_to_use") or meta.get("when-to-use"),
            allowed_tools=allowed_tools,
            user_invocable=user_invocable,
            context=context,
            prompt_template=result.body,
            source=source,
            skill_dir=skill_dir,
        )
    except Exception:
        return None


def discover_skills(workspace_root: str | Path | None = None) -> list[SkillDefinition]:
    """发现当前项目 `.nano/skills` 中的全部有效 Skill。"""
    root = Path(workspace_root or Path.cwd()).resolve()
    skills_root = root / SKILL_DIRECTORY
    if not skills_root.is_dir():
        return []

    skills = []
    for file_path in sorted(skills_root.rglob("SKILL.md")):
        skill = _parse_skill_file(file_path, source="project", skill_dir=str(file_path.parent))
        if skill is not None:
            skills.append(skill)
    return skills


def get_skill_by_name(name: str, workspace_root: str | Path | None = None) -> SkillDefinition | None:
    """按名称返回项目 Skill，不存在时返回 None。"""
    return next((skill for skill in discover_skills(workspace_root) if skill.name == name), None)


def resolve_skill_prompt(skill: SkillDefinition, args: str) -> str:
    """将 Skill 参数和目录路径替换进提示词模板。"""
    prompt = skill.prompt_template
    prompt = re.sub(r"\$ARGUMENTS|\$\{ARGUMENTS\}", args, prompt)
    return prompt.replace("${CLAUDE_SKILL_DIR}", skill.skill_dir)


def execute_skill(name: str, args: str = "", workspace_root: str | Path | None = None) -> dict[str, str] | None:
    """查找并展开指定 Skill，供模型工具调用使用。"""
    skill = get_skill_by_name(name, workspace_root)
    if skill is None:
        return None
    return {"prompt": resolve_skill_prompt(skill, args)}


def build_skill_descriptions(workspace_root: str | Path | None = None) -> str:
    """构建供模型判断是否主动调用 Skill 的 system prompt 片段。"""
    skills = discover_skills(workspace_root)
    if not skills:
        return ""

    lines = ["# Available Skills", ""]
    invocable = [skill for skill in skills if skill.user_invocable]
    auto_only = [skill for skill in skills if not skill.user_invocable]

    if invocable:
        lines.append("User-invocable skills (user types /<name> to invoke):")
        for skill in invocable:
            lines.append(f"- **/{skill.name}**: {skill.description}")
            if skill.when_to_use:
                lines.append(f"  When to use: {skill.when_to_use}")
        lines.append("")

    if auto_only:
        lines.append("Auto-invocable skills (use the skill tool when appropriate):")
        for skill in auto_only:
            lines.append(f"- **{skill.name}**: {skill.description}")
            if skill.when_to_use:
                lines.append(f"  When to use: {skill.when_to_use}")
        lines.append("")

    lines.append("To invoke a skill programmatically, use the `skill` tool.")
    return "\n".join(lines)
