from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano.cli import _resolve_user_skill_command
from nano.skills import build_skill_descriptions, discover_skills, execute_skill, get_skill_by_name, resolve_skill_prompt


def _write_skill(root, name, content):
    """在临时项目中写入一个测试 Skill。"""
    path = root / ".nano" / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    return path


def _build_agent(root):
    """构建可直接执行 Skill 工具的测试 agent。"""
    (root / "README.md").write_text("demo\n", encoding="utf-8")
    return AgentRuntime(
        model_client=FakeModelClient(["<final>Done.</final>"]),
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".nano" / "sessions"),
        approval_policy="auto",
    )


def test_skill_discovery_parses_frontmatter_and_expands_template(tmp_path):
    skill_path = _write_skill(
        tmp_path,
        "commit",
        """---
name: commit
description: Create a git commit with a descriptive message
when_to_use: When the user asks to commit changes or says \"commit\"
allowed-tools: [\"run_shell\", \"read_file\"]
user-invocable: true
context: fork
---
Request: $ARGUMENTS / ${ARGUMENTS}
Directory: ${CLAUDE_SKILL_DIR}
""",
    )

    skills = discover_skills(tmp_path)

    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "commit"
    assert skill.when_to_use == 'When the user asks to commit changes or says "commit"'
    assert skill.allowed_tools == ["run_shell", "read_file"]
    assert skill.user_invocable is True
    assert skill.context == "fork"
    assert resolve_skill_prompt(skill, "stage all") == f"Request: stage all / stage all\nDirectory: {skill_path.parent}"
    assert execute_skill("commit", "stage all", tmp_path) == {"prompt": resolve_skill_prompt(skill, "stage all")}


def test_skill_descriptions_separate_manual_and_model_only_skills(tmp_path):
    _write_skill(
        tmp_path,
        "commit",
        """---
name: commit
description: Create a commit
when-to-use: After code changes
---
Commit the changes.
""",
    )
    _write_skill(
        tmp_path,
        "audit",
        """---
name: audit
description: Inspect security boundaries
user-invocable: false
---
Inspect the code.
""",
    )

    descriptions = build_skill_descriptions(tmp_path)

    assert "User-invocable skills" in descriptions
    assert "**/commit**: Create a commit" in descriptions
    assert "When to use: After code changes" in descriptions
    assert "Auto-invocable skills" in descriptions
    assert "**audit**: Inspect security boundaries" in descriptions
    assert "use the `skill` tool" in descriptions


def test_skill_tool_returns_expanded_instructions_and_prefix_refreshes(tmp_path):
    runtime = _build_agent(tmp_path)
    _write_skill(
        tmp_path,
        "commit",
        """---
name: commit
description: Create a commit
---
Commit request: $ARGUMENTS
""",
    )

    assert "# Available Skills" in runtime.prompt("Commit the changes")
    assert runtime.run_tool("skill", {"skill_name": "commit", "args": "stage all"}) == (
        '[Skill "commit" activated]\n\nCommit request: stage all'
    )
    assert runtime.run_tool("skill", {"skill_name": "missing"}) == "Unknown skill: missing"


def test_user_skill_command_only_resolves_user_invocable_skills(tmp_path):
    runtime = _build_agent(tmp_path)
    _write_skill(
        tmp_path,
        "commit",
        """---
name: commit
description: Create a commit
---
Commit request: $ARGUMENTS
""",
    )
    _write_skill(
        tmp_path,
        "hidden",
        """---
name: hidden
description: Hidden skill
user-invocable: false
---
Hidden request.
""",
    )

    assert _resolve_user_skill_command(runtime, "/commit stage all") == ("commit", "Commit request: stage all")
    assert _resolve_user_skill_command(runtime, "/hidden") is None
    assert get_skill_by_name("missing", tmp_path) is None
