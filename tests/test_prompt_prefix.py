from nano.runtime.prompt_prefix import build_dynamic_system_context, build_prompt_prefix, tool_signature
from nano.tools.tools import build_tool_registry
from nano.workspace.context import MAX_INCLUDE_DEPTH, WorkspaceContext


class _Agent:
    depth = 0
    max_depth = 1

    def __init__(self, root):
        self.root = root


def test_tool_signature_is_stable_across_registry_insertion_order(tmp_path):
    tools = build_tool_registry(_Agent(tmp_path))
    reordered = {name: tools[name] for name in reversed(tuple(tools))}

    assert tool_signature(tools) == tool_signature(reordered)


def test_build_prompt_prefix_renders_tools_and_workspace_metadata(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    tools = build_tool_registry(_Agent(tmp_path))

    prefix = build_prompt_prefix(workspace=workspace, tools=tools, built_at="2026-06-02T00:00:00+08:00")

    assert "# 1. Identity" in prefix.text
    assert "interactive agent" in prefix.text
    assert "# 2. System" in prefix.text
    assert "Git context:" in prefix.text
    assert "# 3. Doing Tasks" in prefix.text
    assert "Do not propose changes to code you haven't read. Read files first." in prefix.text
    assert "Avoid over-engineering. Only make changes that were requested." in prefix.text
    assert "Do not expand scope" in prefix.text
    assert "Do not add defensive programming" in prefix.text
    assert "Three similar lines of code is better than a premature abstraction." in prefix.text
    assert "# 4. Actions" in prefix.text
    assert "Reversibility: determine whether the action can be safely undone." in prefix.text
    assert "# 5. Using Tools" in prefix.text
    assert "Use read_file / edit_file / list_files / grep_search instead of shell cat," in prefix.text
    assert "If several tool calls are independent, make them in parallel." in prefix.text
    assert "# 6. Tone & Style" in prefix.text
    assert "Reference code as file_path:line_number." in prefix.text
    assert "# 7. Output Efficiency" in prefix.text
    assert "## Available tools" not in prefix.text
    assert prefix.hash
    assert prefix.workspace_fingerprint == workspace.fingerprint()
    assert prefix.tool_signature == tool_signature(tools)
    assert prefix.built_at == "2026-06-02T00:00:00+08:00"


def test_build_prompt_prefix_includes_memory_system_when_provided(tmp_path):
    workspace = WorkspaceContext.build(tmp_path)
    tools = build_tool_registry(_Agent(tmp_path))

    prefix = build_prompt_prefix(workspace=workspace, tools=tools, memory_prompt_section="# Memory System\n/index/MEMORY.md")

    assert "# Memory System" in prefix.dynamic_system
    assert "/index/MEMORY.md" in prefix.text


def test_build_prompt_prefix_includes_skill_descriptions_when_provided(tmp_path):
    workspace = WorkspaceContext.build(tmp_path)
    tools = build_tool_registry(_Agent(tmp_path))

    prefix = build_prompt_prefix(workspace=workspace, tools=tools, skill_descriptions="# Available Skills\n- **/commit**: Create a commit")

    assert "# Available Skills" in prefix.dynamic_system
    assert "**/commit**: Create a commit" in prefix.text


def test_build_dynamic_system_context_expands_local_includes_up_to_five_layers(tmp_path):
    instruction_paths = [tmp_path / "AGENTS.md"]
    for depth in range(1, MAX_INCLUDE_DEPTH + 2):
        instruction_paths.append(tmp_path / f"instructions-{depth}.md")
    for depth, path in enumerate(instruction_paths):
        content = f"instruction depth {depth}"
        if depth < len(instruction_paths) - 1:
            content += f"\n@include {instruction_paths[depth + 1].name}"
        path.write_text(content, encoding="utf-8")

    workspace = WorkspaceContext.build(tmp_path)
    dynamic_context = build_dynamic_system_context(workspace)

    assert not dynamic_context.startswith(" ")
    assert "Git context:" in dynamic_context
    assert "Project instructions:" in dynamic_context
    # AGENTS.md 不在静态 system block 中，而是随首条 user 消息单独注入。
    assert f"instruction depth {MAX_INCLUDE_DEPTH}" not in dynamic_context
