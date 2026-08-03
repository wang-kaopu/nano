from nano.memory.frontmatter import parse_frontmatter
from nano.memory.memory import (
    MAX_INDEX_BYTES,
    MAX_INDEX_LINES,
    MAX_MEMORY_BYTES_PER_FILE,
    LayeredMemory,
    get_memory_dir,
    load_memory_index,
    save_memory,
)


def test_working_memory_tracks_summary_and_recent_files():
    memory = LayeredMemory()

    memory.set_task_summary("Investigate flaky tests")
    memory.remember_file("README.md")
    memory.remember_file("src/app.py")
    memory.remember_file("README.md")

    snapshot = memory.to_dict()

    assert snapshot["working"]["task_summary"] == "Investigate flaky tests"
    assert snapshot["working"]["recent_files"] == ["src/app.py", "README.md"]


def test_file_memory_is_saved_with_frontmatter_and_indexed_by_project_hash(tmp_path):
    memory_dir = get_memory_dir(tmp_path)

    filename = save_memory(
        "Do not add response summaries",
        "The user prefers direct final responses.",
        "feedback",
        'The user said "do not add a summary at the end".\n\n**Why:** They review the diff directly.',
        memory_dir,
    )

    assert memory_dir == tmp_path / ".nano" / "projects" / memory_dir.parent.name / "memory"
    assert filename == "feedback_do-not-add-response-summaries.md"
    parsed = parse_frontmatter((memory_dir / filename).read_text(encoding="utf-8"))
    assert parsed.meta == {
        "name": "Do not add response summaries",
        "description": "The user prefers direct final responses.",
        "type": "feedback",
    }
    assert "They review the diff directly." in parsed.body
    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "**[Do not add response summaries](feedback_do-not-add-response-summaries.md)** (feedback)" in index


def test_memory_index_truncates_by_lines_then_bytes(tmp_path):
    memory_dir = get_memory_dir(tmp_path)
    memory_dir.mkdir(parents=True)
    index_path = memory_dir / "MEMORY.md"
    index_path.write_text("\n".join(f"entry-{index}" for index in range(MAX_INDEX_LINES + 5)), encoding="utf-8")

    assert "too many memory entries" in load_memory_index(memory_dir)

    index_path.write_text("x" * (MAX_INDEX_BYTES + 1), encoding="utf-8")
    assert "index too large" in load_memory_index(memory_dir)


def test_semantic_recall_reads_selected_files_once_and_applies_file_limit(tmp_path):
    memory = LayeredMemory(workspace_root=tmp_path)
    assert memory.memory_dir is not None
    ci_filename = save_memory(
        "CI dashboard",
        "URL and workflow notes for continuous integration.",
        "reference",
        "x" * (MAX_MEMORY_BYTES_PER_FILE + 100),
        memory.memory_dir,
    )
    save_memory(
        "Database notes",
        "PostgreSQL indexing guidance.",
        "project",
        "Unrelated to deployment.",
        memory.memory_dir,
    )
    calls = []

    def side_query(system_prompt, user_prompt):
        calls.append((system_prompt, user_prompt))
        return '{"selected_memories": ["' + ci_filename + '"]}'

    selected = memory.select_relevant_memories("How do I deploy?", side_query)

    assert len(selected) == 1
    assert selected[0].filename == ci_filename
    assert selected[0].content.endswith("[... truncated, memory file too large ...]")
    assert len(selected[0].content.encode("utf-8")) <= MAX_MEMORY_BYTES_PER_FILE
    assert ci_filename in calls[0][1]
    assert "x" * 100 not in calls[0][1]
    assert memory.to_dict()["surfaced_memory_bytes"] == len(selected[0].content.encode("utf-8"))

    assert memory.select_relevant_memories("How do I deploy?", side_query) == []
    assert len(calls) == 2


def test_file_summaries_use_canonical_paths_and_freshness(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\n", encoding="utf-8")
    memory = LayeredMemory(workspace_root=tmp_path)

    memory.set_file_summary("./sample.txt", "sample.txt: alpha")
    memory.remember_file("./sample.txt")

    assert "sample.txt: alpha" in memory.render_memory_text()
    file_path.write_text("beta\n", encoding="utf-8")
    assert "sample.txt: alpha" not in memory.render_memory_text()
