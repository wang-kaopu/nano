import json
import os

from nano.storage.session_store import SessionStore


def test_session_store_saves_loads_and_finds_latest_session(tmp_path):
    store = SessionStore(tmp_path / ".nano" / "sessions")
    first = {"id": "session_001", "history": [{"role": "user", "content": "first"}]}
    second = {"id": "session_002", "history": [{"role": "user", "content": "second"}]}

    first_path = store.save(first)
    second_path = store.save(second)

    assert first_path == store.path("session_001")
    assert json.loads(first_path.read_text(encoding="utf-8"))["id"] == "session_001"
    loaded = store.load("session_002")
    assert loaded["id"] == second["id"]
    assert loaded["history"] == second["history"]
    assert loaded["checkpoints"] == {"current_id": "", "items": {}}
    assert loaded["resume_state"]["status"] == "no-checkpoint"
    assert store.latest() == second_path.stem


def test_session_store_latest_is_none_when_empty(tmp_path):
    store = SessionStore(tmp_path / ".nano" / "sessions")

    assert store.latest() is None


def test_session_store_recreates_removed_directory_before_save(tmp_path):
    store = SessionStore(tmp_path / ".nano" / "sessions")
    store.root.rmdir()

    path = store.save({"id": "session_recreated", "history": []})

    assert path.exists()


def test_session_store_lists_summaries_by_update_time_with_latest_message(tmp_path):
    store = SessionStore(tmp_path / ".nano" / "sessions")
    older_path = store.save({"id": "session_older", "history": [{"role": "user", "content": "Older message"}]})
    newer_path = store.save(
        {
            "id": "session_newer",
            "history": [
                {"role": "user", "content": "Initial question"},
                {"role": "assistant", "content": "Latest reply"},
            ],
        }
    )
    os.utime(older_path, (100, 100))
    os.utime(newer_path, (200, 200))

    summaries = store.list_summaries()

    assert [item["id"] for item in summaries] == ["session_newer", "session_older"]
    assert summaries[0]["title"] == "Initial question"
    assert summaries[0]["updated_at"]
