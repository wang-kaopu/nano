from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano.runtime.checkpoint import (
    CHECKPOINT_FULL_VALID_STATUS,
    CHECKPOINT_NONE_STATUS,
    CHECKPOINT_SCHEMA_MISMATCH_STATUS,
    CHECKPOINT_SCHEMA_VERSION,
    current_runtime_identity,
    evaluate_resume_state,
)


def build_agent(tmp_path, outputs=None, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".nano" / "sessions")
    return AgentRuntime(
        model_client=FakeModelClient(outputs or []),
        workspace=workspace,
        session_store=store,
        approval_policy=kwargs.pop("approval_policy", "auto"),
        **kwargs,
    )


def test_current_runtime_identity_captures_execution_contract(tmp_path):
    runtime = build_agent(tmp_path, max_steps=9, max_new_tokens=1024, max_final_tokens=4096, max_final_retries=1, read_only=True)

    identity = current_runtime_identity(runtime)

    assert identity["session_id"] == runtime.session["id"]
    assert identity["cwd"] == str(tmp_path)
    assert identity["read_only"] is True
    assert identity["max_steps"] == 9
    assert identity["max_new_tokens"] == 1024
    assert identity["max_final_tokens"] == 4096
    assert identity["max_final_retries"] == 1
    assert identity["workspace_fingerprint"] == runtime.workspace.fingerprint()
    assert identity["tool_signature"] == runtime.tool_signature()


def test_evaluate_resume_state_distinguishes_no_checkpoint_full_valid_and_schema_mismatch(tmp_path):
    runtime = build_agent(tmp_path)

    assert evaluate_resume_state(runtime)["status"] == CHECKPOINT_NONE_STATUS

    identity = current_runtime_identity(runtime)
    runtime.session["checkpoints"] = {
        "current_id": "ckpt_valid",
        "items": {
            "ckpt_valid": {
                "checkpoint_id": "ckpt_valid",
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "key_files": [],
                "runtime_identity": identity,
            }
        },
    }
    assert evaluate_resume_state(runtime)["status"] == CHECKPOINT_FULL_VALID_STATUS

    runtime.session["checkpoints"]["items"]["ckpt_valid"]["schema_version"] = "old"
    assert evaluate_resume_state(runtime)["status"] == CHECKPOINT_SCHEMA_MISMATCH_STATUS
