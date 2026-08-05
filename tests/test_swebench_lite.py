import json
from pathlib import Path

import pytest

from nano.evaluation.swebench_lite import (
    PUBLIC_INSTANCE_FIELDS,
    build_task_prompt,
    load_pilot_manifest,
    scorer_command,
    write_predictions,
)


def test_pilot_manifest_uses_public_schema_and_unique_ids():
    manifest = load_pilot_manifest(Path("benchmarks/swebench_lite_pilot.json"))

    instance_ids = [task["instance_id"] for task in manifest["tasks"]]
    assert len(instance_ids) == len(set(instance_ids))
    assert instance_ids[0] == "sympy__sympy-20590"
    assert PUBLIC_INSTANCE_FIELDS == {"instance_id", "repo", "base_commit", "problem_statement"}
    for task in manifest["tasks"]:
        assert task["allowed_tools"] == ["read_file", "list_files", "search", "grep_search", "patch_file", "run_shell"]
        assert task["step_budget"] > 0
        assert task["time_budget_seconds"] > 0


def test_pilot_manifest_rejects_non_public_fields(tmp_path):
    manifest = load_pilot_manifest(Path("benchmarks/swebench_lite_pilot.json"))
    manifest["tasks"][0]["private_field"] = "forbidden"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="public schema"):
        load_pilot_manifest(path)


def test_prompt_has_no_private_evaluation_material():
    task = load_pilot_manifest(Path("benchmarks/swebench_lite_pilot.json"))["tasks"][0]
    prompt = build_task_prompt(task).lower()

    for forbidden in ("test_patch", "gold patch", "run_evaluation", "git diff --"):
        assert forbidden not in prompt


def test_write_predictions_uses_official_json_shape(tmp_path):
    path = write_predictions(
        tmp_path / "predictions.json",
        [{"instance_id": "sympy__sympy-20590", "model_name_or_path": "provider/model", "model_patch": "diff --git a/a b/a\n"}],
    )

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {"instance_id": "sympy__sympy-20590", "model_name_or_path": "provider/model", "model_patch": "diff --git a/a b/a\n"}
    ]


def test_scorer_command_calls_official_harness_with_isolated_run_id(tmp_path):
    command = scorer_command("/venv/bin/python", "princeton-nlp/SWE-bench_Lite", tmp_path / "predictions.json", "sympy__sympy-20590", "pilot-run", 900)

    assert command[:3] == ["/venv/bin/python", "-m", "swebench.harness.run_evaluation"]
    assert command[command.index("--predictions_path") + 1] == str(tmp_path / "predictions.json")
    assert command[command.index("--instance_ids") + 1] == "sympy__sympy-20590"
    assert command[command.index("--namespace") + 1] == ""
    assert command[command.index("--timeout") + 1] == "900"


@pytest.mark.integration
def test_sandbox_lifecycle_requires_local_docker_and_swebench_image():
    pytest.skip("Requires an explicitly provisioned SWE-bench instance image and Docker disk capacity.")
