# SWE-bench Lite Pilot

This is a separate real-model evaluation. It does not replace or aggregate with `benchmarks/coding_tasks.json`, which remains a deterministic scripted-harness regression suite.

## Data And Selection

Tasks come from the official `princeton-nlp/SWE-bench_Lite` dataset through the official `swebench` loader. The manifest stores only public `instance_id`, repository, base commit, issue description, image identifier, tool list, and budgets. It never stores patches, hidden tests, evaluator commands, or test output.

`sympy__sympy-20590` is the first locked task. Its local official gold run resolved on this machine. The Requests and Django candidates are intentionally absent: their local gold pre-run was blocked while Docker reported insufficient disk space building the Requests environment image. Add one task from each repository only after that exact official gold pre-run resolves locally.

The target selection is one SymPy, one Requests, and one Django task. Three tasks are a pilot sample, not evidence for a general SWE capability ranking.

## Isolation

For every task and repeat, Pico copies `/testbed` from the declared prebuilt SWE-bench instance image into a new worktree, resets it to the recorded base commit, and starts a new `--network none` container. The model sees only the worktree and receives shell access through `docker exec`; it never receives host shell access. Evaluation metadata is not mounted into the container. The official harness applies its private evaluation material only after Pico has collected the final git diff.

The pilot permission policy is `benchmarks/swebench_permissions.json`. It permits file inspection, search, `patch_file`, and test/format/status/diff shell commands. Docker, network download, workspace-external deletion, and evaluation-configuration mutation are not available to the model.

## Configuration And Metrics

Lock provider, model name, temperature, task manifest, step budget, timeout, and repeat count for a comparison. Use at least three repeats per task per model. Report overall resolved rate, unresolved count, errors, and every task/repeat result. The resolved decision is read only from the official `swebench.harness.run_evaluation` report.

Each run saves `prompt.json`, Pico runtime trace/session artifacts, `final.patch`, `predictions.json`, official scorer stdout/stderr and report, and `container.log` under the requested output directory.

## Reproduction

Install benchmark dependencies with:

```bash
uv sync --group benchmark
```

After all three gold-validated tasks are present in the manifest, run:

```bash
uv run --group benchmark python scripts/run_swebench_lite_pilot.py --provider openai --model YOUR_MODEL --swebench-path ../SWE-bench --output-dir artifacts/swebench-lite/YOUR_MODEL --temperature 0 --max-steps 40 --timeout 900 --repeats 3
```

The command emits machine-readable JSON and writes `summary.json` plus per-run artifacts.
