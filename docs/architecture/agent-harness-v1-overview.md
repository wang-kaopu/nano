# Agent Harness v1 Overview

Agent Harness v1 is Nano's current runtime shape: a local control loop around a model, repository context, constrained tools, task state, memory, and auditable run artifacts.

## Runtime Flow

1. Build workspace context and runtime prefix.
2. Record the user request in session history.
3. Create task state for the run.
4. Build bounded prompt context.
5. Request the model response.
6. Parse the response into a tool call, retry notice, or final answer.
7. Start eligible read-only tools while the response stream is still active, then execute remaining tools in call order through runtime policy.
8. Write task state, trace events, checkpoints, and report artifacts.

`delegate` is a structured join operation: one call accepts a bounded task batch, starts all child agents concurrently, and returns only after every child reaches a terminal state. Child results are part of the tool result rather than asynchronous parent-session notifications. Explorer targets are registered before the child starts; child results expose `evidenceComplete` and `missingTargets`, so a parent can recover only unread evidence. Explorer `list_files` calls use a runtime-owned five-call quota and do not consume ordinary tool steps. `read_file` returns opaque-cursor pagination metadata and runtime-maintained coverage so long files can be read without relying on model-managed line arithmetic. A final response is successful only when the provider reports a complete termination; output-limit responses are regenerated once without tools or remain stopped.

## State Artifacts

- `task_state.json` records attempts, tool steps, status, stop reason, and final answer.
- `trace.jsonl` records the event timeline for prompt, model, tool, checkpoint, and finish phases.
- `report.json` records the review summary, prompt metadata, and execution metadata. Persistent memories are independent Markdown files under `.nano/projects/<cwd-hash>/memory/`.
