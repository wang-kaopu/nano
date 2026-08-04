import asyncio
import os
import shlex
import sys
from unittest.mock import AsyncMock, patch

from nano import FakeModelClient, AgentRuntime, SessionStore, WorkspaceContext
from nano import cli as nano_cli
from nano.runtime.task_state import TaskState


def build_workspace(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def build_agent(tmp_path, outputs, **kwargs):
    workspace = build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".nano" / "sessions")
    approval_policy = kwargs.pop("approval_policy", "auto")
    return AgentRuntime(
        model_client=FakeModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy=approval_policy,
        **kwargs,
    )


def test_workspace_escape_is_rejected(tmp_path):
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")
    runtime = build_agent(tmp_path, [])

    result = runtime.run_tool("read_file", {"path": "../outside.txt"})

    assert "path escapes workspace" in result


def test_symlink_path_traversal_is_rejected(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (tmp_path / "linked.txt").symlink_to(outside)
    runtime = build_agent(tmp_path, [])

    result = runtime.run_tool("read_file", {"path": "linked.txt"})

    assert "path escapes workspace" in result


def test_safe_shell_command_does_not_require_approval(tmp_path):
    (tmp_path / "permissions.json").write_text(
        '{"permissions":{"tools":{"allow":[],"deny":[]},"shell":{"allow":["echo*"],"deny":[]}}}',
        encoding="utf-8",
    )
    runtime = build_agent(tmp_path, [], approval_policy="never")

    result = runtime.run_tool("run_shell", {"command": "echo hi", "timeout": 20})

    assert "exit_code: 0" in result


def test_project_allowed_write_file_does_not_require_approval(tmp_path):
    """项目显式放行写文件后，--approval never 不应阻断该操作。"""
    (tmp_path / "permissions.json").write_text(
        '{"permissions":{"tools":{"allow":["write_file"],"deny":[]},"shell":{"allow":[],"deny":[]}}}',
        encoding="utf-8",
    )
    runtime = build_agent(tmp_path, [], approval_policy="never")

    result = runtime.run_tool("write_file", {"path": "created.txt", "content": "created\n"})

    assert result == "wrote created.txt (8 chars)"
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "created\n"


def test_dangerous_shell_command_requires_approval(tmp_path):
    runtime = build_agent(tmp_path, [], approval_policy="never")

    result = runtime.run_tool("run_shell", {"command": "git rm -f README.md", "timeout": 20})

    assert result == "error: approval denied for run_shell"


def test_project_deny_rule_blocks_shell_command_even_when_approval_is_automatic(tmp_path):
    """项目 deny 是硬拒绝，不能被 --approval auto 覆盖。"""
    (tmp_path / "permissions.json").write_text(
        '{"permissions":{"tools":{"allow":[],"deny":[]},"shell":{"allow":["*"],"deny":["git rm*"]}}}',
        encoding="utf-8",
    )
    runtime = build_agent(tmp_path, [], approval_policy="auto")

    result = runtime.run_tool("run_shell", {"command": "git rm -f README.md", "timeout": 20})

    assert "permission denied by permissions.json" in result
    assert (tmp_path / "README.md").exists()


def test_read_only_agent_cannot_run_safe_shell_command(tmp_path):
    runtime = build_agent(tmp_path, [], approval_policy="auto", read_only=True)

    result = runtime.run_tool("run_shell", {"command": "echo hi", "timeout": 20})

    assert result == "error: approval denied for run_shell"


def test_double_ctrl_c_handler_requires_two_presses(capsys):
    class InterruptibleAgent:
        def __init__(self):
            self.interrupts = 0

        def interrupt_current_request(self):
            self.interrupts += 1
            return True

    times = iter([10.0, 11.5])
    runtime = InterruptibleAgent()
    handler = nano_cli._DoubleCtrlCInterruptHandler(runtime, clock=lambda: next(times))

    handler(None, None)
    assert runtime.interrupts == 0
    assert "Press Ctrl-C again" in capsys.readouterr().out

    handler(None, None)
    assert runtime.interrupts == 1
    assert "Interrupting current request" in capsys.readouterr().out


def test_cli_build_agent_wires_secret_env_names_from_parser(tmp_path):
    class DummyModelClient:
        model = "dummy"
        base_url = ""
        supports_prompt_cache = False
        supports_native_tool_calls = False
        native_tool_call_protocol = "openai"
        last_completion_metadata = {}

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(os.environ, {"GITHUB_PAT": "ghp-1", "GH_PAT": "ghp-2"}, clear=True):
        args = nano_cli.build_arg_parser().parse_args(
            [
                "--cwd",
                str(tmp_path),
                "--approval",
                "auto",
                "--secret-env-name",
                "GITHUB_PAT",
                "--secret-env-name",
                "GH_PAT",
            ]
        )
        runtime = nano_cli.build_agent(args)
        assert set(runtime.secret_env_summary()["secret_env_names"]) == {"GITHUB_PAT", "GH_PAT"}


def test_cli_build_agent_uses_default_configured_secret_names(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(os.environ, {"GH_PAT": "ghp-default-1"}, clear=True):
        args = nano_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--approval", "auto"])
        runtime = nano_cli.build_agent(args)
        assert runtime.secret_env_summary()["secret_env_names"] == ["GH_PAT"]


def test_cli_build_agent_loads_project_env_secrets_before_redaction_setup(tmp_path):
    class DummyModelClient:
        model = "dummy"
        base_url = ""
        supports_prompt_cache = False
        supports_native_tool_calls = False
        native_tool_call_protocol = "openai"
        last_completion_metadata = {}

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    (tmp_path / ".env").write_text("NANO_DEEPSEEK_API_KEY=sk-project-secret\n", encoding="utf-8")
    with patch.dict(os.environ, {}, clear=True), patch("nano.cli.AnthropicCompatibleModelClient", DummyModelClient):
        args = nano_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--provider", "deepseek"])
        runtime = nano_cli.build_agent(args)
        assert runtime.secret_env_summary()["secret_env_names"] == ["NANO_DEEPSEEK_API_KEY"]


def test_cli_build_agent_reads_secret_names_from_environment_config(tmp_path):
    class DummyModelClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def complete(self, prompt, max_new_tokens):
            raise AssertionError("model should not be invoked")

    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    with patch.dict(
        os.environ,
        {
            "NANO_CUSTOM_SECRET": "custom-secret-value",
            "NANO_SECRET_ENV_NAMES": "NANO_CUSTOM_SECRET",
        },
        clear=True,
    ):
        args = nano_cli.build_arg_parser().parse_args(["--cwd", str(tmp_path), "--approval", "auto"])
        runtime = nano_cli.build_agent(args)
        assert runtime.secret_env_summary()["secret_env_names"] == ["NANO_CUSTOM_SECRET"]


def test_run_shell_uses_allowlisted_environment_only(tmp_path):
    secret = "shh-allowlist-secret"
    runtime = build_agent(tmp_path, [], approval_policy="auto")
    script = 'import os; print(os.getenv("NANO_ALLOWLIST_SECRET", "missing"))'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    with patch.dict(os.environ, {"NANO_ALLOWLIST_SECRET": secret}, clear=False):
        result = runtime.run_tool("run_shell", {"command": command, "timeout": 20})

    assert secret not in result
    assert "missing" in result


def test_bound_tool_methods_delegate_into_tools_module(tmp_path):
    runtime = build_agent(tmp_path, [], approval_policy="auto")

    with patch("nano.tools.tools.subprocess.run") as fake_run:
        fake_run.return_value = type(
            "Result",
            (),
            {"returncode": 0, "stdout": "toolkit-shell\n", "stderr": ""},
        )()
        shell_result = runtime.tool_run_shell({"command": "echo bypass", "timeout": 20})

    assert "toolkit-shell" in shell_result
    fake_run.assert_called_once()
    assert runtime.tool_run_shell.__func__.__module__ == "nano.runtime.runtime"

    with patch("nano.tools.tools.tool_delegate", new=AsyncMock(return_value="toolkit-delegate")) as fake_delegate:
        delegate_result = asyncio.run(runtime.tool_delegate({"task": "inspect README.md", "max_steps": 2}))

    assert delegate_result == "toolkit-delegate"
    fake_delegate.assert_called_once()


def test_delegate_depth_limit_is_enforced(tmp_path):
    runtime = build_agent(tmp_path, [], depth=1, max_depth=1)

    try:
        runtime.validate_tool("delegate", {"task": "inspect README.md", "max_steps": 2})
    except ValueError as exc:
        assert "delegate depth exceeded" in str(exc)
    else:
        raise AssertionError("delegate depth validation did not fail")


def test_explorer_delegate_child_is_read_only(tmp_path):
    target = tmp_path / "child-was-not-allowed.txt"
    runtime = build_agent(
        tmp_path,
        [
            '<tool>{"name":"delegate","args":{"task":"write a file","type":"explorer","max_steps":2}}</tool>',
            '<tool>{"name":"write_file","args":{"path":"child-was-not-allowed.txt","content":"nope"}}</tool>',
            "<final>parent done</final>",
            "<final>explorer done</final>",
            "<final>parent done</final>",
            "<final>parent done</final>",
        ],
    )

    result = asyncio.run(runtime.ask_async("Delegate the work"))

    assert result == "parent done"
    assert not target.exists()
    tool_events = [item for item in runtime.session["history"] if item["role"] == "tool"]
    delegate_event = next(item for item in tool_events if item["name"] == "delegate")
    assert '"status": "async_launched"' in delegate_event["content"]
    assert any(item["name"] == "async_agent_notification" for item in tool_events)


def test_configured_secret_env_names_are_redacted_in_trace_and_report(tmp_path):
    github_pat = "ghp_configured_secret_123"
    gh_pat = "ghp_configured_secret_456"
    with patch.dict(os.environ, {"GITHUB_PAT": github_pat, "GH_PAT": gh_pat}, clear=True):
        runtime = build_agent(
            tmp_path,
            [],
            secret_env_names=("GITHUB_PAT", "GH_PAT"),
        )
        state = TaskState.create(run_id="run_001", task_id="task_001", user_request="Mask configured secrets")
        runtime.run_store.start_run(state)

        assert set(runtime.secret_env_summary()["secret_env_names"]) == {"GITHUB_PAT", "GH_PAT"}

        payload = {
            "GITHUB_PAT": github_pat,
            "GH_PAT": gh_pat,
            "nested": {"GITHUB_PAT": github_pat, "GH_PAT": gh_pat},
            "list": [github_pat, gh_pat],
        }
        runtime.emit_trace(state, "tool_executed", payload)
        runtime.run_store.write_report(
            state,
            runtime.redact_artifact({"task_state": state.to_dict(), "payload": payload}),
        )

    run_dir = runtime.run_store.run_dir(state.run_id)
    trace_text = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
    report_text = (run_dir / "report.json").read_text(encoding="utf-8")

    assert github_pat not in trace_text
    assert gh_pat not in trace_text
    assert github_pat not in report_text
    assert gh_pat not in report_text
    assert trace_text.count("<redacted>") >= 4
    assert report_text.count("<redacted>") >= 4
