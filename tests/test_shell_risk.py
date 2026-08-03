import json

import pytest

from nano.permissions import ProjectPermissions, load_project_permissions
from nano.tools.shell_risk import ShellCommandParseError, shell_command_segments


def test_shell_permission_rules_match_each_ast_command_segment():
    """复合命令必须让每个 AST 命令片段都命中 allow，才能免审批。"""
    policy = ProjectPermissions(allow=("run_shell(git status*)", "run_shell(git diff*)"))

    assert policy.decision("run_shell", "git status --short && git diff --cached") == "allow"
    assert policy.decision("run_shell", "git status --short && rm README.md") == "no_match"


def test_deny_rule_takes_priority_over_broad_run_shell_allow():
    """deny 必须覆盖 run_shell 的宽泛 allow，防止配置放开后无法收紧。"""
    policy = ProjectPermissions(allow=("run_shell(git *)",), deny=("run_shell(git push --force*)",))

    assert policy.decision("run_shell", "git status --short") == "allow"
    assert policy.decision("run_shell", "git push --force origin main") == "deny"


def test_generic_run_shell_allow_still_respects_deny_rules():
    """用户显式放开 run_shell 时，deny 仍能阻止危险子命令。"""
    policy = ProjectPermissions(allow=("run_shell",), deny=("run_shell(git rm*)",))

    assert policy.decision("run_shell", "git status") == "allow"
    assert policy.decision("run_shell", "git rm -f README.md") == "deny"


def test_load_project_permissions_reads_root_permissions_file(tmp_path):
    """项目级策略只从仓库根目录的 permissions.json 加载。"""
    (tmp_path / "permissions.json").write_text(
        json.dumps({"permissions": {"allow": ["grep_search"], "deny": ["run_shell(rm *)"]}}),
        encoding="utf-8",
    )

    policy = load_project_permissions(tmp_path)

    assert policy.decision("search") == "allow"
    assert policy.decision("run_shell", "rm -rf build") == "deny"


def test_shell_command_segments_include_nested_command_substitutions():
    """嵌套命令也必须参与权限匹配，不能借安全外层命令绕过策略。"""
    segments = shell_command_segments("echo $(rm -f README.md)")

    assert [segment.text for segment in segments] == ["echo $(rm -f README.md)", "rm -f README.md"]


def test_invalid_shell_syntax_is_rejected_before_permission_matching():
    """无法构建 AST 的命令不能绕过项目权限规则。"""
    with pytest.raises(ShellCommandParseError):
        shell_command_segments("echo 'unterminated")
