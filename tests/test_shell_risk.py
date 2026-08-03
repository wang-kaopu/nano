import json

import pytest

from nano.permissions import PermissionRules, ProjectPermissions, load_project_permissions
from nano.tools.shell_risk import ShellCommandParseError, shell_command_segments


def test_shell_permission_rules_match_each_ast_command_segment():
    """复合命令必须让每个 AST 命令片段都命中 allow，才能免审批。"""
    policy = ProjectPermissions(shell=PermissionRules(allow=("git status*", "git diff*")))

    assert policy.decision("run_shell", "git status --short && git diff --cached") == "allow"
    assert policy.decision("run_shell", "git status --short && rm README.md") == "no_match"


def test_deny_rule_takes_priority_over_broad_run_shell_allow():
    """deny 必须覆盖 run_shell 的宽泛 allow，防止配置放开后无法收紧。"""
    policy = ProjectPermissions(shell=PermissionRules(allow=("git *",), deny=("git push --force*",)))

    assert policy.decision("run_shell", "git status --short") == "allow"
    assert policy.decision("run_shell", "git push --force origin main") == "deny"


def test_generic_run_shell_allow_still_respects_deny_rules():
    """用户显式放开 run_shell 时，deny 仍能阻止危险子命令。"""
    policy = ProjectPermissions(shell=PermissionRules(allow=("*",), deny=("git rm*",)))

    assert policy.decision("run_shell", "git status") == "allow"
    assert policy.decision("run_shell", "git rm -f README.md") == "deny"


def test_load_project_permissions_reads_root_permissions_file(tmp_path):
    """项目级策略只从仓库根目录的 permissions.json 加载。"""
    (tmp_path / "permissions.json").write_text(
        json.dumps(
            {
                "permissions": {
                    "tools": {"allow": ["grep_search"], "deny": []},
                    "shell": {"allow": [], "deny": ["rm *"]},
                }
            }
        ),
        encoding="utf-8",
    )

    policy = load_project_permissions(tmp_path)

    assert policy.decision("search") == "allow"
    assert policy.decision("run_shell", "rm -rf build") == "deny"


def test_load_project_permissions_rejects_legacy_mixed_rules(tmp_path):
    """权限配置必须显式区分普通工具与 shell 命令规则。"""
    (tmp_path / "permissions.json").write_text(
        json.dumps({"permissions": {"allow": ["read_file", "run_shell(echo*)"], "deny": []}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected tools and shell objects"):
        load_project_permissions(tmp_path)


def test_shell_command_segments_include_nested_command_substitutions():
    """嵌套命令也必须参与权限匹配，不能借安全外层命令绕过策略。"""
    segments = shell_command_segments("echo $(rm -f README.md)")

    assert [segment.text for segment in segments] == ["echo $(rm -f README.md)", "rm -f README.md"]


def test_invalid_shell_syntax_is_rejected_before_permission_matching():
    """无法构建 AST 的命令不能绕过项目权限规则。"""
    with pytest.raises(ShellCommandParseError):
        shell_command_segments("echo 'unterminated")
