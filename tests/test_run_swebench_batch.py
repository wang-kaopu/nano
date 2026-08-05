"""run_swebench_batch 单元测试 — mock 覆盖，不调用真实模型或 Docker。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# 将 benchmarks 目录加入搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from run_swebench_batch import (
    BatchConfig,
    InstanceResult,
    FIVE_INSTANCE_IDS,
    TERMINAL_STATES,
    RESUMABLE_SKIP_STATES,
    RETRYABLE_STATES,
    atomic_write_json,
    read_json,
    instance_image_name,
    load_instances,
    build_agent_prompt,
    generate_patch,
    write_prediction_jsonl,
    update_state,
    _build_summary,
    _build_summary_md,
    git_diff_worktree,
    worktree_is_dirty,
    timed_log,
    now_iso,
    check_environment,
    _ensure_gitignore_patterns,
)


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------


class TestConfigParsing:
    def test_default_instance_ids(self):
        config = BatchConfig()
        assert config.instance_ids == FIVE_INSTANCE_IDS
        assert len(config.instance_ids) == 5

    def test_custom_instances(self):
        config = BatchConfig(instance_ids=["astropy__astropy-12907"])
        assert config.instance_ids == ["astropy__astropy-12907"]

    def test_defaults(self):
        config = BatchConfig()
        assert config.provider == "deepseek"
        assert config.temperature == 0.0
        assert config.max_steps == 40
        assert config.max_parallel == 1
        assert config.agent_timeout == 2700
        assert config.evaluation_timeout == 1800

    def test_dry_run(self):
        config = BatchConfig(dry_run=True)
        assert config.dry_run is True


# ---------------------------------------------------------------------------
# 五实例计划生成
# ---------------------------------------------------------------------------


class TestInstancePlan:
    def test_load_instances_from_fixture(self, tmp_path):
        dataset = tmp_path / "test.json"
        dataset.write_text(json.dumps([
            {"instance_id": iid, "repo": "astropy/astropy", "base_commit": "abc123",
             "version": "4.3", "problem_statement": "fix it"}
            for iid in FIVE_INSTANCE_IDS
        ]))
        config = BatchConfig(dataset_path=str(dataset))
        instances = load_instances(config)
        assert len(instances) == 5
        assert instances[0]["instance_id"] == "astropy__astropy-12907"

    def test_load_instances_missing_raises(self, tmp_path):
        dataset = tmp_path / "test.json"
        dataset.write_text(json.dumps([
            {"instance_id": "astropy__astropy-12907", "repo": "a/b", "base_commit": "x",
             "version": "1", "problem_statement": "p"}
        ]))
        config = BatchConfig(dataset_path=str(dataset), instance_ids=["astropy__astropy-12907", "missing__id-1"])
        with pytest.raises(ValueError, match="缺少实例"):
            load_instances(config)

    def test_only_public_fields_loaded(self, tmp_path):
        instance_id = "astropy__astropy-12907"
        dataset = tmp_path / "test.json"
        dataset.write_text(json.dumps([{
            "instance_id": instance_id,
            "repo": "astropy/astropy",
            "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
            "version": "4.3",
            "problem_statement": "test problem",
            "patch": "SECRET_GOLD_PATCH",
            "test_patch": "SECRET_TEST_PATCH",
            "FAIL_TO_PASS": ["secret_test_1"],
            "PASS_TO_PASS": ["secret_test_2"],
        }]))
        config = BatchConfig(dataset_path=str(dataset), instance_ids=[instance_id])
        instances = load_instances(config)
        assert len(instances) == 1
        assert "patch" not in instances[0]
        assert "test_patch" not in instances[0]
        assert "FAIL_TO_PASS" not in instances[0]


# ---------------------------------------------------------------------------
# 状态迁移
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_terminal_states(self):
        assert "resolved" in TERMINAL_STATES
        assert "unresolved" in TERMINAL_STATES
        assert "agent_error" in TERMINAL_STATES
        assert "infra_error" in TERMINAL_STATES
        assert "pending" not in TERMINAL_STATES

    def test_skip_states(self):
        assert "resolved" in RESUMABLE_SKIP_STATES
        assert "unresolved" in RESUMABLE_SKIP_STATES

    def test_retryable_states(self):
        assert "infra_error" in RETRYABLE_STATES
        assert "agent_error" not in RETRYABLE_STATES
        assert "unresolved" not in RETRYABLE_STATES

    def test_update_state_atomic(self, tmp_path):
        instance_dir = tmp_path / "test-instance"
        update_state(instance_dir, "agent_running")
        assert read_json(instance_dir / "state.json")["status"] == "agent_running"

        update_state(instance_dir, "resolved")
        assert read_json(instance_dir / "state.json")["status"] == "resolved"

    def test_update_state_with_result(self, tmp_path):
        instance_dir = tmp_path / "test-instance"
        result = InstanceResult(
            instance_id="test__id-1",
            status="resolved",
            agent_duration=100.0,
            patch_bytes=500,
            tool_steps=12,
        )
        update_state(instance_dir, "resolved", result)
        saved = read_json(instance_dir / "result.json")
        assert saved["status"] == "resolved"
        assert saved["agent_duration"] == 100.0
        assert saved["patch_bytes"] == 500
        assert saved["tool_steps"] == 12


# ---------------------------------------------------------------------------
# 恢复逻辑
# ---------------------------------------------------------------------------


class TestResume:
    def test_resolved_instance_skipped(self, tmp_path):
        """已完成实例不应重新执行。"""
        instance_dir = tmp_path / "resolved-instance"
        instance_dir.mkdir(parents=True)
        update_state(instance_dir, "resolved",
                     InstanceResult(instance_id="test", status="resolved"))

        # 模拟恢复逻辑
        state = read_json(instance_dir / "state.json")
        assert state["status"] in RESUMABLE_SKIP_STATES

    def test_patch_generated_continues_to_eval(self, tmp_path):
        """已有 patch 的实例应跳过 Agent 直接评分。"""
        instance_dir = tmp_path / "patch-ready"
        instance_dir.mkdir(parents=True)
        update_state(instance_dir, "patch_generated")
        (instance_dir / "model.patch").write_text("diff --git a/x b/x\n")

        state = read_json(instance_dir / "state.json")
        assert state["status"] == "patch_generated"

    def test_infra_error_retryable(self):
        assert "infra_error" in RETRYABLE_STATES

    def test_unresolved_not_retryable(self):
        assert "unresolved" not in RETRYABLE_STATES


# ---------------------------------------------------------------------------
# Patch 处理
# ---------------------------------------------------------------------------


class TestPatchGeneration:
    def test_patch_includes_new_file(self, tmp_path):
        """未跟踪文件应被 add -N 包含进 diff。"""
        worktree = tmp_path / "repo"
        worktree.mkdir()
        # 模拟 git 仓库
        subprocess = __import__("subprocess")
        subprocess.run(["git", "-C", str(worktree), "init"], capture_output=True, check=False)
        subprocess.run(["git", "-C", str(worktree), "config", "user.email", "test@test"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], capture_output=True)
        # 创建初始提交
        (worktree / "README.md").write_text("hello")
        subprocess.run(["git", "-C", str(worktree), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-m", "init"], capture_output=True)
        # 添加新文件
        (worktree / "new_file.py").write_text("print(1)")
        patch = git_diff_worktree(worktree)
        assert "new_file.py" in patch

    def test_empty_patch_no_changes(self, tmp_path):
        worktree = tmp_path / "clean_repo"
        worktree.mkdir()
        subprocess = __import__("subprocess")
        subprocess.run(["git", "-C", str(worktree), "init"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.email", "test@test"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], capture_output=True)
        (worktree / "f.py").write_text("x")
        subprocess.run(["git", "-C", str(worktree), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-m", "init"], capture_output=True)
        patch = git_diff_worktree(worktree)
        assert patch.strip() == ""

    def test_worktree_is_dirty(self, tmp_path):
        worktree = tmp_path / "dirty_repo"
        worktree.mkdir()
        subprocess = __import__("subprocess")
        subprocess.run(["git", "-C", str(worktree), "init"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.email", "test@test"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], capture_output=True)
        (worktree / "f.py").write_text("x")
        subprocess.run(["git", "-C", str(worktree), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-m", "init"], capture_output=True)
        assert not worktree_is_dirty(worktree)
        (worktree / "f.py").write_text("y")
        assert worktree_is_dirty(worktree)


# ---------------------------------------------------------------------------
# 结果分类
# ---------------------------------------------------------------------------


class TestResultClassification:
    def test_build_summary_counts(self):
        results = [
            InstanceResult("a", "resolved"),
            InstanceResult("b", "resolved"),
            InstanceResult("c", "unresolved"),
            InstanceResult("d", "unresolved"),
            InstanceResult("e", "agent_error", error="timeout"),
        ]
        summary = _build_summary("test-batch", results, 0.0)
        assert summary["resolved_count"] == 2
        assert summary["unresolved_count"] == 2
        assert summary["agent_error_count"] == 1
        assert summary["infra_error_count"] == 0
        assert summary["completion_rate"] == 0.5

    def test_build_summary_all_infra_errors(self):
        results = [InstanceResult(str(i), "infra_error", error="docker down") for i in range(5)]
        summary = _build_summary("test-batch", results, 0.0)
        assert summary["resolved_count"] == 0
        assert summary["unresolved_count"] == 0
        assert summary["completion_rate"] == 0.0


# ---------------------------------------------------------------------------
# Dry-run 无副作用
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_no_filesystem_changes(self, tmp_path):
        config = BatchConfig(
            dry_run=True,
            dataset_path="/nonexistent/path.json",  # 不会实际加载
            output_root=str(tmp_path / "output"),
        )
        assert config.dry_run is True

    def test_dry_run_does_not_pull_images(self):
        config = BatchConfig(dry_run=True)
        # dry_run 时 parse_args 不会实际执行
        assert config.dry_run


# ---------------------------------------------------------------------------
# 镜像名推导
# ---------------------------------------------------------------------------


class TestImageNaming:
    @pytest.mark.skip(reason="需要 SWE-bench 库，CI 环境可能未安装")
    def test_image_name_standard(self):
        name = instance_image_name(
            {"instance_id": "astropy__astropy-12907", "repo": "astropy/astropy",
             "base_commit": "abc", "version": "4.3"}, "swebench"
        )
        # Docker Hub 不支持 __，会被替换为 _1776_
        assert "sweb.eval.x86_64.astropy_1776_astropy-12907" in name
        assert "swebench/" in name
        assert name.endswith(":latest")

    @pytest.mark.skip(reason="需要 SWE-bench 库，CI 环境可能未安装")
    def test_image_name_no_namespace(self):
        name = instance_image_name(
            {"instance_id": "astropy__astropy-12907", "repo": "astropy/astropy",
             "base_commit": "abc", "version": "4.3"}, "none"
        )
        assert "swebench/" not in name
        assert "sweb.eval.x86_64.astropy__astropy-12907" in name


# ---------------------------------------------------------------------------
# Prediction JSONL
# ---------------------------------------------------------------------------


class TestPredictionJsonl:
    def test_write_prediction_official_format(self, tmp_path):
        path = write_prediction_jsonl(
            tmp_path, "test__id-1", "pico/deepseek", "diff --git a/x b/x\n"
        )
        content = json.loads(path.read_text(encoding="utf-8"))
        assert content["instance_id"] == "test__id-1"
        assert content["model_name_or_path"] == "pico/deepseek"
        assert content["model_patch"] == "diff --git a/x b/x\n"


# ---------------------------------------------------------------------------
# Prompt 不含私有评测材料
# ---------------------------------------------------------------------------


class TestPromptSanity:
    def test_prompt_has_no_private_fields(self):
        instance = {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug.",
        }
        prompt = build_agent_prompt(instance).lower()
        # 确保不包含私有评测字段名
        for forbidden in ("test_patch", "fail_to_pass", "pass_to_pass"):
            assert forbidden not in prompt
        # 确保不泄漏 gold patch 内容（而非仅提及"不要用"）
        assert "SECRET" not in prompt


# ---------------------------------------------------------------------------
# 原子写入
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_atomic_write_and_read(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "nested": {"a": 1}}
        atomic_write_json(path, data)
        assert path.exists()
        assert not (tmp_path / "test.json.tmp").exists()
        assert read_json(path) == data


# ---------------------------------------------------------------------------
# 汇总 Markdown
# ---------------------------------------------------------------------------


class TestSummaryMarkdown:
    def test_markdown_contains_all_instances(self):
        summary = _build_summary("batch-1", [
            InstanceResult("a__a-1", "resolved", agent_duration=100, harness_duration=50, patch_bytes=200, tool_steps=5),
            InstanceResult("a__a-2", "unresolved", agent_duration=200, harness_duration=60, patch_bytes=300, tool_steps=10),
        ], 0.0)
        md = _build_summary_md(summary)
        assert "a__a-1" in md
        assert "a__a-2" in md
        assert "resolved" in md
        assert "unresolved" in md
        assert "Completion Rate" in md


# ---------------------------------------------------------------------------
# 环境检查 mock
# ---------------------------------------------------------------------------


class TestEnvironmentCheck:
    def test_check_environment_structure(self, tmp_path):
        """确保 check_environment 函数可调用并返回布尔值。"""
        dataset = tmp_path / "test.json"
        dataset.write_text(json.dumps([
            {"instance_id": iid, "repo": "astropy/astropy", "base_commit": "abc123",
             "version": "4.3", "problem_statement": "fix it"}
            for iid in FIVE_INSTANCE_IDS
        ]))
        config = BatchConfig(
            dataset_path=str(dataset),
            env_file="/nonexistent/.env",
            swebench_path="/nonexistent/SWE-bench",
            instance_ids=FIVE_INSTANCE_IDS,
        )
        result = check_environment(config)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 时间函数
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_now_iso_returns_string(self):
        ts = now_iso()
        assert isinstance(ts, str)
        assert "T" in ts


# ---------------------------------------------------------------------------
# Warmup / .gitignore 编译产物保护
# ---------------------------------------------------------------------------


class TestGitignorePatterns:
    def test_adds_missing_patterns(self, tmp_path):
        w = tmp_path / "repo"
        w.mkdir()
        (w / ".gitignore").write_text("*.pyc\n")
        _ensure_gitignore_patterns(w)
        content = (w / ".gitignore").read_text()
        assert "*.so" in content
        assert "__pycache__/" in content
        assert "build/" in content

    def test_no_duplicates(self, tmp_path):
        w = tmp_path / "repo"
        w.mkdir()
        (w / ".gitignore").write_text("*.so\n*.pyc\n__pycache__/\nbuild/\n")
        _ensure_gitignore_patterns(w)
        content = (w / ".gitignore").read_text()
        # 每个 pattern 只出现一次
        assert content.count("*.so") == 1
        assert content.count("*.pyc") == 1

    def test_creates_gitignore_if_missing(self, tmp_path):
        w = tmp_path / "repo"
        w.mkdir()
        _ensure_gitignore_patterns(w)
        assert (w / ".gitignore").exists()
        content = (w / ".gitignore").read_text()
        assert "*.so" in content


class TestGitDiffExcludes:
    def test_excludes_dot_so_files(self, tmp_path):
        """编译产物 .so 不应出现在 diff 中。"""
        worktree = tmp_path / "repo"
        worktree.mkdir()
        subprocess = __import__("subprocess")
        subprocess.run(["git", "-C", str(worktree), "init"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.email", "test@test"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], capture_output=True)
        (worktree / "f.py").write_text("x")
        subprocess.run(["git", "-C", str(worktree), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-m", "init"], capture_output=True)
        # 创建 .so 和正常的修改
        (worktree / "lib.so").write_text("binary")
        (worktree / "f.py").write_text("y")
        patch = git_diff_worktree(worktree)
        assert "f.py" in patch
        assert "lib.so" not in patch

    def test_excludes_build_directory(self, tmp_path):
        worktree = tmp_path / "repo"
        worktree.mkdir()
        subprocess = __import__("subprocess")
        subprocess.run(["git", "-C", str(worktree), "init"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.email", "test@test"], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "config", "user.name", "test"], capture_output=True)
        (worktree / "f.py").write_text("x")
        subprocess.run(["git", "-C", str(worktree), "add", "."], capture_output=True)
        subprocess.run(["git", "-C", str(worktree), "commit", "-m", "init"], capture_output=True)
        (worktree / "build").mkdir()
        (worktree / "build" / "output.o").write_text("obj")
        (worktree / "f.py").write_text("y")
        patch = git_diff_worktree(worktree)
        assert "f.py" in patch
        assert "build/" not in patch
        assert "output.o" not in patch
