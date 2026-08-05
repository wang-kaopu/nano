# SWE-bench Lite 五实例批次分析

**批次 ID:** five-astropy-20260805-195955
**结果:** 0/5 resolved
**分析日期:** 2026-08-05

---

## 根本原因

所有 5 个实例的 patch 中均**没有任何实际代码改动**，全部是 `.nano/` 运行时产物。Agent 探索了代码但没有使用 `patch_file` 工具写入修复。

## 各实例详情

### Instance 1: astropy__astropy-12907 (separability_matrix)
- **9 次尝试，6 步有效，3 步无效**
- Agent 调用了 `search`, `list_files`, `read_file` 但没有 `patch_file`
- 3 次无效调用：使用了 `cursor`/`page_size` 参数（pico 工具 schema 不支持）
- 最终状态: "Stopped after too many invalid tool calls"
- **根因**: DeepSeek 生成的工具调用参数与 pico schema 不匹配

### Instance 2: astropy__astropy-14182 (fixedwidth)
- **44 次尝试，全部有效**
- 30 次 `run_shell` (sed/grep)、11 次 `read_file`、3 次 `search`
- 零次 `patch_file` — 探索了代码但没有修改
- 最后一次尝试 invalid "during the final step"
- **根因**: Agent 没有进入实施阶段

### Instance 3: astropy__astropy-14365 (qdp case sensitivity) ⭐
- **41 次尝试**
- **正确识别了根因和修复方案**（`qdp.py:63` 正则表达式区分大小写）
- Agent 的 final_answer 明确描述了需要将 `_command_re` 改为不区分大小写
- **但没有实施修复** — "Fix not yet applied due to exhausted tool budget"
- 浪费了 ~20 次调用尝试编译 C 扩展、搜索 `extension_helpers`、遍历文件系统
- **如果 Agent 能跳过编译尝试直接修改代码，此实例很可能 resolved**

### Instance 4: astropy__astropy-14995 (time coordinates)
- **42 次尝试**
- 34 次 `run_shell`、7 次 `read_file`、1 次 `search`
- 零次 `patch_file`
- **根因**: Agent 没有进入实施阶段

### Instance 5: astropy__astropy-6938 (fits D exponent)
- **20 次尝试，4 次无效**
- 9 次 `search`、12 次 `read_file`、2 次 `run_shell`
- 无效调用：`cursor` 参数不匹配
- "Stopped after too many invalid tool calls"
- **根因**: DeepSeek 工具参数不匹配 + 零次 `patch_file`

---

## 问题分类与修复方案

### P0: Agent 从不调用 `patch_file` 工具（核心问题）

**现象**: 5 个 Agent 共产生 ~170 次工具调用，`patch_file` 调用次数为 0。

**分析**:
- Agent 使用 `run_shell` 的 sed/grep 探索代码，使用 `read_file` 阅读
- 但从不使用 `patch_file` 写入修改
- 可能是 `patch_file` 工具的描述或 schema 对 DeepSeek 不友好
- 也可能 Agent 的 prompt 没有充分引导使用 `patch_file`

**修复方案**:
1. **Prompt 增强**: 在 `build_agent_prompt` 中显式指引 Agent 使用 `patch_file` 工具
2. **工具说明优化**: 检查 `patch_file` 的工具描述是否足够清晰
3. **添加 `write_file` 工具**: 如果 `patch_file` 的 unified diff 格式对模型不友好，可考虑支持简单的文件写入

```python
# 改进后的 prompt 示例
def build_agent_prompt(instance):
    return textwrap.dedent(f"""\
        Fix {instance['instance_id']}.
        {instance['problem_statement']}
        
        ## 工作步骤
        1. 使用 read_file/list_files/grep 理解代码
        2. 使用 patch_file 工具实施修改（不要只分析不修改！）
        3. 使用 run_shell 运行相关测试验证修复
        
        ## patch_file 使用说明
        patch_file 接受 unified diff 格式的补丁。示例用法：
        要修改 astropy/example.py 第 42 行的 `x = 1` 为 `x = 2`：
        --- a/astropy/example.py
        +++ b/astropy/example.py
        @@ -40,7 +40,7 @@
         import os
         
        -x = 1
        +x = 2
         
         def foo():
        
        你必须在分析完代码后立即使用 patch_file 写入修改。
        Do not stop after explaining the solution; implement it in the workspace.
    """)
```

### P1: DeepSeek 无效工具调用

**现象**: Agent 使用 pico schema 不存在的参数（`cursor`、`page_size`）

**分析**: DeepSeek 模型在 Anthropic API 格式训练中学习了某些工具参数，但 pico 的工具 schema 不同。模型会"幻觉"出不存在于当前 schema 的参数。

**修复方案（按优先级）**:
1. **增加容错**: 修改 pico 工具执行器，忽略未知参数而非拒绝（strict → permissive）
2. **添加兼容参数**: 在 `read_file` 等工具中添加 `cursor`、`page_size` 等兼容参数（接受但不使用）
3. **降低无效调用惩罚**: 提高 `max_invalid_tool_calls` 上限（当前可能是 3 或 5）

### P2: Agent 浪费步数在非必要操作上

**现象**: Instance 3 的 Agent 正确识别了 bug，但用 ~20 步尝试编译 C 扩展直到预算耗尽

**修复方案**:
1. **移除 `build_extensions` 步骤** ✅ 已完成
2. **在容器启动时注入信息**: 告知 Agent 扩展已预编译，不需要重新构建
3. **Shell 白名单优化**: 禁止 `pip install`、`setup.py build_ext` 等无网络环境下必然失败的命令

### P3: `.nano/` 产物污染 patch

**现象**: 所有 patch 都包含 `.nano/` 运行时目录的内容

**修复方案**: ✅ 已完成 — `git_diff_worktree` 将 `.nano/` 移至 `/tmp` 临时目录

### P4: `started_at` 时间戳错误

**现象**: 显示 `1970-01-01T04:14:17` (epoch 时间)

**修复方案**: ✅ 已完成 — 使用 `now_iso()` 替代 `datetime.fromtimestamp(time.monotonic())`

---

## 改进优先级

| 优先级 | 改进项 | 影响 | 复杂度 |
|--------|--------|------|--------|
| 🔴 P0 | Agent prompt 强制要求使用 patch_file | 5/5 实例受益 | 低 |
| 🟠 P1 | 工具执行器容错（忽略未知参数） | 2/5 实例受益 | 中 |
| 🟡 P2 | 容器注入预编译信息 + 禁止无网络命令 | 1/5 实例明显受益 | 低 |
| 🟢 P3 | `.nano/` 产物排除 | ✅ 已修复 | - |
| 🟢 P4 | 时间戳修复 | ✅ 已修复 | - |

## 预期效果

如果 Instance 3 的 Agent 能跳过编译尝试直接写入修复，该实例很可能 resolved。
P0 修复后，5 个 agent 都应能进入代码修改阶段，预期至少 1-2/5 resolved。

## 后续实验建议

1. 对 Instance 3 (qdp) 单独重新运行，验证改进效果
2. 如果 `patch_file` 格式是障碍，考虑添加 `edit_file` (old_str/new_str) 工具
3. 考虑在容器启动前运行一次 `pip install -e .`（允许短暂网络），确保测试可运行
