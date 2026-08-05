# SWE-bench Lite 第二轮批次分析

**批次 ID:** five-astropy-20260805-204149
**结果:** 2/5 resolved (50%)
**对比:** 上一轮 0/5 (0%)
**总耗时:** 420s（并行 max_parallel=5）

---

## 各实例结果

| Instance | 状态 | Agent | Eval | Patch | Steps | 分析 |
|---|---|---|---|---|---|---|
| 12907 | ⚠ agent_error | 115s | — | 0B | 10 | 无效工具调用，未调用 patch_file |
| 14182 | ✗ unresolved | 89s | 154s | 724B | 13 | 修复不完整，缺少 read() + start_line 移除 |
| 14365 | ✗ unresolved | 253s | 165s | 619B | 35 | 修复不完整，缺少 v.upper() 匹配 |
| 14995 | ✓ RESOLVED | 131s | 177s | 609B | 23 | 正确处理 operand.mask is None 边界情况 |
| 6938 | ✓ RESOLVED | 175s | 25s | 572B | 32 | 正确修复 chararray.replace 赋值问题 |

## 两个 Resolved 实例分析

### ✅ 14995 — NDArithmeticMixin 空 mask 处理
- **Patch**: 单行修复，处理 `operand.mask is None` 的边界情况
- **关键**: Agent 从 `ndarithmetic.py` 定位问题，添加 `elif operand.mask is None:` 分支
- **正确性**: 与 gold patch 一致

### ✅ 6938 — FITS_rec D exponent replace 修复
- **Patch**: 将 `output_field.replace(...)` 改为 `output_field[:] = output_field.replace(...)`
- **关键**: chararray.replace() 返回副本而非原地修改，需显式赋值
- **正确性**: 与 gold patch 逻辑一致

## 两个 Unresolved 根因分析

### ✗ 14365 — qdp 大小写 (修复不完整)
- **Agent 修复**: `re.compile(_type_re, re.IGNORECASE)` ✓
- **缺少的修复**: `if v.upper() == "NO":` (line 306)
- **原因**: 正则大小写后 "NO"/"no" 都能匹配，但 `v == "NO"` 是比较字符串文字，不会匹配小写
- **Agent 用了 35 步仍遗漏**: 未做完整的 call-chain 追踪

### ✗ 14182 — RST header_rows (修复不完整)
- **Agent 修复**: `__init__` + `write` 方法 ✓
- **缺少的修复**: 
  1. `read()` 方法需添加 `self.data.start_line = 2 + len(self.header.header_rows)`
  2. `SimpleRSTData.start_line = 3` 硬编码需删除
- **原因**: Agent 只修复了 `write` 路径，漏了 `read` 路径

## Agent 改进效果对比

| 指标 | 第一轮 | 第二轮 | 改进 |
|---|---|---|---|
| Resolved | 0 | 2 | ∞ |
| 有代码改动 | 0/5 | 4/5 | +80% |
| patch_file 调用 | 0 次 | 多次 | ∞ |
| Patch 大小 | 155K-707K | 572-724B | 99.9% 缩小 |
| `.nano/` 污染 | 100% | 0% | 完全修复 |
| 无效工具调用 | 2/5 实例 | 1/5 实例 | -50% |
| Agent 平均步数 | 19 | 23 | +21% |

## 关键改进点总结

1. **Prompt 增强** (P0) → 4/5 Agent 成功调用 patch_file
2. **`.nano/` 排除** (P3) → Patch 从 700KB 缩小到 ~600B
3. **并行调度** (新增) → 总耗时从 1226s 降至 420s (3x)
4. **ReadFileArguments 容错** (P1) → 无效调用明显减少

## 剩余问题

1. **Agent 修复不完整** (14182, 14365): Agent 能找到核心问题但遗漏下游影响
2. **Agent 仍尝试 pip install** (12907): 尽管 prompt 禁止，Agent 仍浪费步数在必然失败的命令上
3. **Instance 12907 全程无效调用**: 14 次尝试 0 次 patch_file，需进一步调查

## 后续建议

1. **针对 14365**: 在 prompt 中加入 "trace all callers/callees of your change"
2. **针对 14182**: 类似的 call-chain 追踪提示
3. **针对 12907**: 单独调试，可能是 separability_matrix 问题的代码结构对 Agent 理解有障碍
4. **Shell 拒绝**: 实现真正的 shell 命令过滤（当前 permissions.json deny 未生效）
