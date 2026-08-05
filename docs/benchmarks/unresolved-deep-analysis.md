# 三个 Unresolved 实例深度根因分析

**批次:** five-astropy-20260805-204149
**结果:** 2/5 resolved, 3 unresolved

---

## Instance 1: astropy__astropy-12907 — 零代码修改 (agent_error)

**Bug**: `_cstack` 中 `cright[...] = 1` 应为 `cright[...] = right`
**Gold fix**: 单行改动，`separable.py:245`，`1` → `right`
**难度**: 极低（单个字符替换）

| 指标 | 值 |
|---|---|
| 工具调用 | 14 次 |
| read_file (ok) | 4 |
| run_shell (ok) | 5 |
| run_shell (error) | 2 (Python import 失败，C 扩展未编译) |
| run_shell (rejected) | 2 (`build_ext`、`pip install` — 权限拦截生效 ✅) |
| read_file (rejected) | 1 (用了绝对路径 `/testbed/...`) |
| patch_file | **0 次** |
| 无效/错误调用占比 | 5/14 = 36% |

**根因链**:
```
read_file 正确阅读代码 ✓
→ 想用 python 验证理解 ✗ (import astropy 失败)
→ 尝试 build_ext / pip install ✗ (权限拦截 + 无网络)
→ 步数耗尽，从未调用 patch_file
```

**Agent 做了什么**: 阅读了 `separable.py`（正确文件），看了第 219-247 行（包含 `_cstack` 函数），但转身去修环境而非修代码。

**为什么 prompt 引导失效**: Agent 正确理解了 "Explore" 步骤，但卡在了 "Verify" 执念上 — 想在修代码前先确认能跑 Python。应该先 patch 再 verify。

---

## Instance 2: astropy__astropy-14182 — 修复不完整 (unresolved)

**Bug**: RST 格式不支持 `header_rows` 参数
**Gold fix**: 3 处改动
1. `SimpleRSTData.start_line = 3` → 删除（硬编码移除）
2. `__init__` → 接受 `header_rows` 参数
3. `write()` → 动态计算 header 行索引
4. `read()` → 设置 `self.data.start_line = 2 + len(...)`（**遗漏**）

| 指标 | 值 |
|---|---|
| 工具调用 | 16 次 |
| search (ok) | 2 |
| read_file (ok) | 5 |
| patch_file (ok) | **1 ✅** |
| run_shell (ok) | 2 |
| run_shell (error) | 3 (Python 测试失败) |
| 其他 rejected/error | 3 |
| 无效/错误调用占比 | 6/16 = 38% |

**Agent 修复** (部分正确):
```diff
-    def __init__(self):
-        super().__init__(delimiter_pad=None, bookend=False)
+    def __init__(self, header_rows=None):
+        super().__init__(delimiter_pad=None, bookend=False, header_rows=header_rows)
 
     def write(self, lines):
         lines = super().write(lines)
-        lines = [lines[1]] + lines + [lines[1]]
+        idx = len(self.header.header_rows)
+        lines = [lines[idx]] + lines + [lines[idx]]
```

**遗漏的修复**:
```diff
-    start_line = 3                                    # ← 未删除

+    def read(self, table):                             # ← 整个方法未添加
+        self.data.start_line = 2 + len(self.header.header_rows)
+        return super().read(table)
```

**根因**: Agent 只修复了 `write()` 路径，没有追踪 `read()` 路径。16 次调用中 6 次(38%)浪费在失败/被拒的操作上，只剩 10 次有效调用 — 不足以完成完整的 call-chain 追踪。

---

## Instance 3: astropy__astropy-14365 — 修复不完整 (unresolved) ⭐

**Bug**: QDP 命令大小写敏感 + "NO" 值大小写比较
**Gold fix**: 2 处改动
1. `re.compile(_type_re)` → `re.compile(_type_re, re.IGNORECASE)` ✅ Agent 已做
2. `if v == "NO":` → `if v.upper() == "NO":` ❌ Agent 遗漏

| 指标 | 值 |
|---|---|
| 工具调用 | **38 次**（最努力的 Agent） |
| search (ok) | 2 |
| read_file (ok) | 5 |
| patch_file (ok) | **3 ✅** |
| run_shell (ok) | 15 |
| run_shell (error) | **11 (Python import 反复失败)** |
| run_shell (rejected) | 2 |
| 无效/错误调用占比 | 13/38 = 34% |

**Agent 做了什么**:
- 调用 1-5: 搜索→阅读→修改（5 步完成初版修复！非常高效）
- 调用 6-26: **疯狂验证正则**（20 步！）— 用 15+ 种不同方式测试 regex 行为
- 调用 27-38: 继续读代码+尝试 pytest（仍然失败于 C 扩展）

**Agent 修复**:
```diff
-    _line_type_re = re.compile(_type_re)
+    _line_type_re = re.compile(_type_re, re.IGNORECASE)
```

**遗漏的修复** (line 306):
```diff
-                if v == "NO":
+                if v.upper() == "NO":
```

**根因**: Agent 对 regex 修改做了极其详尽的验证（20 步！），但验证范围过度聚焦在 `_line_type` 函数本身，从未向下追踪到 `_get_tables_from_qdp_file` 函数中解析出的值如何被使用。11 次 Python 调用失败（"tool_failed"），Agent 不断重试不同的 import 方式，陷入死循环。

---

## 三类失败模式总结

| 模式 | 实例 | 特征 | 浪费原因 |
|---|---|---|---|
| **"先验证再修改"** | 12907 | 0 patch_file, 想先跑 Python 验证 | 环境限制导致验证必然失败 |
| **"修复不完整"** | 14182 | 1 patch_file, 修了 write 漏了 read | 步数不够 + 未追踪 call-chain |
| **"过度验证"** | 14365 | 3 patch_file, 正则验证 20 步 | 聚焦过窄 + Python 反复失败 |

---

## 改进优先级

### P0: 防止 Agent 陷入"先验证 Python→失败→修复环境"循环

**影响**: 12907(完全卡住) + 14182(浪费 38%) + 14365(浪费 34%)

**方案**:
1. **Prompt 强化**: 在 Workflow 步骤中明确加入 `⚠ 不要先运行 python/pytest — 导入可能因 C 扩展失败。先 patch_file 修改代码，评分器会验证测试。`
2. **Shell 白名单收紧**: 把 `python *` 从隐式允许改为需要确认（但对于 auto 模式，可能需要在 prompt 层面解决）
3. **环境注入**: 在 Agent 启动前告知 `Python import astropy 可能失败，使用 read_file 理解代码即可`

### P1: 引导 call-chain 追踪

**影响**: 14182 + 14365（两个都是修复主逻辑正确但遗漏下游影响）

**方案**: Prompt 中加入步骤：
```
3. **Trace downstream**: 修改后，搜索所有使用你修改的函数/变量的地方。
   使用 grep_search 查找调用者，确保没有其他地方需要适配。
```

### P2: 步数预算保护

**影响**: 所有实例（平均 30%+ 步数浪费在失败操作上）

**方案**: 
- 相同命令连续失败 3 次 → 自动拒绝后续类似命令
- 在 runtime 层面实现（非 batch runner 层面）

---

## 数据对比

| 指标 | 12907 | 14182 | 14365 | 14995(✅) | 6938(✅) |
|---|---|---|---|---|---|
| patch_file 调用 | 0 | 1 | 3 | ? | ? |
| run_shell error 占比 | 29% | 19% | 29% | ? | ? |
| 被拒命令数 | 3 | 3 | 2 | ? | ? |
| 遇到 Python import 问题 | ✓ | ✓ | ✓ | ? | ? |
| 修复完整性 | 未开始 | 50% | 50% | 100% | 100% |
| **根本失败原因** | 验证执念 | 追踪不全 | 追踪不全+过度验证 | — | — |
