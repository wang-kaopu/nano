# nano

`nano` 是一个面向代码仓库的轻量本地 coding agent。它直接跑在终端里，先看当前工作区，再用一组受约束的工具去读文件、改文件、跑命令，并把会话状态保存在本地 `.nano/` 目录里。

它更像一个能在仓库里持续工作的命令行助手，不是纯聊天窗口。你可以拿它做代码排查、测试修复、仓库分析，或者让它在当前项目里执行一次性的工程任务。

## 适合做什么

- 在本地仓库里排查测试失败
- 读取当前代码结构并给出修改建议
- 基于现有文件做小步迭代，而不是脱离仓库空想
- 在会话中保留上下文，支持继续上一次工作

## 主要特性

- 包名是 `nano`
- CLI 命令是 `nano`
- 模块入口是 `python -m nano`
- 会话保存在 `.nano/sessions/`
- 每次运行的工件保存在 `.nano/runs/<run_id>/`
- 项目记忆保存在 `.nano/projects/<cwd-sha256 前 16 位>/memory/`，其中 `MEMORY.md` 是索引
- 支持三类模型后端：
  - OpenAI 兼容 Responses API
  - Anthropic 兼容 Messages API
  - DeepSeek Anthropic 兼容 API
- CLI 会实时打印模型响应正文；模型响应使用异步流式处理，`list_files`、`read_file`、`search` 的完整调用会在响应结束前并行抢跑，其余工具在同一轮流结束后按顺序执行
- provider 遇到 429、503、529、连接重置、超时或 `overloaded` 时最多重试 3 次；等待时间为 `min(1000 × 2^attempt, 30000) + random(0, 1000)` 毫秒，以指数退避和抖动避免重试风暴
- 模型上下文保留最新消息；达到 6 条后只保留最近 3 条，完整运行过程仍写入 run trace

## 使用截图

CLI 帮助信息：

![nano help](assets/screenshots/nano-help.png)

启动界面：

![nano start](assets/screenshots/nano-start.png)

REPL 内置命令与会话路径：

![nano repl](assets/screenshots/nano-repl.png)

## 安装

需要 Python 3.10+。

如果你用 `uv`，直接安装依赖：

```bash
uv sync
```

如果你已经在自己的 Python 环境里工作，也可以直接装成可编辑模式：

```bash
pip install -e .
```

开发时可运行静态类型检查：

```bash
uv run pyright
```

## 快速开始

在当前仓库里启动交互模式。默认 provider 是 DeepSeek：

```bash
uv run nano
```

指定另一个工作目录：

```bash
uv run nano --cwd /path/to/repo
```

直接跑一次性任务：

```bash
uv run nano "inspect the test failures and propose a fix"
```

如果当前环境已经安装过包，也可以直接这样启动：

```bash
python -m nano
```

## 模型后端

Nano 启动时会读取项目根目录的 `.env`。本地真实 key 放在 `.env`，仓库只保留 `.env.example`。配置优先级是：

```text
显式 CLI 参数 > .env 里的 NANO_* 变量 > 旧环境变量 > 代码默认值
```

不传 `--provider` 时默认使用 `deepseek`。这是推荐配置路径：DeepSeek 的 Anthropic-compatible endpoint 比 OpenAI-compatible/Anthropic-compatible 代理少一层默认 gateway 假设。其他 provider 仍然保留，可以显式传 `--provider openai` 或 `--provider anthropic`。

`.env` 会在构建 provider client 前加载，并覆盖当前进程里的同名环境变量。模型名和 base URL 可以通过 `--model`、`--base-url` 临时覆盖；API key 只从环境变量读取。

本地第一次配置：

```bash
cp .env.example .env
```

然后把要使用的 provider key 填进去。`.env` 已经被 `.gitignore` 忽略，不要提交真实 key。

### 推荐配置：DeepSeek

最小配置只需要 key：

```bash
NANO_DEEPSEEK_API_KEY="your-api-key"
```

默认模型和接口是：

```bash
NANO_DEEPSEEK_API_BASE="https://api.deepseek.com/anthropic"
NANO_DEEPSEEK_MODEL="deepseek-v4-pro"
```

所以常规情况下 `.env` 里只填 `NANO_DEEPSEEK_API_KEY` 就能直接启动：

```bash
uv run nano
```

如果你需要临时切模型或代理地址，不必改 `.env`，可以直接覆盖：

```bash
uv run nano --model deepseek-v4-pro --base-url https://api.deepseek.com/anthropic
```

DeepSeek 当前走 Anthropic-compatible Messages API，所以 runtime 里复用的是 Anthropic-compatible client；这只影响 HTTP 协议，不影响 CLI 用法。

### 可选配置：right.codes

right.codes 在 Nano 里有两条可选 provider 路径：

- `--provider openai`：走 OpenAI-compatible `/responses`，默认 base URL 是 `https://www.right.codes/codex/v1`，默认模型是 `gpt-5.4`
- `--provider anthropic`：走 Anthropic-compatible `/messages`，默认 base URL 是 `https://www.right.codes/claude/v1`，默认模型是 `claude-sonnet-4-6`

如果 right.codes 给你的是一把共享 key，推荐只填这一项：

```bash
NANO_RIGHT_CODES_API_KEY="your-right-codes-key"
```

然后按需要选择 provider：

```bash
uv run nano --provider openai
uv run nano --provider anthropic
```

如果你想显式区分两条 provider 的 key，也可以分别配置：

```bash
NANO_OPENAI_API_KEY="your-right-codes-key-for-codex"
NANO_ANTHROPIC_API_KEY="your-right-codes-key-for-claude"
```

不要在 `.env` 里写 `NANO_OPENAI_API_KEY=$NANO_RIGHT_CODES_API_KEY` 这种 shell 展开形式；Nano 的 `.env` 解析器只读取字面量，不展开变量引用。要么只写 `NANO_RIGHT_CODES_API_KEY`，要么把 key 字符串分别填到 provider-specific 变量里。

如果请求 right.codes 返回 `API Key额度不足`，说明协议和 endpoint 已经打通，但当前 key 没有可用额度；换一把有额度的 key，或到 right.codes 后台处理额度。

当前 provider 环境变量：

| provider | base URL | API key | model |
| --- | --- | --- | --- |
| `deepseek` | `NANO_DEEPSEEK_API_BASE`，回退 `DEEPSEEK_API_BASE`，默认 `https://api.deepseek.com/anthropic` | `NANO_DEEPSEEK_API_KEY`，回退 `DEEPSEEK_API_KEY` | `NANO_DEEPSEEK_MODEL`，回退 `DEEPSEEK_MODEL`，默认 `deepseek-v4-pro` |
| `openai` | `NANO_OPENAI_API_BASE`，回退 `OPENAI_API_BASE`，默认 `https://www.right.codes/codex/v1` | `NANO_OPENAI_API_KEY`，回退 `OPENAI_API_KEY`、`NANO_RIGHT_CODES_API_KEY`、`RIGHT_CODES_API_KEY`、`NANO_ANTHROPIC_API_KEY`、`ANTHROPIC_API_KEY` | `NANO_OPENAI_MODEL`，回退 `OPENAI_MODEL`，默认 `gpt-5.4` |
| `anthropic` | `NANO_ANTHROPIC_API_BASE`，回退 `ANTHROPIC_API_BASE`，默认 `https://www.right.codes/claude/v1` | `NANO_ANTHROPIC_API_KEY`，回退 `ANTHROPIC_API_KEY`、`NANO_RIGHT_CODES_API_KEY`、`RIGHT_CODES_API_KEY`、`NANO_OPENAI_API_KEY`、`OPENAI_API_KEY` | `NANO_ANTHROPIC_MODEL`，回退 `ANTHROPIC_MODEL`，默认 `claude-sonnet-4-6` |

如果有额外的敏感环境变量需要从 trace/report 里脱敏，可以用 `NANO_SECRET_ENV_NAMES` 配置逗号分隔的变量名，或启动时重复传 `--secret-env-name NAME`。

### OpenAI 兼容接口

如果要改用 OpenAI-compatible `/responses` 服务，显式传 `--provider openai`：

```bash
uv run nano --provider openai
```

默认 OpenAI 兼容接口使用 right.codes 的 Codex endpoint：

```bash
NANO_OPENAI_API_BASE="https://www.right.codes/codex/v1"
NANO_RIGHT_CODES_API_KEY="your-right-codes-key"
NANO_OPENAI_MODEL="gpt-5.4"
```

也可以改成其他 OpenAI-compatible 服务：

```bash
NANO_OPENAI_API_BASE="https://your-api.example/v1"
NANO_OPENAI_API_KEY="your-api-key"
NANO_OPENAI_MODEL="gpt-5.4"
```

### Anthropic 兼容接口

如果要改用 Anthropic-compatible 服务，显式传 `--provider anthropic`：

```bash
uv run nano --provider anthropic
```

默认 Anthropic 兼容接口使用 right.codes 的 Claude endpoint：

```bash
NANO_ANTHROPIC_API_BASE="https://www.right.codes/claude/v1"
NANO_RIGHT_CODES_API_KEY="your-right-codes-key"
NANO_ANTHROPIC_MODEL="claude-sonnet-4-6"
```

如果你的服务端对多个兼容接口复用了同一套密钥，`nano` 也支持从 `NANO_ANTHROPIC_API_KEY` 回退到 `ANTHROPIC_API_KEY`、`NANO_RIGHT_CODES_API_KEY`、`RIGHT_CODES_API_KEY`、`NANO_OPENAI_API_KEY` 或 `OPENAI_API_KEY`。

## 常用交互命令

在 `nano>` 输入 `/` 会显示命令菜单；使用上下方向键选择，按回车确认。

- `/help`：查看内置命令
- `/memory`：列出当前项目的持久文件记忆
- `/session`：查看当前会话文件路径
- `/resume`：弹出已保存会话列表，显示最新消息和更新时间；上下键选择，回车恢复，`Esc` 退出
- `/reset`：清空当前会话状态
- `/exit` 或 `/quit`：退出 REPL

## 项目记忆

持久记忆以独立 Markdown 文件保存在 `.nano/projects/<cwd-sha256 前 16 位>/memory/`。`MEMORY.md` 是自动维护的索引；其他文件以 `{type}_{slugified_name}.md` 命名，并使用 `name`、`description`、`type` 三个 YAML frontmatter 字段。类型是封闭集合：`user` 记录用户身份、偏好和知识背景；`feedback` 记录用户的纠正及验证过的正向行为；`project` 记录项目进展、决策和截止日期；`reference` 记录外部系统位置。

`feedback` 正文必须包含 `**Why:**` 和 `**How to apply:**`，使规则在边界情况中可判断。`project` 记忆必须把“周四”“下周”等相对日期改为绝对 ISO 日期，例如 `2026-03-05`，避免记忆过期后失去语义。

模型保存记忆时使用 `write_file` 直接写入该目录。每次请求会将文件名和描述交给同一模型的 side query 进行语义选择；完整正文只读取最多 5 条未在当前会话展示过的记忆。单文件最多 4KB，整场会话累计最多 60KB。

## 安全与持久化

`nano` 不会默认把所有动作都放开。仓库根目录的 `permissions.json` 定义项目级工具策略；默认策略已随项目提交。`run_shell` 会用 `bashlex` 解析 Bash AST，复合命令中的每个片段都需要独立匹配 allow 规则，避免用安全前缀夹带危险子命令。无法解析为 AST 的命令会在执行前拒绝。

- `--approval ask`
- `--approval auto`
- `--approval never`

规则使用 glob 模式，支持普通工具名和 `run_shell(...)`：allow 命中可免审批，未命中则继续走审批，deny 命中则直接拒绝执行。deny 永远优先于 allow，因此可以先放开一组命令再排除危险子命令：

```json
{
  "permissions": {
    "allow": ["run_shell(git *)"],
    "deny": ["run_shell(git push --force*)"]
  }
}
```

`permissions.json` 默认只放行本地检查及常见构建、测试命令，例如 `rg`、`git status`、`git diff`、`pytest`、`ruff`、`pyright`、`uv run pytest`、`npm test` 与 `npm run build`；默认 `deny` 为空，未被 allow 的命令继续走审批。项目需要硬禁止某类命令时，可自行在 `deny` 中添加规则。用户拒绝审批时，本次运行会立即停止，不能改用替代命令绕过该决定。

每次运行结束后，都会在 `.nano/runs/<run_id>/` 下写出这些文件：

- `task_state.json`
- `trace.jsonl`
- `report.json`

这些内容默认只保存在本地，不需要跟仓库一起提交。

## 开发

常用本地检查：

```bash
uv run pytest tests -q
uv run ruff check nano tests scripts
```

内部代码按职责边界拆分：`nano/runtime/` 管理请求执行与恢复，`nano/tools/` 管理工具与安全边界，`nano/storage/` 管理持久化，`nano/workspace/` 管理仓库快照，`nano/memory/` 管理工作记忆与长期记忆，`nano/evaluation/` 保留 benchmark 与 metrics，`nano/providers/` 管理模型 provider client。新代码应直接使用这些包路径。
