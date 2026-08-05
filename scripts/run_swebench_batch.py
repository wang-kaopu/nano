#!/usr/bin/env python3
"""SWE-bench Lite 批量评测一键启动脚本。

用法:
  python3 scripts/run_swebench_batch.py              # 全量 5 实例，并行 5
  python3 scripts/run_swebench_batch.py --instances 14365            # 单实例
  python3 scripts/run_swebench_batch.py --instances 14365 6938      # 指定实例
  python3 scripts/run_swebench_batch.py --serial                       # 串行
  python3 scripts/run_swebench_batch.py --max-steps 60                 # 60 步上限
  python3 scripts/run_swebench_batch.py --dry-run                     # 预演
  python3 scripts/run_swebench_batch.py --resume /root/swebench-runs/five-astropy-xxx  # 续跑
  python3 scripts/run_swebench_batch.py --tail                         # 启动后 tail -f 日志
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
BATCH_RUNNER = PROJECT_ROOT / "benchmarks" / "run_swebench_batch.py"

# 五实例固定列表
INSTANCE_MAP = {
    "12907": "astropy__astropy-12907",
    "14182": "astropy__astropy-14182",
    "14365": "astropy__astropy-14365",
    "14995": "astropy__astropy-14995",
    "6938": "astropy__astropy-6938",
}
ALL_INSTANCES = list(INSTANCE_MAP.values())


def load_env() -> dict[str, str]:
    """从 .env 文件加载环境变量。"""
    env = os.environ.copy()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


def resolve_instances(short_ids: list[str]) -> list[str]:
    """支持短 ID 匹配，如 14365 → astropy__astropy-14365。"""
    result = []
    for sid in short_ids:
        if sid in INSTANCE_MAP:
            result.append(INSTANCE_MAP[sid])
        elif sid.startswith("astropy__"):
            result.append(sid)
        else:
            candidates = [v for k, v in INSTANCE_MAP.items() if sid in k]
            if len(candidates) == 1:
                result.append(candidates[0])
            else:
                print(f"⚠ 无法识别实例: {sid}（候选: {candidates}）", file=sys.stderr)
                sys.exit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-bench Lite 一键启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  %(prog)s                              # 全量 5 实例，并行\n"
               "  %(prog)s --instances 14365             # 单实例\n"
               "  %(prog)s --serial                       # 串行执行\n",
    )
    parser.add_argument(
        "--instances", nargs="*",
        help="要跑的实例（短 ID 或全名），默认全部 5 个",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="串行执行（默认并行 5）",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=5,
        help="最大并发数（默认 5）",
    )
    parser.add_argument(
        "--max-steps", type=int, default=40,
        help="Agent 最大步数（默认 40）",
    )
    parser.add_argument(
        "--agent-timeout", type=int, default=2700,
        help="Agent 超时秒数（默认 2700）",
    )
    parser.add_argument(
        "--evaluation-timeout", type=int, default=1800,
        help="评分超时秒数（默认 1800）",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="模型温度（默认 0.0）",
    )
    parser.add_argument(
        "--provider", type=str, default="deepseek",
        help="模型提供商（默认 deepseek）",
    )
    parser.add_argument(
        "--output-root", type=str, default="/root/swebench-runs/",
        help="产物根目录",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预演模式，不实际执行",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="续跑：指定已有批次的 output_root 路径",
    )
    parser.add_argument(
        "--tail", action="store_true",
        help="启动后自动 tail -f 日志",
    )
    parser.add_argument(
        "--log-file", type=str, default="/tmp/swe-batch-output.log",
        help="日志文件路径",
    )

    args = parser.parse_args()

    # 解析实例
    if args.instances:
        instances = resolve_instances(args.instances)
    else:
        instances = ALL_INSTANCES

    parallel = 1 if args.serial else args.max_parallel

    print(f"{'🔍 DRY-RUN ' if args.dry_run else '🚀'} SWE-bench Lite 批量评测")
    print(f"   实例: {len(instances)} 个")
    for inst in instances:
        short = next((k for k, v in INSTANCE_MAP.items() if v == inst), inst)
        print(f"     - {short} → {inst}")
    print(f"   并发: {parallel}")
    print(f"   步数: {args.max_steps}  温度: {args.temperature}")
    print(f"   Agent超时: {args.agent_timeout}s  评分超时: {args.evaluation_timeout}s")
    print()

    if args.dry_run:
        # dry-run 也传完整参数验证
        cmd = [
            sys.executable, str(BATCH_RUNNER),
            "--dry-run",
        ]
    else:
        # 加载 .env
        env = load_env()

        cmd = [
            sys.executable, str(BATCH_RUNNER),
            "--instances", *instances,
            "--output-root", args.output_root,
            "--max-parallel", str(parallel),
            "--max-steps", str(args.max_steps),
            "--agent-timeout", str(args.agent_timeout),
            "--evaluation-timeout", str(args.evaluation_timeout),
            "--provider", args.provider,
            "--temperature", str(args.temperature),
        ]

        if args.resume:
            cmd.extend(["--resume", "--output-root", args.resume])

        # 写入日志
        print(f"   日志: {args.log_file}")
        print(f"   命令: {' '.join(cmd)}")
        print()

        with open(args.log_file, "w") as log:
            log.write(f"# SWE-bench Lite batch started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"# Command: {' '.join(cmd)}\n\n")

        proc = subprocess.Popen(
            cmd,
            stdout=open(args.log_file, "a"),
            stderr=subprocess.STDOUT,
            env=env,
        )

        print(f"   PID: {proc.pid}")
        print()

        if args.tail:
            print("--- 实时日志 (Ctrl+C 退出 tail，不影响进程) ---\n")
            try:
                subprocess.run(["tail", "-f", args.log_file])
            except KeyboardInterrupt:
                print(f"\n\n已退出 tail，进程 {proc.pid} 仍在后台运行")
                print(f"查看日志: tail -f {args.log_file}")
                print(f"停止进程: kill {proc.pid}")


if __name__ == "__main__":
    main()
