#!/usr/bin/env python3
"""SWE-bench Lite pilot 的真实模型命令行入口。"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nano.evaluation.swebench_lite import run_pilot  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 pilot 参数，并将默认值锁定为可复现的三次重复实验。"""
    parser = argparse.ArgumentParser(description="Run the isolated SWE-bench Lite real-model pilot.")
    parser.add_argument("--provider", choices=("openai", "anthropic", "deepseek"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="benchmarks/swebench_lite_pilot.json")
    parser.add_argument("--swebench-path", required=True, help="Path to an official SWE-bench checkout.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instance-id", action="append", dest="instance_ids", default=[])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--repeats", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行 pilot，并将完整机器可读汇总写到 stdout。"""
    args = build_arg_parser().parse_args(argv)
    summary = run_pilot(
        provider=args.provider,
        model=args.model,
        manifest_path=args.manifest,
        swebench_path=args.swebench_path,
        output_dir=args.output_dir,
        instance_ids=args.instance_ids or None,
        temperature=args.temperature,
        max_steps=args.max_steps,
        timeout=args.timeout,
        repeats=args.repeats,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
