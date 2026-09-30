#!/usr/bin/env python3
"""Reconstruct and dump full evaluator messages for overlong vLLM requests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    public_system_for,
    public_tools_for,
    render_public_prompt,
)
from train.scripts.eval_two_stage_score import openai_tools  # noqa: E402


def rows_from_input(path: Path) -> dict[str, dict]:
    return {
        str(row["sample_id"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="evaluation test JSONL containing runtime_item")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--stage", choices=("base", "guqinizer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = rows_from_input(args.input).get(args.sample_id)
    if row is None:
        raise SystemExit(f"sample ID not found: {args.sample_id}")
    item = row.get("runtime_item") or row
    basic = args.stage == "base"
    tools = public_tools_for(args.stage, basic=basic)
    messages = [
        {"role": "system", "content": public_system_for(args.stage, basic=basic)},
        {"role": "user", "content": render_public_prompt(item, args.stage)},
    ]
    document = {
        "sample_id": args.sample_id,
        "stage": args.stage,
        "messages": messages,
        "tools": openai_tools(tools),
        "runtime_item": item,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
