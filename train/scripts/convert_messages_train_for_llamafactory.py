#!/usr/bin/env python3
"""Export a generated trajectory bundle to the stable LLaMA-Factory layout.

This reuses the project's production exporter: assistant reasoning and native
tool calls become one Qwen3.5 XML text message, while tool observations remain
context.  It avoids Arrow's heterogeneous nested ``tool_calls`` schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from export_sft_dataset import build_trajectory_example  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    replacements = 0

    def clean(value):
        nonlocal replacements
        if isinstance(value, str):
            hits = value.count("Agent 已填写")
            replacements += hits
            return value.replace("Agent 已填写", "已填写")
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value

    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = clean(json.loads(line))
            if not isinstance(row.get("messages"), list):
                raise ValueError(f"line {line_number}: messages must be an array")
            example, _ = build_trajectory_example(row)
            target.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"rows": count, "replacements": replacements, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
