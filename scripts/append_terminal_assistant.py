#!/usr/bin/env python3
"""Migrate accepted teacher trajectories to an explicit assistant stop turn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def final_text(stage: str) -> str:
    return (
        "工具预览已通过，当前段基础减字填写完成。"
        if stage == "fingering_agent"
        else "工具预览已通过，当前段减字润色完成。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input and output must differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = migrated = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages") or []
            if (
                messages
                and messages[-1].get("role") == "tool"
                and (row.get("termination") or {}).get("kind") == "accepted_edit_plan"
            ):
                messages.append({"role": "assistant", "content": final_text(str(row.get("agent_stage") or ""))})
                migrated += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    print(json.dumps({"rows": rows, "migrated": migrated}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
