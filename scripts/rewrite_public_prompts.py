#!/usr/bin/env python3
"""Rewrite the public system prompt for every accepted trajectory.

The private teacher audit is left untouched.  This is a local, no-API
normalization step used after merging rerun outputs so old and new public
trajectories use the same current prompt contract.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from generate_teacher_tool_trajectories import public_system_for


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    shutil.copytree(args.input_dir, args.output_dir)
    path = args.output_dir / "messages_train.jsonl"
    rows = changed = 0
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in records:
            rows += 1
            stage = row.get("agent_stage")
            if stage in {"fingering_agent", "guqinization", "single_stage"}:
                expected = public_system_for(stage, basic=stage == "fingering_agent")
                for message in row.get("messages") or []:
                    if message.get("role") == "system":
                        if message.get("content") != expected:
                            message["content"] = expected
                            changed += 1
                        break
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"schema_version": "public-prompt-rewrite-1.0",
              "rows": rows, "changed_system_messages": changed,
              "input_dir": str(args.input_dir), "output_dir": str(args.output_dir)}
    (args.output_dir / "public_prompt_rewrite_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
