#!/usr/bin/env python3
"""Assemble the newest successful stale-Guqinizer rerun rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="successful rerun directory; later directories override earlier IDs")
    args = parser.parse_args()
    messages: dict[str, dict] = {}
    audits: dict[str, dict] = {}
    for run in args.run:
        for row in read(run / "messages_train.jsonl"):
            messages[row["sample_id"]] = row
        for row in read(run / "teacher_trajectory_audit.jsonl"):
            audits[row["sample_id"]] = row
    # Quality-quarantined rows can have a private audit without a public row;
    # only successful public replacements belong in the replacement bundle.
    audits = {sample_id: row for sample_id, row in audits.items() if sample_id in messages}
    if set(messages) != set(audits):
        raise SystemExit("successful replacement lacks a private audit")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("messages_train.jsonl", messages), ("teacher_trajectory_audit.jsonl", audits)):
        with (args.output / name).open("w", encoding="utf-8", newline="\n") as out:
            for sample_id in sorted(values):
                out.write(json.dumps(values[sample_id], ensure_ascii=False) + "\n")
    print(json.dumps({"replacements": len(messages), "sample_ids": sorted(messages)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
