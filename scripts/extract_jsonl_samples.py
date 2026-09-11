#!/usr/bin/env python3
"""Extract selected sample IDs from matching JSONL files without altering rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args()
    wanted = set(args.sample_id)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("messages_train.jsonl", "teacher_trajectory_audit.jsonl"):
        found: set[str] = set()
        with (args.source / name).open(encoding="utf-8") as source, \
                (args.output / name).open("w", encoding="utf-8", newline="\n") as output:
            for line in source:
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or "")
                if sample_id in wanted:
                    output.write(line.rstrip("\r\n") + "\n")
                    found.add(sample_id)
        missing = wanted - found
        if missing:
            raise SystemExit(f"{name} missing sample IDs: {sorted(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
