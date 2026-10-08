#!/usr/bin/env python3
"""Merge disjoint vLLM evaluation shard predictions by sample ID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shard-dir", type=Path, action="append", required=True,
        help="Exact shard output directory; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: dict[str, dict] = {}
    for shard_dir in args.shard_dir:
        path = shard_dir / "predictions.jsonl"
        if not path.exists():
            raise SystemExit(f"missing shard prediction file: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[str(record["sample_id"])] = record
    ordered = sorted(records.values(), key=lambda record: str(record["sample_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "records": len(ordered)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
