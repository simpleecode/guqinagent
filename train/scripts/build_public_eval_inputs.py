#!/usr/bin/env python3
"""Strip sealed references from validation/test evaluation pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        source = args.source_dir / f"evaluation_pairs_{split}.jsonl"
        target = args.output_dir / f"{split}.jsonl"
        count = 0
        with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8", newline="\n") as dst:
            for line in src:
                if not line.strip():
                    continue
                row = json.loads(line)
                dst.write(json.dumps({
                    "schema_version": "agent-eval-input-1.0",
                    "sample_id": row["sample_id"],
                    "split": split,
                    "score_key": row.get("score_key"),
                    "input": row["input"],
                }, ensure_ascii=False) + "\n")
                count += 1
        print(split, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
