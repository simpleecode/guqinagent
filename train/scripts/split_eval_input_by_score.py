#!/usr/bin/env python3
"""Split an evaluation JSONL into balanced, whole-score shards."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be positive")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[str(row["score_key"])].append(row)
    if not grouped:
        raise SystemExit("input is empty")

    buckets: list[list[dict]] = [[] for _ in range(args.shards)]
    loads = [0] * args.shards
    for _, rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        index = min(range(args.shards), key=lambda candidate: loads[candidate])
        buckets[index].extend(rows)
        loads[index] += len(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, rows in enumerate(buckets):
        rows.sort(key=lambda row: (str(row["score_key"]), str(row["phrase_id"])))
        path = args.output_dir / f"shard_{index:02d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        manifest.append({"shard": index, "phrases": len(rows), "scores": len({row["score_key"] for row in rows}), "path": str(path)})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
