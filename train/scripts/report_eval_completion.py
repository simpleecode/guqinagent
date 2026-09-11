#!/usr/bin/env python3
"""Report phrase and whole-score completion for public evaluation output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for split in ("validation", "test"):
        expected: dict[str, set[str]] = {}
        for row in rows(args.input_root / f"{split}.jsonl"):
            expected.setdefault(row["score_key"], set()).add(row["sample_id"])
        latest = {row["sample_id"]: row for row in rows(args.output_root / f"{split}_predictions.jsonl")}
        valid = {
            sample_id for sample_id, row in latest.items()
            if row.get("protocol_valid") and row.get("jianzi_rows")
        }
        complete = [
            {"score_key": score, "phrases": len(ids)}
            for score, ids in expected.items() if ids <= valid
        ]
        partial = [
            {"score_key": score, "valid": len(ids & valid), "phrases": len(ids)}
            for score, ids in expected.items() if ids & valid and not ids <= valid
        ]
        partial.sort(key=lambda row: (-row["valid"] / row["phrases"], row["score_key"]))
        report[split] = {
            "expected_phrases": sum(map(len, expected.values())),
            "output_unique": len(latest),
            "valid_phrases": len(valid),
            "complete_score_count": len(complete),
            "complete_scores": sorted(complete, key=lambda row: row["score_key"]),
            "best_partial": partial[:10],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
