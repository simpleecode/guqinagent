#!/usr/bin/env python3
"""Merge accepted outputs from parallel worker output directories."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


FILES = {
    "messages_train.jsonl": "sample_id",
    "teacher_trajectory_audit.jsonl": "sample_id",
    "fingering_intermediates.jsonl": "trajectory_id",
}


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dirs = []
    for root in args.worker_root:
        dirs.extend(sorted(root.glob("worker_*/output")))
        if (root / "messages_train.jsonl").exists():
            dirs.append(root)
    merged: dict[str, list[dict]] = {}
    for filename, key in FILES.items():
        by_key: dict[str, dict] = {}
        for directory in dirs:
            for row in rows(directory / filename):
                by_key[str(row[key])] = row
        merged[filename] = list(by_key.values())
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="\n") as out:
            for row in merged[filename]:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    rejected = []
    for directory in dirs:
        rejected.extend(rows(directory / "teacher_rejected_io.jsonl"))
    with (args.output_dir / "teacher_rejected_io.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for row in rejected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    public_ids = {row["sample_id"] for row in merged["messages_train.jsonl"]}
    private = [row for row in merged["teacher_trajectory_audit.jsonl"] if row.get("sample_id") in public_ids]
    with (args.output_dir / "teacher_trajectory_audit.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for row in private:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "teacher-parallel-merged-1.0",
        "worker_roots": [str(root) for root in args.worker_root],
        "accepted": len(merged["messages_train.jsonl"]),
        "by_stage": dict(Counter(row.get("agent_stage") for row in merged["messages_train.jsonl"])),
        "public_private_ids_match": public_ids == {row["sample_id"] for row in private},
    }
    (args.output_dir / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
