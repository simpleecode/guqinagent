#!/usr/bin/env python3
"""Fail-closed filter for teacher trajectory outputs.

Only trajectories whose phrase id occurs in ``--allowed-ids`` are copied.
Later ``--input-dir`` values replace earlier rows with the same sample id, so
targeted reruns can safely override a broad initial generation.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


STAGE_SUFFIX = re.compile(
    r"-(?:fingering_agent|guqinization|single_stage)-teacher-tools$"
)
FILES = {
    "messages_train.jsonl": "sample_id",
    "teacher_trajectory_audit.jsonl": "sample_id",
    "fingering_intermediates.jsonl": "trajectory_id",
}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def phrase_id(value: object) -> str:
    return STAGE_SUFFIX.sub("", str(value or ""))


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowed-ids", type=Path, required=True)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    allowed = {line.strip() for line in args.allowed_ids.open(encoding="utf-8") if line.strip()}
    if not allowed:
        raise SystemExit("allowed phrase-id list is empty")
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    report: dict[str, object] = {
        "schema_version": "teacher-output-whitelist-filter-1.0",
        "allowed_ids": str(args.allowed_ids),
        "allowed_phrase_count": len(allowed),
        "input_dirs": [str(p) for p in args.input_dir],
        "files": {},
    }
    public_ids: set[str] = set()
    for filename, key in FILES.items():
        merged: dict[str, dict] = {}
        seen_phrases: set[str] = set()
        rejected_phrases: set[str] = set()
        for directory in args.input_dir:
            for row in read_rows(directory / filename):
                pid = phrase_id(row.get(key))
                if pid in allowed:
                    merged[str(row[key])] = row
                    seen_phrases.add(pid)
                else:
                    rejected_phrases.add(pid)
        rows = list(merged.values())
        if filename == "teacher_trajectory_audit.jsonl":
            rows = [row for row in rows if row.get("sample_id") in public_ids]
        write_rows(args.output_dir / filename, rows)
        if filename == "messages_train.jsonl":
            public_ids = {str(row["sample_id"]) for row in rows}
        report["files"][filename] = {
            "accepted_rows": len(rows),
            "accepted_phrase_count": len(seen_phrases),
            "rejected_phrase_count": len(rejected_phrases),
            "by_stage": dict(Counter(row.get("agent_stage") for row in rows)),
        }
    report["public_private_ids_match"] = public_ids == {
        str(row["sample_id"]) for row in read_rows(args.output_dir / "teacher_trajectory_audit.jsonl")
    }
    (args.output_dir / "whitelist_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
