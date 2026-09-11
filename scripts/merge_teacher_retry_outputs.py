#!/usr/bin/env python3
"""Merge a teacher pilot and retry directories without duplicate samples."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


KEYS = {
    "messages_train.jsonl": "sample_id",
    "teacher_trajectory_audit.jsonl": "sample_id",
    "fingering_intermediates.jsonl": "trajectory_id",
}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-sample-id", action="append", default=[],
                        help="drop a known-invalid accepted sample during final curation")
    parser.add_argument("--exclude-sample-id-file", type=Path,
                        help="newline-delimited sample IDs to exclude")
    args = parser.parse_args()
    if args.exclude_sample_id_file:
        args.exclude_sample_id.extend(
            line.strip() for line in args.exclude_sample_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    args.exclude_sample_id = list(dict.fromkeys(args.exclude_sample_id))
    args.output_dir.mkdir(parents=True, exist_ok=False)

    merged: dict[str, list[dict]] = {}
    for filename, key in KEYS.items():
        by_key: dict[str, dict] = {}
        for directory in args.input_dir:
            for row in read_rows(directory / filename):
                if filename != "fingering_intermediates.jsonl" and row[key] in args.exclude_sample_id:
                    continue
                by_key[str(row[key])] = row  # later retry wins
        rows = list(by_key.values())
        merged[filename] = rows
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rejected = []
    for directory in args.input_dir:
        rejected.extend(read_rows(directory / "teacher_rejected_io.jsonl"))
    with (args.output_dir / "teacher_rejected_io.jsonl").open(
            "w", encoding="utf-8", newline="\n") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    public = merged["messages_train.jsonl"]
    public_ids = {row["sample_id"] for row in public}
    # Retry directories can contain accepted audits/intermediates for rows
    # that were later quarantined (for example, high-confidence pitch
    # mismatches).  Keep private and intermediate streams aligned with the
    # actual public sample set instead of silently producing orphan records.
    private = [row for row in merged["teacher_trajectory_audit.jsonl"]
               if row.get("sample_id") in public_ids]
    allowed_trajectory_ids = {
        str((row.get("provenance") or {}).get("source_trajectory_id"))
        for row in public
        if (row.get("provenance") or {}).get("source_trajectory_id")
    }
    intermediates = [row for row in merged["fingering_intermediates.jsonl"]
                     if str(row.get("trajectory_id")) in allowed_trajectory_ids]
    merged["teacher_trajectory_audit.jsonl"] = private
    merged["fingering_intermediates.jsonl"] = intermediates
    for filename, rows in (("teacher_trajectory_audit.jsonl", private),
                           ("fingering_intermediates.jsonl", intermediates)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    public_ids = {row["sample_id"] for row in public}
    private_ids = {row["sample_id"] for row in private}
    stages = Counter(row.get("agent_stage") for row in public)
    report = {
        "schema_version": "teacher-pilot-merged-1.0",
        "source_phrases": len(intermediates),
        "accepted": len(public),
        "by_stage": dict(stages),
        "public_private_ids_match": public_ids == private_ids,
        "historical_rejected_attempts": len(rejected),
        "excluded_sample_ids": list(args.exclude_sample_id),
        "input_dirs": [str(path) for path in args.input_dir],
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
