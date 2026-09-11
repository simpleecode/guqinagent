#!/usr/bin/env python3
"""Build a redacted quality-filtered candidate and append redacted retries."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-raw", type=Path, required=True)
    parser.add_argument("--redacted-base", type=Path, required=True)
    parser.add_argument("--redacted-retry", type=Path, required=True)
    parser.add_argument("--raw-retry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    allowed = {row["sample_id"] for row in read(args.quality_raw / "messages_train.jsonl")}
    files = ("messages_train.jsonl", "teacher_trajectory_audit.jsonl")
    for filename in files:
        rows = []
        for row in read(args.redacted_base / filename):
            if row.get("sample_id") in allowed:
                rows.append(row)
        rows.extend(read(args.redacted_retry / filename))
        by_id = {row["sample_id"]: row for row in rows}
        write(args.output / filename, list(by_id.values()))

    # Keep intermediate plans for lineage/audit; they contain no public reasoning.
    intermediates = {row["trajectory_id"]: row for row in read(args.quality_raw / "fingering_intermediates.jsonl")}
    for row in read(args.raw_retry / "fingering_intermediates.jsonl"):
        intermediates[row["trajectory_id"]] = row
    write(args.output / "fingering_intermediates.jsonl", list(intermediates.values()))
    write(args.output / "teacher_rejected_io.jsonl", [])

    public = read(args.output / "messages_train.jsonl")
    private = read(args.output / "teacher_trajectory_audit.jsonl")
    report = {
        "schema_version": "teacher-pilot-redacted-merged-1.0",
        "accepted": len(public),
        "by_stage": dict(Counter(row.get("agent_stage") for row in public)),
        "source_phrases": len(intermediates),
        "public_private_ids_match": {row.get("sample_id") for row in public} == {row.get("sample_id") for row in private},
        "quality_source": str(args.quality_raw),
        "redacted_base": str(args.redacted_base),
        "redacted_retry": str(args.redacted_retry),
    }
    (args.output / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
