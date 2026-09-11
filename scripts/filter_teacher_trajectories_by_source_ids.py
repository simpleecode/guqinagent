#!/usr/bin/env python3
"""Create a traceable teacher-output dataset without selected source phrases."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


STAGE_RE = re.compile(r"^(.*)-(?:fingering_agent|guqinization)-teacher-tools$")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, values: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in values) + ("\n" if values else ""), encoding="utf-8")


def source_id(row: dict) -> str:
    value = str(row.get("sample_id") or "")
    match = STAGE_RE.match(value)
    return match.group(1) if match else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--drop-report", type=Path,
                        help="optional report listing exact trajectory IDs to drop")
    parser.add_argument("--drop-score-key", action="append", default=[],
                        help="drop every phrase whose trajectory ID starts with SCORE_KEY-")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    drop_report = []
    if args.drop_report:
        drop_report = read(args.drop_report) if args.drop_report.suffix == ".jsonl" else json.loads(args.drop_report.read_text(encoding="utf-8"))
    drop = {item["trajectory_id"] for item in drop_report if item.get("all_actual_empty") and not item.get("prior_repeat_markers")}
    drop_score_keys = {str(value).strip() for value in args.drop_score_key if str(value).strip()}

    def should_drop(trajectory_id: object) -> bool:
        value = str(trajectory_id or "")
        return value in drop or any(value.startswith(f"{score_key}-") for score_key in drop_score_keys)

    args.output_dir.mkdir(parents=True)
    report: dict = {"schema_version": "teacher-filter-source-1.0",
                    "dropped_source_trajectory_ids": sorted(drop),
                    "dropped_score_keys": sorted(drop_score_keys), "files": {}}
    for filename, key in (("messages_train.jsonl", "sample_id"), ("teacher_trajectory_audit.jsonl", "sample_id"), ("fingering_intermediates.jsonl", "trajectory_id"), ("teacher_rejected_io.jsonl", "sample_id")):
        values = read(args.input_dir / filename)
        kept = [value for value in values
                if not should_drop(value.get(key) if key == "trajectory_id" else source_id(value))]
        write(args.output_dir / filename, kept)
        report["files"][filename] = {"input": len(values), "kept": len(kept), "dropped": len(values) - len(kept)}
    public = read(args.output_dir / "messages_train.jsonl")
    private = read(args.output_dir / "teacher_trajectory_audit.jsonl")
    report["accepted"] = len(public)
    report["by_stage"] = dict(Counter(row.get("agent_stage") for row in public))
    report["public_private_ids_match"] = {row.get("sample_id") for row in public} == {row.get("sample_id") for row in private}
    (args.output_dir / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
