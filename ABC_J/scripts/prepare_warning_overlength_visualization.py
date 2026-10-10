#!/usr/bin/env python3
"""Materialize a full-score-context viewer input for problematic teacher phrases."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


STAGE_SUFFIX = re.compile(r"-(?:fingering_agent|guqinization)-teacher-tools$")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def phrase_id(sample_id: str) -> str:
    result = STAGE_SUFFIX.sub("", sample_id)
    if result == sample_id:
        raise ValueError(f"unrecognized teacher sample ID: {sample_id}")
    return result


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--over-8192", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    warning_phrases: set[str] = set()
    warning_codes: Counter[str] = Counter()
    audits_by_phrase: dict[str, list[dict]] = {}
    for audit in read_jsonl(args.audit):
        sample_id = str(audit["sample_id"])
        item_phrase_id = phrase_id(sample_id)
        audits_by_phrase.setdefault(item_phrase_id, []).append(audit)
        if "-guqinization-teacher-tools" not in sample_id:
            continue
        warnings = ((audit.get("teacher_private") or {})
                    .get("jianzi_quality_report") or {}).get("warnings") or []
        if warnings:
            warning_phrases.add(item_phrase_id)
            warning_codes.update(str(item.get("code") or "unknown") for item in warnings)

    over_phrases: set[str] = set()
    with args.over_8192.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sample_id = str(row.get("source_sample_id") or "")
            if sample_id:
                over_phrases.add(phrase_id(sample_id))
    selected = warning_phrases | over_phrases

    messages_by_phrase: dict[str, list[dict]] = {}
    for row in read_jsonl(args.messages):
        item_phrase_id = str((row.get("provenance") or {}).get("source_trajectory_id") or "")
        if item_phrase_id in selected:
            messages_by_phrase.setdefault(item_phrase_id, []).append(row)
    missing_messages = sorted(selected - set(messages_by_phrase))
    missing_audits = sorted(selected - set(audits_by_phrase))
    if missing_messages or missing_audits:
        raise ValueError({"missing_messages": missing_messages, "missing_audits": missing_audits})

    selected_score_keys = {
        str((rows[0].get("provenance") or {}).get("score_key") or "")
        for rows in messages_by_phrase.values()
    }
    source_rows = [
        row for row in read_jsonl(args.source)
        if str(row.get("score_key") or "") in selected_score_keys
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "messages_train.jsonl",
                (row for item_phrase_id in sorted(selected) for row in messages_by_phrase[item_phrase_id]))
    write_jsonl(args.output_dir / "teacher_trajectory_audit.jsonl",
                (row for item_phrase_id in sorted(selected) for row in audits_by_phrase[item_phrase_id]))
    write_jsonl(args.output_dir / "inferred_source_fullscorecontext.jsonl", source_rows)
    (args.output_dir / "selected_phrase_ids.txt").write_text(
        "".join(f"{value}\n" for value in sorted(selected)), encoding="utf-8")
    summary = {
        "warning_phrase_count": len(warning_phrases),
        "over_8192_phrase_count": len(over_phrases),
        "overlap_phrase_count": len(warning_phrases & over_phrases),
        "selected_phrase_count": len(selected),
        "warning_codes": dict(sorted(warning_codes.items())),
        "selected_score_count": len(selected_score_keys),
        "source_phrase_count_with_full_score_context": len(source_rows),
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
