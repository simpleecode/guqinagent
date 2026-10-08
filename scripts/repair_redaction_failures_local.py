#!/usr/bin/env python3
"""Apply the repository's fail-closed lexical fallback to a redaction failure tail.

Operates either on a single redaction output directory (--output) or on every
``worker_*_out`` shard under a parallel run root (--workers).
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ABC_J.scripts import redact_teacher_reasoning as redact  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def repair_directory(source: dict, private: dict, output: Path) -> tuple[int, list]:
    output_rows = read_rows(output / "messages_train.jsonl")
    existing = {row["sample_id"] for row in output_rows}
    report_path = output / "reasoning_redaction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    repaired = 0
    remaining = []
    for failure in report.get("failures", []):
        sample_id = str(failure.get("sample_id") or "")
        item = source.get(sample_id)
        if not item or sample_id in existing:
            remaining.append(failure)
            continue
        private_row = private.get(sample_id, {})
        rewritten = deepcopy(item)
        audits = []
        try:
            for index, turn in redact.public_turns(item):
                summary = redact.validate_rewritten_summary(
                    redact.sanitize_fallback_summary(turn["original_summary"], private_row),
                    private_row,
                    turn["original_summary"] + "\n" + str(turn.get("tool_result") or ""),
                )
                rewritten["messages"][index]["content"] = summary
                audits.append({"sample_id": sample_id, "message_index": index,
                               "original_summary": turn["original_summary"],
                               "redacted_summary": summary})
        except Exception:
            remaining.append(failure)
            continue
        rewritten["generation_mode"] = str(item.get("generation_mode") or "") + "+reasoning_redacted"
        rewritten["reasoning_redaction"] = {"status": "redacted", "model": "lexical_fallback"}
        output_rows.append(rewritten)
        existing.add(sample_id)
        with (output / "reasoning_redaction_audit.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            for audit in audits:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
        repaired += 1
    with (output / "messages_train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["failures"] = remaining
    report["failed"] = len(remaining)
    report["written"] = len(output_rows)
    audit_path = output / "reasoning_redaction_audit.jsonl"
    report["redacted"] = len({row["sample_id"] for row in read_rows(audit_path)}) if audit_path.exists() else 0
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return repaired, remaining


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path, help="single redaction output directory")
    group.add_argument("--workers", type=Path, help="parallel run root containing worker_*_out shards")
    args = parser.parse_args()
    source = {row["sample_id"]: row for row in read_rows(args.source / "messages_train.jsonl")}
    private = {row["sample_id"]: row for row in read_rows(args.source / "teacher_trajectory_audit.jsonl")}
    targets = [args.output] if args.output else sorted(args.workers.glob("worker_*_out"))
    repaired = 0
    remaining = []
    for target in targets:
        if not (target / "reasoning_redaction_report.json").exists():
            continue
        repaired_count, rest = repair_directory(source, private, target)
        repaired += repaired_count
        remaining.extend(rest)
    print(json.dumps({"repaired": repaired, "remaining": len(remaining)}, ensure_ascii=False))
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
