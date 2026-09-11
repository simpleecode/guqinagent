"""Repair a small redaction failure tail with the repository's lexical fallback."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import redact_teacher_reasoning as redact


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workers", type=Path, required=True)
    args = parser.parse_args()
    source = {row["sample_id"]: row for row in read_rows(args.source / "messages_train.jsonl")}
    private = {row["sample_id"]: row for row in read_rows(args.source / "teacher_trajectory_audit.jsonl")}
    repaired = 0
    for worker in sorted(args.workers.glob("worker_*_out")):
        report_path = worker / "reasoning_redaction_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        failures = list(report.get("failures") or [])
        if not failures:
            continue
        existing = {row["sample_id"] for row in read_rows(worker / "messages_train.jsonl")}
        remaining = []
        for failure in failures:
            sample_id = str(failure.get("sample_id") or "")
            item = source.get(sample_id)
            if item is None or sample_id in existing:
                remaining.append(failure)
                continue
            private_row = private.get(sample_id, {})
            rewritten = deepcopy(item)
            audit_rows = []
            try:
                for index, turn in redact.public_turns(item):
                    summary = redact.validate_rewritten_summary(
                        redact.sanitize_fallback_summary(turn["original_summary"], private_row),
                        private_row,
                        turn["original_summary"] + "\n" + str(turn.get("tool_result") or ""),
                    )
                    rewritten["messages"][index]["content"] = summary
                    audit_rows.append({
                        "sample_id": sample_id,
                        "message_index": index,
                        "original_summary": turn["original_summary"],
                        "redacted_summary": summary,
                    })
                rewritten["generation_mode"] = str(item.get("generation_mode") or "") + "+reasoning_redacted"
                rewritten["reasoning_redaction"] = {"status": "redacted", "model": "lexical_fallback"}
            except Exception:
                remaining.append(failure)
                continue
            append_row(worker / "messages_train.jsonl", rewritten)
            for audit in audit_rows:
                append_row(worker / "reasoning_redaction_audit.jsonl", audit)
            existing.add(sample_id)
            repaired += 1
        report["failures"] = remaining
        report["failed"] = len(remaining)
        report["written"] = len(existing)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"repaired": repaired}, ensure_ascii=False))
    return 0 if repaired else 1


if __name__ == "__main__":
    raise SystemExit(main())
