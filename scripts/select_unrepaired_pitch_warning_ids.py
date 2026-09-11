#!/usr/bin/env python3
"""Select source phrase IDs whose final pitch warning had no later edit."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_id(sample_id: str) -> str:
    return re.sub(r"-(?:fingering_agent|guqinization)-teacher-tools$", "", sample_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", nargs="+", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = {str(row["trajectory_id"]): row for row in rows(args.source)}
    selected: set[str] = set()
    selected_stages = 0
    warning_stages = 0
    for audit_path in args.audit:
        for record in rows(audit_path):
            teacher = record.get("teacher_private") or {}
            report = teacher.get("jianzi_quality_report") or {}
            final_warnings = [
                warning for warning in report.get("warnings") or []
                if warning.get("code") == "jianzi_pitch_mismatch"
                and warning.get("source_index") is not None
            ]
            if not final_warnings:
                continue
            warning_stages += 1
            tid = source_id(str(record.get("sample_id") or ""))
            item = source.get(tid) or {}
            event_for_source = {
                int(note["index"]): int(note["event_index"])
                for note in (item.get("input") or {}).get("notes_without_jianzi") or []
                if note.get("index") is not None and note.get("event_index") is not None
            }
            warning_events = {
                event_for_source.get(int(warning["source_index"]), int(warning["source_index"]))
                for warning in final_warnings
            }
            calls = teacher.get("tool_execution_log") or []
            seen_warning_events: set[int] = set()
            repaired_events: set[int] = set()
            for call in calls:
                name = call.get("name")
                audit_result = ((call.get("result") or {}).get("result") or {})
                if name == "edit_plan":
                    edited = {
                        int(row[0]) for row in (call.get("arguments") or {}).get("jianzi_rows") or []
                        if isinstance(row, list) and row and isinstance(row[0], int)
                    }
                    repaired_events.update(seen_warning_events & edited)
                    current = {
                        event_for_source.get(int(warning["source_index"]), int(warning["source_index"]))
                        for warning in audit_result.get("warnings") or []
                        if warning.get("code") == "jianzi_pitch_mismatch"
                        and warning.get("source_index") is not None
                    }
                    seen_warning_events.update(current)
            if warning_events - repaired_events:
                selected.add(tid)
                selected_stages += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{tid}\n" for tid in sorted(selected)), encoding="utf-8")
    print(json.dumps({
        "selected_source_phrases": len(selected),
        "selected_warning_stages": selected_stages,
        "final_warning_stages": warning_stages,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
