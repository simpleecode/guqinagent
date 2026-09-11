#!/usr/bin/env python3
"""Merge redacted Guqinizer replacements into the final public dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-trajectory", action="append", default=[])
    args = parser.parse_args()
    excluded = {value for value in args.exclude_trajectory}
    excluded_ids = {
        f"{value}-{stage}-teacher-tools"
        for value in excluded for stage in ("fingering_agent", "guqinization")
    }
    base_messages = read(args.base / "messages_train.jsonl")
    base_audits = {row["sample_id"]: row for row in read(args.base / "teacher_trajectory_audit.jsonl")}
    replacements = {row["sample_id"]: row for row in read(args.replacement / "messages_train.jsonl")}
    replacement_audits = {row["sample_id"]: row for row in read(args.replacement / "teacher_trajectory_audit.jsonl")}
    if set(replacements) != set(replacement_audits):
        raise SystemExit("replacement public/private IDs differ")
    unknown = set(replacements) - set(base_audits)
    if unknown:
        raise SystemExit(f"replacement IDs absent from base: {sorted(unknown)[:3]}")
    merged_messages = []
    for row in base_messages:
        sid = row["sample_id"]
        if sid in excluded_ids:
            continue
        merged_messages.append(replacements.get(sid, row))
    merged_audits = []
    for row in read(args.base / "teacher_trajectory_audit.jsonl"):
        sid = row["sample_id"]
        if sid in excluded_ids:
            continue
        merged_audits.append(replacement_audits.get(sid, row))
    base_redaction = read(args.base / "reasoning_redaction_audit.jsonl") if (args.base / "reasoning_redaction_audit.jsonl").exists() else []
    repl_redaction = read(args.replacement / "reasoning_redaction_audit.jsonl") if (args.replacement / "reasoning_redaction_audit.jsonl").exists() else []
    merged_redaction = [row for row in base_redaction if row.get("sample_id") not in replacements and row.get("sample_id") not in excluded_ids]
    merged_redaction.extend(row for row in repl_redaction if row.get("sample_id") not in excluded_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "messages_train.jsonl", merged_messages)
    write(args.output / "teacher_trajectory_audit.jsonl", merged_audits)
    write(args.output / "reasoning_redaction_audit.jsonl", merged_redaction)
    report = {
        "schema_version": "teacher-repaired-merge-1.0",
        "base_rows": len(base_messages), "replacement_rows": len(replacements),
        "replaced_rows": len(replacements), "excluded_trajectory_count": len(excluded),
        "excluded_trajectories": sorted(excluded), "output_rows": len(merged_messages),
        "by_stage": {}, "public_private_ids_match": {"messages": len(merged_messages), "audits": len(merged_audits)},
        "redaction_audit_rows": len(merged_redaction),
    }
    for row in merged_messages:
        report["by_stage"][row.get("agent_stage")] = report["by_stage"].get(row.get("agent_stage"), 0) + 1
    (args.output / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
