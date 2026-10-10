#!/usr/bin/env python3
"""Merge redacted Guqinizer replacements into the final public dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def stream_merge(
    base_path: Path,
    replacement_rows: dict[str, dict],
    output_path: Path,
    excluded_ids: set[str],
) -> tuple[int, set[str]]:
    """Replace selected JSONL rows without materialising a multi-GB base set."""
    written = 0
    seen: set[str] = set()
    with base_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in excluded_ids:
                continue
            row = replacement_rows.get(sample_id, row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            seen.add(sample_id)
        for sample_id in sorted(set(replacement_rows) - seen - excluded_ids):
            out.write(json.dumps(replacement_rows[sample_id], ensure_ascii=False) + "\n")
            written += 1
            seen.add(sample_id)
    return written, seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, action="append", required=True,
                        help="public replacement directory; may be passed more than once")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-trajectory", action="append", default=[])
    parser.add_argument("--exclude-sample-id", action="append", default=[],
                        help="drop an exact stage sample while retaining its paired stage")
    parser.add_argument("--allow-new-sample", action="store_true",
                        help="permit replacement samples absent from the base, for completing a missing stage")
    args = parser.parse_args()
    excluded = {value for value in args.exclude_trajectory}
    excluded_ids = set(args.exclude_sample_id)
    excluded_ids.update({
        f"{value}-{stage}-teacher-tools"
        for value in excluded for stage in ("fingering_agent", "guqinization")
    })
    base_message_path = args.base / "messages_train.jsonl"
    base_audit_path = args.base / "teacher_trajectory_audit.jsonl"
    replacements: dict[str, dict] = {}
    replacement_audits: dict[str, dict] = {}
    replacement_redaction: list[dict] = []
    for replacement_dir in args.replacement:
        for row in read(replacement_dir / "messages_train.jsonl"):
            sample_id = row["sample_id"]
            if sample_id in replacements:
                raise SystemExit(f"duplicate replacement public ID: {sample_id}")
            replacements[sample_id] = row
        for row in read(replacement_dir / "teacher_trajectory_audit.jsonl"):
            sample_id = row["sample_id"]
            if sample_id in replacement_audits:
                raise SystemExit(f"duplicate replacement private ID: {sample_id}")
            replacement_audits[sample_id] = row
        audit_file = replacement_dir / "reasoning_redaction_audit.jsonl"
        if audit_file.exists():
            replacement_redaction.extend(read(audit_file))
    if set(replacements) != set(replacement_audits):
        raise SystemExit("replacement public/private IDs differ")
    # Base teacher sets can exceed 2 GB.  Stream them below instead of doing a
    # preliminary full audit scan merely to discover unknown replacements.  A
    # replacement absent from the base is detected from the streamed IDs.
    base_ids: set[str] = set()
    unknown: set[str] = set()
    base_redaction = read(args.base / "reasoning_redaction_audit.jsonl") if (args.base / "reasoning_redaction_audit.jsonl").exists() else []
    merged_redaction = [row for row in base_redaction if row.get("sample_id") not in replacements and row.get("sample_id") not in excluded_ids]
    merged_redaction.extend(row for row in replacement_redaction if row.get("sample_id") not in excluded_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    message_count, message_ids = stream_merge(
        base_message_path, replacements, args.output / "messages_train.jsonl", excluded_ids
    )
    audit_count, audit_ids = stream_merge(
        base_audit_path, replacement_audits,
        args.output / "teacher_trajectory_audit.jsonl", excluded_ids
    )
    if message_ids != audit_ids:
        raise SystemExit("merged public/private IDs differ")
    unknown = set(replacements) - message_ids
    if unknown and not args.allow_new_sample:
        raise SystemExit(f"replacement IDs absent from base: {sorted(unknown)[:3]}")
    with (args.output / "reasoning_redaction_audit.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for row in merged_redaction:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "schema_version": "teacher-repaired-merge-1.0",
        "base_rows": message_count - len(unknown), "replacement_rows": len(replacements),
        "replaced_rows": len(set(replacements) - unknown), "added_rows": len(unknown),
        "excluded_trajectory_count": len(excluded),
        "excluded_trajectories": sorted(excluded), "output_rows": message_count,
        "excluded_sample_ids": sorted(set(args.exclude_sample_id)),
        "by_stage": {}, "public_private_ids_match": {"messages": message_count, "audits": audit_count},
        "redaction_audit_rows": len(merged_redaction),
    }
    with (args.output / "messages_train.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            stage = row.get("agent_stage")
            report["by_stage"][stage] = report["by_stage"].get(stage, 0) + 1
    (args.output / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
