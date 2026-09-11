"""Merge independently redacted JSONL shards in source order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-sample-id", action="append", default=[],
                        help="drop a failed or quarantined sample from the public merge")
    args = parser.parse_args()
    excluded = set(args.exclude_sample_id)
    source_rows = [row for row in read_rows(args.source / "messages_train.jsonl")
                   if row.get("sample_id") not in excluded]
    source_private = {row["sample_id"]: row for row in read_rows(args.source / "teacher_trajectory_audit.jsonl")}
    merged: dict[str, dict] = {}
    audit_rows: list[dict] = []
    failures: list[dict] = []
    for worker in sorted(args.workers.glob("worker_*_out")):
        report_path = worker / "reasoning_redaction_report.json"
        if not report_path.exists():
            raise SystemExit(f"missing report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        failures.extend(row for row in report.get("failures", [])
                        if row.get("sample_id") not in excluded)
        for row in read_rows(worker / "messages_train.jsonl"):
            sample_id = row["sample_id"]
            if sample_id in excluded:
                continue
            if sample_id in merged:
                raise SystemExit(f"duplicate sample_id across shards: {sample_id}")
            merged[sample_id] = row
        audit_rows.extend(row for row in read_rows(worker / "reasoning_redaction_audit.jsonl")
                          if row.get("sample_id") not in excluded)
    ordered = [row for row in source_rows if row["sample_id"] in merged]
    if len(ordered) != len(merged):
        raise SystemExit("shard output contains an ID absent from source")
    audit_keys = {(row["sample_id"], row["message_index"]) for row in audit_rows}
    if len(audit_keys) != len(audit_rows):
        raise SystemExit("duplicate reasoning audit key")
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "messages_train.jsonl", [merged[row["sample_id"]] for row in ordered])
    write_rows(args.output / "teacher_trajectory_audit.jsonl",
               [source_private[row["sample_id"]] for row in ordered if row["sample_id"] in source_private])
    write_rows(args.output / "reasoning_redaction_audit.jsonl", audit_rows)
    report = {
        "schema_version": "reasoning-redaction-1.0",
        "model": "glm-5.3",
        "requested": len(source_rows),
        "written": len(ordered),
        "redacted": len({row["sample_id"] for row in audit_rows}),
        "failed": len(failures),
        "failures": failures,
        "original_private_audit_preserved": True,
        "shards": len(list(args.workers.glob("worker_*_out"))),
    }
    (args.output / "reasoning_redaction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
