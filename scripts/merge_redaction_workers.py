#!/usr/bin/env python3
"""Merge redaction shards in source order, failing closed on missing rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    source = read(args.source / "messages_train.jsonl")
    source_ids = [str(row["sample_id"]) for row in source]
    public: dict[str, dict] = {}
    private: dict[str, dict] = {}
    audits: list[dict] = []
    reports: list[dict] = []
    for worker in sorted(args.worker_root.glob("worker_[0-9]*")):
        report_path = worker / "reasoning_redaction_report.json"
        if not report_path.exists():
            raise SystemExit(f"unfinished worker: {worker}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("failed"):
            raise SystemExit(f"worker reports failures: {worker}")
        reports.append(report)
        for row in read(worker / "messages_train.jsonl"):
            sample_id = str(row["sample_id"])
            if sample_id in public:
                raise SystemExit(f"duplicate public sample: {sample_id}")
            public[sample_id] = row
        for row in read(worker / "teacher_trajectory_audit.jsonl"):
            private[str(row["sample_id"])] = row
        audits.extend(read(worker / "reasoning_redaction_audit.jsonl"))
    missing = set(source_ids) - set(public)
    extra = set(public) - set(source_ids)
    if missing or extra:
        raise SystemExit(f"redaction coverage mismatch: missing={len(missing)} extra={len(extra)}")
    if set(private) != set(source_ids):
        raise SystemExit("private audit coverage mismatch")
    args.output_dir.mkdir(parents=True)
    ordered = [public[sample_id] for sample_id in source_ids]
    write(args.output_dir / "messages_train.jsonl", ordered)
    write(args.output_dir / "teacher_trajectory_audit.jsonl", [private[sample_id] for sample_id in source_ids])
    write(args.output_dir / "reasoning_redaction_audit.jsonl", audits)
    report = {
        "schema_version": "reasoning-redaction-merged-1.0",
        "source": str(args.source),
        "worker_root": str(args.worker_root),
        "requested": len(source_ids),
        "written": len(ordered),
        "failed": 0,
        "redaction_audit_rows": len(audits),
        "public_private_ids_match": {row["sample_id"] for row in ordered} == set(private),
        "workers": reports,
    }
    (args.output_dir / "reasoning_redaction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("requested", "written", "failed", "redaction_audit_rows", "public_private_ids_match")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
