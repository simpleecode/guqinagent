"""Merge successful retry outputs into an existing redacted dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for value in values:
            out.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    args = parser.parse_args()

    source = rows(args.source / "messages_train.jsonl")
    private = {row["sample_id"]: row for row in rows(args.source / "teacher_trajectory_audit.jsonl")}
    merged = {row["sample_id"]: row for row in rows(args.base / "messages_train.jsonl")}
    audit = rows(args.base / "reasoning_redaction_audit.jsonl")
    report = json.loads((args.base / "reasoning_redaction_report.json").read_text(encoding="utf-8"))
    old_failures = {row["sample_id"]: row for row in report.get("failures", [])}
    retry_success: set[str] = set()
    retry_audit: list[dict] = []
    retry_failures: dict[str, dict] = {}
    for worker in sorted(args.retry.glob("worker_*_out")):
        worker_report = json.loads((worker / "reasoning_redaction_report.json").read_text(encoding="utf-8"))
        for row in rows(worker / "messages_train.jsonl"):
            sample_id = row["sample_id"]
            if sample_id in merged:
                raise SystemExit(f"retry duplicates existing sample_id: {sample_id}")
            merged[sample_id] = row
            retry_success.add(sample_id)
        retry_audit.extend(rows(worker / "reasoning_redaction_audit.jsonl"))
        for failure in worker_report.get("failures", []):
            retry_failures[failure["sample_id"]] = failure
    audit_keys = {(row["sample_id"], row["message_index"]) for row in audit}
    for row in retry_audit:
        key = (row["sample_id"], row["message_index"])
        if key in audit_keys:
            raise SystemExit(f"retry duplicates audit key: {key}")
        audit.append(row)
        audit_keys.add(key)
    failures = {key: value for key, value in old_failures.items() if key not in retry_success}
    failures.update(retry_failures)
    ordered = [merged[row["sample_id"]] for row in source if row["sample_id"] in merged]
    if len(ordered) != len(merged):
        raise SystemExit("merged output contains an ID absent from source")
    write(args.base / "messages_train.jsonl", ordered)
    existing_private = {row["sample_id"]: row for row in rows(args.base / "teacher_trajectory_audit.jsonl")}
    existing_private.update({sample_id: private[sample_id] for sample_id in retry_success if sample_id in private})
    write(args.base / "teacher_trajectory_audit.jsonl",
          [existing_private[row["sample_id"]] for row in ordered if row["sample_id"] in existing_private])
    write(args.base / "reasoning_redaction_audit.jsonl", audit)
    report.update({
        "requested": len(source),
        "written": len(ordered),
        "redacted": len({row["sample_id"] for row in audit}),
        "failed": len(failures),
        "failures": [failures[key] for key in sorted(failures)],
        "retry_success": len(retry_success),
    })
    (args.base / "reasoning_redaction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"written": len(ordered), "retry_success": len(retry_success), "failed": len(failures)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
