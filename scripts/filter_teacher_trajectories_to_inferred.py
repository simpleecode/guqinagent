#!/usr/bin/env python3
"""Restrict teacher outputs to the accepted inferred trajectory IDs."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE_RE = re.compile(r"^(.*)-(fingering_agent|guqinization)-teacher-tools$")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--inferred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    allowed: set[str] = set()
    for split in ("train", "validation", "test"):
        path = args.inferred_dir / f"inferred_trajectories_{split}.jsonl"
        allowed.update(item["trajectory_id"] for item in read_jsonl(path))
    args.output_dir.mkdir(parents=True)
    report: dict = {"allowed_inferred_trajectory_ids": len(allowed), "files": {}}
    for filename, key in (("messages_train.jsonl", "sample_id"), ("teacher_trajectory_audit.jsonl", "sample_id"), ("fingering_intermediates.jsonl", "trajectory_id")):
        rows = read_jsonl(args.teacher_dir / filename)
        kept: list[dict] = []
        dropped = 0
        for row in rows:
            value = str(row.get(key, ""))
            if key == "sample_id":
                match = STAGE_RE.match(value)
                trajectory_id = match.group(1) if match else ""
            else:
                trajectory_id = value
            if trajectory_id in allowed:
                kept.append(row)
            else:
                dropped += 1
        write_jsonl(args.output_dir / filename, kept)
        report["files"][filename] = {"input": len(rows), "kept": len(kept), "dropped": dropped}

    # Rejected attempts are audit history; retain them but report separately.
    rejected = read_jsonl(args.teacher_dir / "teacher_rejected_io.jsonl")
    write_jsonl(args.output_dir / "teacher_rejected_io.jsonl", rejected)
    report["rejected_attempts_retained"] = len(rejected)
    public = read_jsonl(args.output_dir / "messages_train.jsonl")
    private = read_jsonl(args.output_dir / "teacher_trajectory_audit.jsonl")
    public_ids = {row.get("sample_id") for row in public}
    private_ids = {row.get("sample_id") for row in private}
    report["public_private_ids_match"] = public_ids == private_ids
    report["by_stage"] = dict(Counter(row.get("agent_stage") for row in public))
    (args.output_dir / "generation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
