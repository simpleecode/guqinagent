#!/usr/bin/env python3
"""Add reasoned no-op stages and explicit assistant stops to a message set."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def stop_text(stage: str) -> str:
    return (
        "工具预览已通过，当前段基础减字填写完成。"
        if stage == "fingering_agent"
        else "工具预览已通过，当前段减字润色完成。"
    )


def add_stop(row: dict) -> bool:
    messages = row.get("messages") or []
    if (
        messages
        and messages[-1].get("role") == "tool"
        and (row.get("termination") or {}).get("kind") == "accepted_edit_plan"
    ):
        messages.append({"role": "assistant", "content": stop_text(str(row.get("agent_stage") or ""))})
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="inferred trajectory JSONL used to restore phrase/stage order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    source_order = {
        row["trajectory_id"]: index
        for index, row in enumerate(read(args.source))
    }
    stage_order = {"fingering_agent": 0, "guqinization": 1}
    public_by_id: dict[str, dict] = {}
    private_by_id: dict[str, dict] = {}
    for directory in (args.base, args.additions):
        for row in read(directory / "messages_train.jsonl"):
            public_by_id[row["sample_id"]] = row
        for row in read(directory / "teacher_trajectory_audit.jsonl"):
            private_by_id[row["sample_id"]] = row
    if set(public_by_id) != set(private_by_id):
        raise ValueError("public/private sample IDs differ after merge")

    migrated = sum(add_stop(row) for row in public_by_id.values())
    public = sorted(
        public_by_id.values(),
        key=lambda row: (
            source_order.get((row.get("provenance") or {}).get("source_trajectory_id"), 10**9),
            stage_order.get(str(row.get("agent_stage") or ""), 9),
            row["sample_id"],
        ),
    )
    private = [private_by_id[row["sample_id"]] for row in public]
    write(args.output / "messages_train.jsonl", public)
    write(args.output / "teacher_trajectory_audit.jsonl", private)
    write(args.output / "fingering_intermediates.jsonl", read(args.base / "fingering_intermediates.jsonl"))
    write(args.output / "teacher_rejected_io.jsonl", [])
    report = {
        "schema_version": "teacher-noop-terminal-merge-1.0",
        "accepted": len(public),
        "by_stage": dict(Counter(row.get("agent_stage") for row in public)),
        "reasoned_noop_added": len(read(args.additions / "messages_train.jsonl")),
        "terminal_assistant_added": migrated,
        "public_private_ids_match": True,
        "base": str(args.base),
        "additions": str(args.additions),
    }
    (args.output / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
