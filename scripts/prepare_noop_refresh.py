#!/usr/bin/env python3
"""Prepare current true no-op Guqinizer rows for a clean regeneration pass."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_teacher_tool_trajectories as generator  # noqa: E402


FINGER_SUFFIX = "-fingering_agent-teacher-tools"
GUQIN_SUFFIX = "-guqinization-teacher-tools"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = {row["trajectory_id"]: row for row in read_jsonl(args.source)}
    no_op_ids = set()
    for row in read_jsonl(args.final_dir / "messages_train.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id.endswith(GUQIN_SUFFIX):
            continue
        if (row.get("termination") or {}).get("kind") == "no_changes":
            no_op_ids.add(sample_id[: -len(GUQIN_SUFFIX)])

    fingering = {}
    for row in read_jsonl(args.final_dir / "teacher_trajectory_audit.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        if sample_id.endswith(FINGER_SUFFIX):
            fingering[sample_id[: -len(FINGER_SUFFIX)]] = row

    selected = []
    refresh_ids = []
    actionable_ids = []
    intermediates = []
    missing_source = []
    missing_fingering = []
    no_longer_noop = []
    for trajectory_id in sorted(no_op_ids):
        item = source.get(trajectory_id)
        audit = fingering.get(trajectory_id)
        if item is None:
            missing_source.append(trajectory_id)
            continue
        if audit is None:
            missing_fingering.append(trajectory_id)
            continue
        private = audit.get("teacher_private") or {}
        plan = private.get("accepted_plan")
        if not isinstance(plan, dict):
            missing_fingering.append(trajectory_id)
            continue
        refresh_ids.append(trajectory_id)
        intermediates.append({
            "trajectory_id": trajectory_id,
            "score_key": item["score_key"],
            "phrase_id": item["phrase_id"],
            "normalized_tuning": item["input"]["normalized_tuning"],
            "plan": plan,
            "teacher_model": private.get("teacher_model"),
            "verification": audit.get("verification") or {},
        })
        patches = generator.infer_jianzi_text_patches(
            plan, item["reference_plan"]["actions"]
        )["patches"]
        if patches:
            actionable_ids.append(trajectory_id)
            no_longer_noop.append({
                "trajectory_id": trajectory_id,
                "patch_count": len(patches),
            })
            continue
        selected.append(trajectory_id)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "fingering_intermediates.jsonl", intermediates)
    (args.output_dir / "noop_ids.txt").write_text(
        "".join(value + "\n" for value in selected), encoding="utf-8"
    )
    (args.output_dir / "actionable_ids.txt").write_text(
        "".join(value + "\n" for value in actionable_ids), encoding="utf-8"
    )
    (args.output_dir / "refresh_ids.txt").write_text(
        "".join(value + "\n" for value in refresh_ids), encoding="utf-8"
    )
    report = {
        "schema_version": "noop-refresh-selection-1.0",
        "existing_noop": len(no_op_ids),
        "selected_true_noop": len(selected),
        "refreshable": len(refresh_ids),
        "missing_source": missing_source,
        "missing_fingering": missing_fingering,
        "no_longer_noop": no_longer_noop,
        "actionable_ids": actionable_ids,
        "selected_ids": selected,
    }
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "existing_noop": len(no_op_ids),
        "selected_true_noop": len(selected),
        "refreshable": len(refresh_ids),
        "missing_source": len(missing_source),
        "missing_fingering": len(missing_fingering),
        "no_longer_noop": len(no_longer_noop),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
