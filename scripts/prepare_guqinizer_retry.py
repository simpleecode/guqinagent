"""Prepare a Guqinizer-only retry pool from a merged teacher candidate.

The merged candidate may contain a complete public message file but only a
partial ``fingering_intermediates.jsonl``.  This helper reconstructs current
Fingering intermediates from the private audit, then selects rows whose
written jianzi text differs from the annotation and whose Guqinizer sample
is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_DIR = ROOT / "ABC_J" / "scripts"
if str(GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_DIR))

import generate_teacher_tool_trajectories as generator


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing", type=Path,
                        help="optional prior retry output; successful Guqinizer IDs are excluded")
    args = parser.parse_args()

    source_rows = {row["trajectory_id"]: row for row in read_jsonl(args.input)}
    audit_rows = {}
    for row in read_jsonl(args.candidate / "teacher_trajectory_audit.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        suffix = "-fingering_agent-teacher-tools"
        if sample_id.endswith(suffix):
            audit_rows[sample_id[: -len(suffix)]] = row
    guqinizer_ids = set()
    for row in read_jsonl(args.candidate / "messages_train.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        suffix = "-guqinization-teacher-tools"
        if sample_id.endswith(suffix):
            guqinizer_ids.add(sample_id[: -len(suffix)])
    if args.existing:
        for row in read_jsonl(args.existing / "messages_train.jsonl"):
            sample_id = str(row.get("sample_id") or "")
            suffix = "-guqinization-teacher-tools"
            if sample_id.endswith(suffix):
                guqinizer_ids.add(sample_id[: -len(suffix)])

    intermediates = []
    actionable = []
    no_op = []
    for trajectory_id, audit in sorted(audit_rows.items()):
        item = source_rows.get(trajectory_id)
        if item is None or generator.is_all_empty_reference_phrase(item):
            continue
        private = audit.get("teacher_private") or {}
        plan = private.get("accepted_plan")
        if not isinstance(plan, dict):
            continue
        intermediates.append({
            "trajectory_id": trajectory_id,
            "score_key": item["score_key"],
            "phrase_id": item["phrase_id"],
            "normalized_tuning": item["input"]["normalized_tuning"],
            "plan": plan,
            "teacher_model": private.get("teacher_model"),
            "verification": audit.get("verification") or {},
        })
        if trajectory_id in guqinizer_ids:
            continue
        references = item["reference_plan"]["actions"]
        inferred = generator.infer_jianzi_text_patches(plan, references)
        targets = inferred["patches"]
        (actionable if targets else no_op).append(trajectory_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fingering_intermediates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in intermediates),
        encoding="utf-8",
    )
    (args.output_dir / "guqinizer_retry_ids.txt").write_text(
        "".join(trajectory_id + "\n" for trajectory_id in actionable),
        encoding="utf-8",
    )
    (args.output_dir / "guqinizer_noop_ids.txt").write_text(
        "".join(trajectory_id + "\n" for trajectory_id in no_op),
        encoding="utf-8",
    )
    (args.output_dir / "selection_report.json").write_text(
        json.dumps({
            "candidate": str(args.candidate),
            "fingering_intermediates": len(intermediates),
            "missing_guqinizer": len(actionable) + len(no_op),
            "actionable": len(actionable),
            "no_op": len(no_op),
            "retry_ids": actionable,
            "no_op_ids": no_op,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"intermediates": len(intermediates),
                      "missing_guqinizer": len(actionable) + len(no_op),
                      "actionable": len(actionable), "no_op": len(no_op),
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
