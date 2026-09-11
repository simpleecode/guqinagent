#!/usr/bin/env python3
"""Select trajectories whose accepted plans contain bare 撮+Chinese-digit shorthand."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_teacher_tool_trajectories as generator  # noqa: E402

BAD_CUO = re.compile(r"^撮[一二三四五六七]{2,3}$")
FINGER_SUFFIX = "-fingering_agent-teacher-tools"
GUQIN_SUFFIX = "-guqinization-teacher-tools"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bad_texts(row: dict) -> list[str]:
    plan = (row.get("teacher_private") or {}).get("accepted_plan") or {}
    return [
        str(action.get("jianzi_text"))
        for action in plan.get("actions") or []
        if BAD_CUO.fullmatch(str(action.get("jianzi_text") or ""))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    finger: dict[str, dict] = {}
    guqin: dict[str, dict] = {}
    for row in read_jsonl(args.final_dir / "teacher_trajectory_audit.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        if sample_id.endswith(FINGER_SUFFIX):
            finger[sample_id.removesuffix(FINGER_SUFFIX)] = row
        elif sample_id.endswith(GUQIN_SUFFIX):
            guqin[sample_id.removesuffix(GUQIN_SUFFIX)] = row

    base_bad = {key: bad_texts(row) for key, row in finger.items() if bad_texts(row)}
    guqin_bad = {key: bad_texts(row) for key, row in guqin.items() if bad_texts(row)}
    both_stage_ids = sorted(base_bad)
    guqin_only_ids = sorted(set(guqin_bad) - set(base_bad))
    source = {row["trajectory_id"]: row for row in read_jsonl(args.source)}
    all_empty_ids = sorted(
        trajectory_id for trajectory_id in set(base_bad) | set(guqin_bad)
        if trajectory_id not in source or generator.is_all_empty_reference_phrase(source[trajectory_id])
    )
    eligible_both_ids = sorted(set(both_stage_ids) - set(all_empty_ids))
    eligible_guqin_only_ids = sorted(set(guqin_only_ids) - set(all_empty_ids))

    intermediates = []
    for trajectory_id in guqin_only_ids:
        row = finger.get(trajectory_id)
        private = (row or {}).get("teacher_private") or {}
        plan = private.get("accepted_plan")
        if not isinstance(plan, dict):
            raise ValueError(f"missing Fingering accepted_plan: {trajectory_id}")
        intermediates.append({
            "trajectory_id": trajectory_id,
            "score_key": private.get("score_key") or trajectory_id.rsplit("-", 1)[0],
            "phrase_id": private.get("phrase_id") or trajectory_id.rsplit("-", 1)[-1],
            "normalized_tuning": private.get("normalized_tuning"),
            "plan": plan,
            "teacher_model": private.get("teacher_model"),
            "verification": row.get("verification") or {},
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "rerun_both_ids.txt").write_text(
        "".join(value + "\n" for value in both_stage_ids), encoding="utf-8"
    )
    (args.output_dir / "rerun_guqinizer_only_ids.txt").write_text(
        "".join(value + "\n" for value in guqin_only_ids), encoding="utf-8"
    )
    (args.output_dir / "eligible_both_ids.txt").write_text(
        "".join(value + "\n" for value in eligible_both_ids), encoding="utf-8"
    )
    (args.output_dir / "eligible_guqinizer_only_ids.txt").write_text(
        "".join(value + "\n" for value in eligible_guqin_only_ids), encoding="utf-8"
    )
    (args.output_dir / "remove_all_empty_ids.txt").write_text(
        "".join(value + "\n" for value in all_empty_ids), encoding="utf-8"
    )
    write_jsonl(args.output_dir / "guqinizer_intermediates.jsonl", intermediates)
    write_jsonl(args.output_dir / "fingering_intermediates.jsonl", intermediates)
    report = {
        "schema_version": "pure-cuo-refresh-selection-1.0",
        "pattern": BAD_CUO.pattern,
        "affected_phrases": len(set(base_bad) | set(guqin_bad)),
        "rerun_both": len(both_stage_ids),
        "rerun_guqinizer_only": len(guqin_only_ids),
        "eligible_both": len(eligible_both_ids),
        "eligible_guqinizer_only": len(eligible_guqin_only_ids),
        "remove_all_empty": len(all_empty_ids),
        "base_bad": base_bad,
        "guqinizer_bad": guqin_bad,
    }
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "affected_phrases", "rerun_both", "rerun_guqinizer_only",
        "eligible_both", "eligible_guqinizer_only", "remove_all_empty"
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
