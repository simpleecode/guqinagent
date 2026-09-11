#!/usr/bin/env python3
"""Audit explicit blank annotation targets in an existing two-stage corpus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_teacher_tool_trajectories as generator


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def source_id(sample_id: str, stage: str) -> str | None:
    suffix = f"-{stage}-teacher-tools"
    return sample_id[:-len(suffix)] if sample_id.endswith(suffix) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="inferred trajectory source JSONL")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="directory containing public/private two-stage rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = {row["trajectory_id"]: row for row in read_jsonl(args.input)}
    current_ids: set[str] = set()
    no_op_ids: set[str] = set()
    for row in read_jsonl(args.candidate / "messages_train.jsonl"):
        for stage in ("fingering_agent", "guqinization"):
            trajectory_id = source_id(str(row.get("sample_id") or ""), stage)
            if trajectory_id is not None:
                current_ids.add(trajectory_id)
                if stage == "guqinization":
                    has_tool_call = any(
                        message.get("role") == "assistant" and message.get("tool_calls")
                        for message in row.get("messages", [])
                    )
                    if ((row.get("termination") or {}).get("kind") == "no_changes"
                            or not has_tool_call):
                        no_op_ids.add(trajectory_id)

    plans: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(args.candidate / "teacher_trajectory_audit.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        for stage in ("fingering_agent", "guqinization"):
            trajectory_id = source_id(sample_id, stage)
            if trajectory_id is not None:
                plan = (row.get("teacher_private") or {}).get("accepted_plan")
                if isinstance(plan, dict):
                    plans[(trajectory_id, stage)] = plan

    all_empty_ids = sorted(
        trajectory_id for trajectory_id in current_ids
        if trajectory_id in sources
        and generator.is_all_empty_reference_phrase(sources[trajectory_id])
    )
    retained_ids = sorted(current_ids - set(all_empty_ids))
    repeat_omission_ids = sorted(
        trajectory_id for trajectory_id in retained_ids
        if trajectory_id in sources and any(
            note.get("notation_omitted")
            for note in sources[trajectory_id].get("input", {}).get(
                "notes_without_jianzi", []
            )
        )
    )
    base_blank_mismatch: list[str] = []
    final_blank_mismatch: list[str] = []
    final_any_mismatch: list[str] = []
    details: dict[str, dict] = {}

    for trajectory_id in retained_ids:
        item = sources.get(trajectory_id)
        if item is None:
            continue
        references = item.get("reference_plan", {}).get("actions", [])
        record: dict[str, list[int]] = {}
        for stage, destination in (
            ("fingering_agent", base_blank_mismatch),
            ("guqinization", final_blank_mismatch),
        ):
            plan = plans.get((trajectory_id, stage))
            if plan is None:
                continue
            patches = generator.infer_jianzi_text_patches(plan, references)["patches"]
            blank_indices = [
                int(patch["source_index"]) for patch in patches
                if patch.get("after", {}).get("jianzi_text") == ""
            ]
            if blank_indices:
                destination.append(trajectory_id)
                record[f"{stage}_blank_indices"] = blank_indices
            if stage == "guqinization" and patches:
                final_any_mismatch.append(trajectory_id)
                record["guqinization_all_diff_indices"] = [
                    int(patch["source_index"]) for patch in patches
                ]
        if record:
            details[trajectory_id] = record

    report = {
        "schema_version": "blank-reference-audit-1.0",
        "source": str(args.input),
        "candidate": str(args.candidate),
        "current_phrases": len(current_ids),
        "all_empty_remove": len(all_empty_ids),
        "all_empty_ids": all_empty_ids,
        "retained_phrases": len(retained_ids),
        "repeat_omission_private_prompt_affected": len(repeat_omission_ids),
        "repeat_omission_ids": repeat_omission_ids,
        "base_has_text_where_reference_blank": len(base_blank_mismatch),
        "base_blank_mismatch_ids": base_blank_mismatch,
        "retained_old_no_op": len(no_op_ids - set(all_empty_ids)),
        "newly_actionable_old_no_op_ids": sorted(
            (no_op_ids - set(all_empty_ids)) & set(base_blank_mismatch)
        ),
        "guqinizer_still_has_text_where_reference_blank": len(final_blank_mismatch),
        "guqinizer_blank_mismatch_ids": final_blank_mismatch,
        "guqinizer_any_surface_difference": len(final_any_mismatch),
        "guqinizer_any_mismatch_ids": final_any_mismatch,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "current_phrases", "all_empty_remove", "retained_phrases",
        "base_has_text_where_reference_blank",
        "guqinizer_still_has_text_where_reference_blank",
        "guqinizer_any_surface_difference",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
