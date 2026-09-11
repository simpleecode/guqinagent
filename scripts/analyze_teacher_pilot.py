#!/usr/bin/env python3
"""Re-audit a merged teacher pilot and write compact aggregate metrics."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    blank_plan_from_item, infer_jianzi_text_patches, validate_basic_fingering,
    validate_toward_quality,
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def add_pitch(total: Counter, report: dict) -> None:
    total.update(report.get("pitch_audit") or {})


def distribution(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "median": 0, "max": 0, "mean": 0}
    return {
        "min": min(values), "median": statistics.median(values), "max": max(values),
        "mean": round(statistics.mean(values), 3),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    public = rows(args.input_dir / "messages_train.jsonl")
    private = rows(args.input_dir / "teacher_trajectory_audit.jsonl")
    intermediate_rows = rows(args.input_dir / "fingering_intermediates.jsonl")
    intermediates = {row["trajectory_id"]: row["plan"] for row in intermediate_rows}
    wanted = set(intermediates)
    source_items = {}
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["trajectory_id"] in wanted:
                source_items[item["trajectory_id"]] = item
                if len(source_items) == len(wanted):
                    break

    pitch = {"fingering_agent": Counter(), "guqinization": Counter()}
    warning_codes = Counter()
    not_direct_reasons = Counter()
    pitch_mismatch_abs_cents: list[float] = []
    mismatch_details: list[dict] = []
    audit_failures = []
    distances_before: list[int] = []
    distances_after: list[int] = []
    patch_counts: list[int] = []
    tool_calls: list[int] = []
    edit_calls: list[int] = []

    for row in private:
        sample_id = row["sample_id"]
        stage = "guqinization" if "-guqinization-" in sample_id else "fingering_agent"
        trajectory_id = sample_id.split(f"-{stage}-", 1)[0]
        item = deepcopy(source_items[trajectory_id])
        teacher = row["teacher_private"]
        patches = teacher.get("accepted_patches") or []
        patch_counts.append(len(patches))
        calls = teacher.get("tool_execution_log") or []
        tool_calls.append(len(calls))
        edit_calls.append(sum(call.get("name") == "edit_plan" for call in calls))

        if stage == "fingering_agent":
            item["baseline_plan"] = blank_plan_from_item(item)
            valid, report = validate_basic_fingering(item, patches)
        else:
            item["baseline_plan"] = intermediates[trajectory_id]
            targets = infer_jianzi_text_patches(
                intermediates[trajectory_id], item["reference_plan"]["actions"]
            )["patches"]
            valid, report = validate_toward_quality(item, patches, targets)
            distances_before.append(int(report["before_distance"]))
            distances_after.append(int(report["after_distance"]))
        add_pitch(pitch[stage], report)
        for warning in report.get("warnings", []):
            warning_codes.update([warning.get("code")])
            if warning.get("code") == "pitch_not_directly_auditable":
                not_direct_reasons.update([warning.get("reason")])
            if warning.get("code") == "pitch_mismatch":
                pitch_mismatch_abs_cents.append(abs(float(warning["cents_error"])))
                source_index = int(warning["source_index"])
                action = next((value for value in report["plan"]["actions"]
                               if int(value["source_index"]) == source_index), {})
                reference = next((value for value in item["reference_plan"]["actions"]
                                  if int(value["source_index"]) == source_index), {})
                note = next((value for value in item["input"]["notes_without_jianzi"]
                             if int(value["index"]) == source_index), {})
                mismatch_details.append({
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "target_midi": warning.get("target_midi"),
                    "cents_error": warning.get("cents_error"),
                    "note": {key: note.get(key) for key in ("abc", "jianpu", "jianpu_alt")},
                    "action": {key: action.get(key) for key in
                               ("mode", "string", "hui", "string2", "mode2", "hui2",
                                "right_finger", "techniques")},
                    "reference": {key: reference.get(key) for key in
                                  ("confidence_class", "mode", "string", "hui", "string2",
                                   "mode2", "hui2", "right_finger", "techniques",
                                   "explicit_fields", "inherited_fields")},
                })
        if not valid:
            audit_failures.append({"sample_id": sample_id,
                                   "problems": report.get("problems")})

    public_text = "\n".join(json.dumps(row, ensure_ascii=False) for row in public)
    surface_scan = {
        "literal_none_left_hand": len(__import__("re").findall(r"\[无[1-7]弦", public_text)),
        "string_before_hui": len(__import__("re").findall(
            r"\[[^\]\n]{0,20}[1-7]弦[^\]\n]{0,12}(?:[一二三四五六七八九十]+徽|徽外)",
            public_text,
        )),
    }
    report = {
        "schema_version": "teacher-pilot-analysis-1.0",
        "source_phrases": len(intermediates),
        "samples": len(public),
        "by_stage": dict(Counter(row.get("agent_stage") for row in public)),
        "reaudit_valid": not audit_failures,
        "reaudit_failures": audit_failures,
        "pitch_audit": {stage: dict(counts) for stage, counts in pitch.items()},
        "warning_codes": dict(warning_codes),
        "not_directly_auditable_reasons": dict(not_direct_reasons),
        "pitch_mismatch_abs_cents": {
            **distribution([round(value) for value in pitch_mismatch_abs_cents]),
            "over_150_count": sum(value > 150 for value in pitch_mismatch_abs_cents),
        },
        "pitch_mismatch_details": mismatch_details,
        "guqin_reference_distance": {
            "before_total": sum(distances_before),
            "after_total": sum(distances_after),
            "improvement": sum(distances_before) - sum(distances_after),
        },
        "accepted_patch_count": distribution(patch_counts),
        "tool_call_count": distribution(tool_calls),
        "edit_call_count": distribution(edit_calls),
        "surface_scan": surface_scan,
        "verification_all_true": all(all(row.get("verification", {}).values()) for row in public),
    }
    (args.input_dir / "pilot_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not audit_failures and not any(surface_scan.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
