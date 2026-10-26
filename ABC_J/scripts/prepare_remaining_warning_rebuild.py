"""Audit current accepted Guqinizer plans and select non-annotation-aligned warnings.

The live validator is replayed with the actual Fingering intermediate as the
Guqinizer baseline.  Identical notation that triggers the same warning in the
annotation is reported but not automatically sent to the teacher again.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ToolRuntime.runtime import (
    canonical_jianzi_text,
    harmonic_region_at_phrase_start,
    validate_jianzi_only,
)

BASE = ROOT / "ABC_J/agent_training/two_stage_audited"
SOURCE = BASE / "pitch_eligible_train_test/inferred_trajectories_train.jsonl"
AUDIT = BASE / "teacher_train_final/teacher_trajectory_audit.jsonl"
OUT = BASE / ".remaining_warning_rebuild"
EXCLUDED_CODES = {"missing_independent_attack"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def notation_by_index(plan: dict) -> dict[int, str]:
    return {
        int(action["source_index"]): str(
            action.get("jianzi_text")
            if action.get("jianzi_text") is not None else action.get("text") or ""
        )
        for action in plan.get("actions") or []
    }


def reference_plan(item: dict) -> dict:
    plan = deepcopy(item["reference_plan"])
    for action in plan["actions"]:
        if action.get("jianzi_text") is None:
            action["jianzi_text"] = action.get("text") or ""
    return plan


def main() -> None:
    if (OUT / "regenerated_raw").exists():
        raise SystemExit("Generation already started; refusing to alter its selection")
    source = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    history = {(row["score_key"], row["phrase_id"]): row for row in source}
    audits = {}
    for line in AUDIT.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row["sample_id"]
        if sample_id in audits:
            raise ValueError(f"Duplicate audit row: {sample_id}")
        audits[sample_id] = row["teacher_private"]
    if len(source) != 3976 or len(audits) != 7952:
        raise ValueError("Unexpected source or audit size")

    all_codes = Counter()
    all_phrases = defaultdict(set)
    selected_codes = Counter()
    selected_phrases = defaultdict(set)
    cases = {}
    annotation_aligned = Counter()
    for number, original in enumerate(source, 1):
        phrase_id = original["trajectory_id"]
        fingering = audits[f"{phrase_id}-fingering_agent-teacher-tools"]
        guqinizer = audits[f"{phrase_id}-guqinization-teacher-tools"]
        item = dict(original)
        item["baseline_plan"] = fingering["accepted_plan"]
        item["harmonic_region_at_start"] = harmonic_region_at_phrase_start(item, history)
        _, final_report = validate_jianzi_only(
            item, guqinizer["accepted_patches"], toward_reference=True,
            historical=history,
        )
        final_text = notation_by_index(final_report["plan"])

        annotation_item = dict(item)
        annotation_item["baseline_plan"] = reference_plan(item)
        _, annotation_report = validate_jianzi_only(
            annotation_item, [], require_complete=False, historical=history,
        )
        annotation_text = notation_by_index(annotation_item["baseline_plan"])
        annotation_codes = {
            (int(warning["source_index"]), warning["code"])
            for warning in annotation_report.get("warnings") or []
            if warning.get("source_index") is not None
        }

        selected = []
        for warning in final_report.get("warnings") or []:
            code = str(warning.get("code") or "unknown")
            if code in EXCLUDED_CODES or warning.get("source_index") is None:
                continue
            index = int(warning["source_index"])
            all_codes[code] += 1
            all_phrases[code].add(phrase_id)
            final_notation = final_text.get(index, "")
            reference_notation = annotation_text.get(index, "")
            same_annotation_warning = (index, code) in annotation_codes
            same_notation = (
                canonical_jianzi_text(final_notation)
                == canonical_jianzi_text(reference_notation)
            )
            if same_annotation_warning and same_notation:
                annotation_aligned[code] += 1
                continue
            selected_codes[code] += 1
            selected_phrases[code].add(phrase_id)
            selected.append({
                "source_index": index,
                "code": code,
                "final_text": final_notation,
                "annotation_text": reference_notation,
                "annotation_warns_too": same_annotation_warning,
                "warning": warning,
            })
        if selected:
            cases[phrase_id] = selected
        if number % 500 == 0:
            print(f"audited {number}/{len(source)}", flush=True)

    selected_ids = sorted(cases)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "remaining-warning-rebuild-1.0",
        "basis": "current final Guqinizer replay on accepted Fingering baseline",
        "source_sha256": sha256(SOURCE),
        "audit_sha256": sha256(AUDIT),
        "source_phrases": len(source),
        "all_non_missing_warning_events": sum(all_codes.values()),
        "all_non_missing_warning_phrases": len(set().union(*all_phrases.values())),
        "all_codes": dict(sorted(all_codes.items())),
        "all_phrases_by_code": {code: len(ids) for code, ids in sorted(all_phrases.items())},
        "annotation_aligned_events": sum(annotation_aligned.values()),
        "annotation_aligned_codes": dict(sorted(annotation_aligned.items())),
        "selected_phrases": len(selected_ids),
        "selected_events": sum(selected_codes.values()),
        "selected_codes": dict(sorted(selected_codes.items())),
        "selected_phrases_by_code": {
            code: len(ids) for code, ids in sorted(selected_phrases.items())
        },
        "selected_ids": selected_ids,
        "cases": cases,
    }
    (OUT / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "trajectory_ids.txt").write_text(
        "".join(phrase_id + "\n" for phrase_id in selected_ids), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"selected_ids", "cases"}},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
