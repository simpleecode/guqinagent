#!/usr/bin/env python3
"""Find teacher phrases whose *intermediate* harmonic warning was rest-leakage.

Unlike a final-audit-only selection, this replays every saved ``edit_plan``
preview.  A phrase is selected when a historical preview reported
``harmonic_to_same_string_attack`` but the current shared state machine does
not report that same warning for the identical preview.  Thus a teacher that
later changed a valid note solely to appease the false warning is still
retried.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE_SUFFIX = re.compile(r"-(fingering_agent|guqinization)-teacher-tools$")


def load_runtime():
    path = ROOT / "agents" / "ToolRuntime" / "runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_for_harmonic_retry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sample_parts(sample_id: str) -> tuple[str, str]:
    match = STAGE_SUFFIX.search(sample_id)
    if not match:
        raise ValueError(f"unrecognized teacher sample ID: {sample_id}")
    return sample_id[:match.start()], match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--inferred", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runtime = load_runtime()
    inferred_rows = list(rows(args.inferred))
    inferred_by_id = {str(row["trajectory_id"]): row for row in inferred_rows}
    historical = {(str(row["score_key"]), str(row["trajectory_id"])): row
                  for row in inferred_rows}

    affected: dict[str, list[dict]] = {}
    old_warning_count = 0
    removed_warning_count = 0
    scanned_edit_previews = 0
    for record_number, record in enumerate(rows(args.audit), 1):
        sample_id = str(record["sample_id"])
        phrase_id, stage = sample_parts(sample_id)
        item = inferred_by_id.get(phrase_id)
        if item is None:
            raise ValueError(f"missing inferred phrase for {sample_id}")
        item = deepcopy(item)
        item["harmonic_region_at_start"] = runtime.harmonic_region_at_phrase_start(
            item, historical
        )
        for call_number, entry in enumerate(
            (record.get("teacher_private") or {}).get("tool_execution_log") or [], 1
        ):
            if entry.get("name") != "edit_plan":
                continue
            payload = (entry.get("result") or {}).get("result") or {}
            old = [warning for warning in (payload.get("warnings") or [])
                   if warning.get("code") == "harmonic_to_same_string_attack"]
            if not old:
                continue
            preview = payload.get("preview_actions") or []
            if not preview:
                continue
            scanned_edit_previews += 1
            old_warning_count += len(old)
            by_source = {
                int(action["source_index"]): action
                for action in preview if action.get("source_index") is not None
            }
            notes = runtime.pitch_audit_notes(item, by_source, historical)
            current_indices = {
                int(note["index"])
                for note in item.get("input", {}).get("notes_without_jianzi") or []
                if note.get("index") is not None
            }
            new_indices = {
                int(warning["index"])
                for warning in runtime.AUDIT.harmonic_continuity_warnings({
                    "metadata": deepcopy(item.get("input", {}).get("metadata") or {}),
                    "open_midi": deepcopy(
                        (item.get("input", {}).get("normalized_tuning") or {}).get("open_midi")
                    ),
                    "notes": notes,
                })
                if warning.get("code") == "harmonic_to_same_string_attack"
                and warning.get("index") is not None
                and int(warning["index"]) in current_indices
            }
            removed = sorted(
                int(warning["source_index"]) for warning in old
                if warning.get("source_index") is not None
                and int(warning["source_index"]) not in new_indices
            )
            if not removed:
                continue
            removed_warning_count += len(removed)
            affected.setdefault(phrase_id, []).append({
                "sample_id": sample_id,
                "stage": stage,
                "edit_plan_call": call_number,
                "removed_warning_source_indices": removed,
                "harmonic_region_at_start": bool(item["harmonic_region_at_start"]),
            })
        if record_number % 500 == 0:
            print(f"scanned {record_number}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    phrase_ids = sorted(affected)
    (args.output_dir / "phrase_ids.txt").write_text(
        "".join(f"{value}\n" for value in phrase_ids), encoding="utf-8"
    )
    details = [
        {"trajectory_id": phrase_id, "affected_previews": affected[phrase_id]}
        for phrase_id in phrase_ids
    ]
    (args.output_dir / "details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "selection_reason": "historical harmonic same-string warning absent after shared rest-state replay",
        "inferred_phrase_count": len(inferred_rows),
        "scanned_edit_previews_with_harmonic_warning": scanned_edit_previews,
        "historical_harmonic_warning_count": old_warning_count,
        "removed_by_fixed_replay_warning_count": removed_warning_count,
        "affected_phrase_count": len(phrase_ids),
        "affected_stage_counts": dict(Counter(
            preview["stage"] for previews in affected.values() for preview in previews
        )),
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
