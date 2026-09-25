#!/usr/bin/env python3
"""Compare phrase-start harmonic marker replay with pitch-auditor state replay.

This is a diagnostic-only script.  It deliberately reuses the same helpers as
teacher generation, so a disagreement identifies an actual contract mismatch
between the prompt-facing state and edit_plan's pitch audit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "ABC_J/scripts/generate_teacher_tool_trajectories.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("teacher_generator", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def action_text(action: dict) -> str:
    return str(action.get("text") or action.get("jianzi_text") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generator = load_generator()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    historical = {(row["score_key"], row["phrase_id"]): row for row in rows}
    findings: list[dict] = []

    for item in rows:
        marker_state = generator.harmonic_region_at_phrase_start(item, historical)
        current_indices = {
            int(note["index"])
            for note in item["input"].get("notes_without_jianzi") or []
            if note.get("index") is not None
        }
        current_actions = {
            int(action["source_index"]): action
            for action in item.get("reference_plan", {}).get("actions") or []
            if action.get("source_index") is not None
        }
        notes = generator.pitch_audit_notes(item, current_actions, historical=historical)
        context = generator.AUDIT.new_context()
        open_midi = item["input"]["normalized_tuning"]["open_midi"]
        replay_log: list[dict] = []
        for note in notes:
            index = note.get("index")
            if index is not None and int(index) in current_indices:
                break
            text = str(note.get("jianzi") or "")
            before = bool(context.get("harmonic_scope") or context.get("harmonic"))
            _pitches, reason = generator.AUDIT.parse_jianzi(text, open_midi, context)
            after = bool(context.get("harmonic_scope") or context.get("harmonic"))
            if "泛起" in text or "泛止" in text or before != after:
                replay_log.append({
                    "source_index": index,
                    "jianzi": text,
                    "parse_reason": reason,
                    "harmonic_before": before,
                    "harmonic_after": after,
                })
        audit_state = bool(context.get("harmonic_scope") or context.get("harmonic"))
        if marker_state != audit_state:
            findings.append({
                "trajectory_id": item["trajectory_id"],
                "score_key": item["score_key"],
                "phrase_id": item["phrase_id"],
                "marker_harmonic_at_start": marker_state,
                "audit_harmonic_at_start": audit_state,
                "audit_harmonic_hui": context.get("harmonic_hui"),
                "control_replay": replay_log,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "input": str(args.input), "total_phrases": len(rows),
        "disagreement_count": len(findings), "findings": findings,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_phrases": len(rows), "disagreement_count": len(findings),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
