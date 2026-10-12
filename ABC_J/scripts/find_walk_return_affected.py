#!/usr/bin/env python3
"""Stream every teacher tool round; select false walk warnings after 复/进/退.

Uses full preview_actions (not just edited rows or the final accepted plan).
Only selects historical warnings that disappear under the repaired shared
parser and whose preceding movement/control can explain the stale hui.
"""
import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("pitch_audit", ROOT / "scripts/audit_jianpu_jianzi_pitch.py")
pitch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pitch)
CONTROL = re.compile(r"(?:复|(?:急|缓|少)?(?:进|退|上|下))$")
CODES = {"walk_endpoint_same_as_start", "walk_direction_reversed", "walk_span_too_large"}


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def scan(source, audit, output):
    # Keep only small replay inputs rather than multi-GB private IO traces.
    inputs = {r["trajectory_id"]: {key: r["input"][key] for key in
              ("metadata", "normalized_tuning", "notes_without_jianzi")} for r in rows(source)}
    counts = Counter()
    cases = []
    for record in rows(audit):
        counts["stage_trajectories_scanned"] += 1
        sample_id = record["sample_id"]
        phrase_id = re.split(r"-(?:fingering_agent|guqinization|single_stage)-", sample_id)[0]
        item = inputs.get(phrase_id)
        if item is None:
            counts["not_in_current_train_input"] += 1
            continue
        for round_number, call in enumerate(record.get("teacher_private", {}).get("tool_execution_log", []), 1):
            if call.get("name") != "edit_plan":
                continue
            result = (call.get("result") or {}).get("result") or {}
            warnings = [w for w in result.get("warnings", []) if w.get("code") in CODES]
            counts["historical_motion_warning_occurrences"] += len(warnings)
            if not warnings:
                continue
            actions = result.get("preview_actions") or []
            by_index = {a["source_index"]: a for a in actions}
            notes = [{**n, "jianzi": by_index.get(n["index"], {}).get("jianzi_text", "")}
                     for n in item["notes_without_jianzi"]]
            data = {"metadata": item["metadata"],
                    "open_midi": item["normalized_tuning"]["open_midi"], "notes": notes}
            fixed = {(w["index"], w["code"]) for w in pitch.walk_motion_warnings(data)}
            # Compare span warnings using their ORIGINAL threshold; otherwise
            # the independent 400 -> 501 cents change would contaminate selection.
            spans = {w["index"]: w for w in pitch.walk_span_warnings(data, 0)}
            for warning in warnings:
                if warning["code"] == "walk_span_too_large":
                    transition = spans.get(warning["source_index"])
                    if transition and transition["span_cents"] >= warning.get("threshold_cents", 501):
                        fixed.add((warning["source_index"], warning["code"]))
            positions = {a["source_index"]: i for i, a in enumerate(actions)}
            for warning in warnings:
                index = warning["source_index"]
                pos = positions.get(index)
                if pos is None:
                    counts["warning_without_preview_action"] += 1
                    continue
                # Empty/sustain rows do not reset the parser. Stop at a new
                # attack/explicit position, so unrelated older 复 is excluded.
                control = None
                for previous in reversed(actions[:pos]):
                    text = str(previous.get("jianzi_text") or "").strip()
                    if CONTROL.fullmatch(text):
                        control = previous
                        break
                    if (pitch.parse_string_numbers(text)
                            or pitch.first_token(text, pitch.RIGHT_HAND_TECHNIQUES)
                            or pitch.parse_hui(text) is not None
                            or "泛起" in text or "泛止" in text):
                        break
                if control is None:
                    continue
                counts["control_preceded_warning_occurrences"] += 1
                disappeared = (index, warning["code"]) not in fixed
                if not disappeared:
                    counts["genuine_warning_preserved"] += 1
                cases.append({"trajectory_id": phrase_id, "sample_id": sample_id,
                              "tool_call_number": round_number,
                              "warning": warning, "control_source_index": control["source_index"],
                              "control_text": control["jianzi_text"],
                              "warning_text": actions[pos].get("jianzi_text"),
                              "previous_rows": [{"source_index": a["source_index"],
                                                 "text": a.get("jianzi_text")}
                                                for a in actions[max(0, pos-4):pos+1]],
                              "warning_disappears_with_fix": disappeared})
    selected = sorted({c["trajectory_id"] for c in cases if c["warning_disappears_with_fix"]})
    output.mkdir(parents=True, exist_ok=True)
    (output / "affected_phrase_ids.txt").write_text("".join(s + "\n" for s in selected), encoding="utf-8")
    selected_scores = {s.split("-p")[0] for s in selected}
    # Keep ALL phrases of each affected score for read-only history, but do
    # not replicate the unrelated 430 MB corpus into every retry worker.
    context_rows = 0
    with (output / "score_context_input.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows(source):
            if row["score_key"] in selected_scores:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                context_rows += 1
    report = {"source": str(source.resolve()), "audit": str(audit.resolve()),
              "selection": "all intermediate edit_plan rounds; preceding 复 or bare movement; repaired replay removes warning",
              "counts": dict(counts), "affected_phrase_count": len(selected),
              "affected_score_count": len(selected_scores), "context_phrase_count": context_rows,
              "affected_phrase_ids": selected,
              "affected_by_control": dict(Counter(c["control_text"] for c in cases if c["warning_disappears_with_fix"])),
              "cases": cases}
    (output / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"cases", "source", "audit"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    scan(args.source, args.audit, args.output_dir)
