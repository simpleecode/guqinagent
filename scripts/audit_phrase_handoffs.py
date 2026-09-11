#!/usr/bin/env python3
"""Audit phrase handoff continuity, compactness, and cross-section behavior."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ABC_J/agent_training/inferred")
    parser.add_argument("--output", type=Path, default=ROOT / "ABC_J/agent_training/phrase_handoff_report.json")
    args = parser.parse_args()
    by_score = defaultdict(list)
    errors, counts = [], Counter()
    context_sizes = []
    previous_sequence_sizes = []
    older_ref_sizes = []
    for split in ("train", "validation", "test"):
        path = args.input_dir / f"inferred_trajectories_{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                item = json.loads(line)
                handoff = item.get("input", {}).get("phrase_handoff")
                if not handoff:
                    errors.append(f"{path}:{line_number}: missing handoff")
                    continue
                by_score[item["score_key"]].append(item)
                counts["phrases"] += 1
                if handoff["section"]["boundary_from_previous"]:
                    counts["cross_section_phrases"] += 1
                if handoff.get("previous_phrase"):
                    counts["phrases_with_prior_output"] += 1
                    counts["previous_sequence_actions"] += len(
                        handoff["previous_phrase"].get("actions", [])
                    )
                    previous_sequence_sizes.append(len(
                        handoff["previous_phrase"].get("actions", [])
                    ))
                    older_ref_sizes.append(len(handoff.get("older_context_refs", [])))
                elif handoff.get("entry_state", {}).get("source_phrase"):
                    counts["phrases_with_prior_output"] += 1
                if handoff.get("right_lookahead"):
                    errors.append(f"{path}:{line_number}: v2 handoff contains forbidden lookahead")
    inherited_entry_fields = Counter()
    for score, items in by_score.items():
        ordered = sorted(items, key=lambda item: item["input"]["event_range"]["start"])
        for index, item in enumerate(ordered):
            handoff = item["input"]["phrase_handoff"]
            if str(handoff.get("schema_version", "")).startswith("phrase-handoff-2"):
                previous_phrase = handoff.get("previous_phrase")
                if index == 0:
                    if previous_phrase or handoff.get("older_context_refs"):
                        errors.append(f"{score}:{item['phrase_id']}: first phrase has history")
                    continue
                expected = ordered[index - 1]
                if not previous_phrase or previous_phrase.get("phrase_id") != expected["phrase_id"]:
                    errors.append(f"{score}:{item['phrase_id']}: wrong previous phrase")
                    continue
                if previous_phrase.get("actions") != expected["reference_plan"]["actions"]:
                    errors.append(f"{score}:{item['phrase_id']}: previous sequence mismatch")
                older_refs = handoff.get("older_context_refs", [])
                if index > 1 and len(older_refs) != index - 1:
                    errors.append(f"{score}:{item['phrase_id']}: wrong older context refs")
                continue
            entry = handoff["entry_state"]
            if index == 0:
                if entry.get("source_phrase") is not None:
                    errors.append(f"{score}:{item['phrase_id']}: first phrase inherits")
                continue
            previous = ordered[index - 1]["input"]["phrase_handoff"]["exit_state_target"]
            if entry != previous and not (
                handoff["section"]["boundary_from_previous"]
                and {**entry, "confidence": previous["confidence"]} == previous
            ):
                errors.append(f"{score}:{item['phrase_id']}: entry/exit mismatch")
            if entry.get("source_phrase") != ordered[index - 1]["phrase_id"]:
                errors.append(f"{score}:{item['phrase_id']}: wrong source phrase")
            references = item["reference_plan"]["actions"]
            if references:
                first = references[0]
                for field in first.get("inherited_fields", []):
                    inherited_entry_fields[field] += 1
                    counts["phrases_starting_with_inheritance"] += 1
                    break
    report = {
        "schema_version": "phrase-handoff-audit-2.0", "valid": not errors,
        "scores": len(by_score), "counts": dict(counts),
        "inherited_fields_at_phrase_start": dict(inherited_entry_fields),
        "context_window": {
            "model": "full_previous_phrase_plus_per_phrase_expand_refs",
            "right_lookahead": False,
            "max_previous_actions": max(previous_sequence_sizes, default=0),
            "max_older_expand_refs": max(older_ref_sizes, default=0),
        },
        "errors": errors[:200],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
