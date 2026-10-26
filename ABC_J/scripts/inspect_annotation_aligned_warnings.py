"""Reproduce warnings shared by accepted notation and its annotation snapshot.

This is read-only with respect to the training datasets.  It records the
local source and reference context so aligned warnings can be inspected rather
than silently classified as either model defects or auditor false positives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_remaining_warning_rebuild import (
    SOURCE,
    canonical_jianzi_text,
    harmonic_region_at_phrase_start,
    notation_by_index,
    reference_plan,
    validate_jianzi_only,
)


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path)
    args = parser.parse_args()

    source = list(rows(SOURCE))
    history = {(row["score_key"], row["phrase_id"]): row for row in source}
    audits = {row["sample_id"]: row["teacher_private"] for row in rows(args.audit)}
    selected_ids = (
        {line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines()
         if line.strip()}
        if args.ids_file else None
    )
    cases = []
    final_events = []
    counts = Counter()
    overall_counts = Counter()
    overall_phrases = set()
    all_counts = Counter()
    all_phrases = set()
    for number, original in enumerate(source, 1):
        phrase_id = original["trajectory_id"]
        if selected_ids is not None and phrase_id not in selected_ids:
            continue
        finger = audits[f"{phrase_id}-fingering_agent-teacher-tools"]
        guqin = audits[f"{phrase_id}-guqinization-teacher-tools"]
        item = dict(original)
        item["baseline_plan"] = finger["accepted_plan"]
        item["harmonic_region_at_start"] = harmonic_region_at_phrase_start(item, history)
        _, final = validate_jianzi_only(
            item, guqin["accepted_patches"], toward_reference=True, historical=history,
        )
        final_text = notation_by_index(final["plan"])
        reference_item = dict(item)
        reference_item["baseline_plan"] = reference_plan(item)
        _, reference = validate_jianzi_only(
            reference_item, [], require_complete=False, historical=history,
        )
        reference_text = notation_by_index(reference_item["baseline_plan"])
        reference_codes = {
            (int(w["source_index"]), str(w["code"]))
            for w in reference.get("warnings") or []
            if w.get("source_index") is not None
        }
        notes = original["input"].get("notes_without_jianzi") or []
        by_index = {int(note["index"]): (position, note)
                    for position, note in enumerate(notes)}
        for warning in final.get("warnings") or []:
            overall_counts[str(warning.get("code") or "unknown")] += 1
            overall_phrases.add(phrase_id)
            final_events.append({"phrase_id": phrase_id,
                                 "index": warning.get("source_index"),
                                 "code": warning.get("code")})
            if warning.get("source_index") is None or warning.get("code") == "missing_independent_attack":
                continue
            index = int(warning["source_index"])
            code = str(warning["code"])
            all_counts[code] += 1
            all_phrases.add(phrase_id)
            if (index, code) not in reference_codes:
                continue
            if canonical_jianzi_text(final_text.get(index, "")) != canonical_jianzi_text(reference_text.get(index, "")):
                continue
            position = by_index.get(index, (None, None))[0]
            context = []
            if position is not None:
                for note in notes[max(0, position - 2):position + 3]:
                    neighbor = int(note["index"])
                    context.append({
                        "index": neighbor,
                        "abc": note.get("abc"),
                        "jianpu": note.get("jianpu"),
                        "reference": reference_text.get(neighbor, ""),
                        "final": final_text.get(neighbor, ""),
                    })
            cases.append({
                "phrase_id": phrase_id,
                "score_key": original.get("score_key"),
                "index": index,
                "code": code,
                "warning": warning,
                "context": context,
            })
            counts[code] += 1
        if number % 500 == 0:
            print(f"audited {number}/{len(source)}", flush=True)
    result = {"audit": str(args.audit), "events": len(cases),
              "codes": dict(sorted(counts.items())),
              "all_final_events": sum(overall_counts.values()),
              "all_final_phrases": len(overall_phrases),
              "all_final_codes": dict(sorted(overall_counts.items())),
              "all_non_missing_events": sum(all_counts.values()),
              "all_non_missing_phrases": len(all_phrases),
              "all_non_missing_codes": dict(sorted(all_counts.items())),
              "final_events": final_events,
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "final_events"}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
