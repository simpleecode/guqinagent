#!/usr/bin/env python3
"""Report phrase and source-section sizes in inferred agent trajectories."""
from __future__ import annotations

import importlib.util
import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges


INPUT = ROOT / "ABC_J" / "agent_training" / "inferred"
spec = importlib.util.spec_from_file_location(
    "training_stats_audit", ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load pitch auditor")
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    args = parser.parse_args()
    items = []
    for split in ("train", "validation", "test"):
        path = args.input_dir / f"inferred_trajectories_{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            items.extend(json.loads(line) for line in handle if line.strip())

    phrase_sizes = []
    sections = defaultdict(lambda: {"rows": 0, "sounding": 0})
    for item in items:
        notes = item["input"]["notes_without_jianzi"]
        tonic = AUDIT.parse_tonic_midi(item["input"]["metadata"])
        sounding = 0
        for note in notes:
            is_sounding = any(
                AUDIT.parse_jianpu(note.get(field), tonic) is not None
                for field in ("jianpu", "jianpu_alt")
            )
            sounding += int(is_sounding)
            section = note.get("section") or {}
            key = (
                item["score_key"], section.get("number"),
                section.get("marker"), section.get("title"),
            )
            sections[key]["rows"] += 1
            sections[key]["sounding"] += int(is_sounding)
        phrase_sizes.append({
            "trajectory_id": item["trajectory_id"],
            "split": item["split"],
            "rows": len(notes), "sounding": sounding,
        })

    max_rows = max(phrase_sizes, key=lambda item: item["rows"])
    max_sounding = max(phrase_sizes, key=lambda item: item["sounding"])
    max_section_rows = max(sections.items(), key=lambda item: item[1]["rows"])
    max_section_sounding = max(sections.items(), key=lambda item: item[1]["sounding"])
    by_score = defaultdict(list)
    for item in items:
        by_score[item["score_key"]].append(item)
    projected_count = 0
    projected_max = 0
    for score_items in by_score.values():
        score_items.sort(key=lambda item: item["input"]["event_range"]["start"])
        notes = [
            note for item in score_items
            for note in item["input"]["notes_without_jianzi"]
        ]
        tonic = AUDIT.parse_tonic_midi(score_items[0]["input"]["metadata"])
        is_sounding = lambda note: any(
            AUDIT.parse_jianpu(note.get(field), tonic) is not None
            for field in ("jianpu", "jianpu_alt")
        )
        ranges = split_phrase_ranges(
            notes, max_sounding=32, is_sounding=is_sounding,
        )
        projected_count += len(ranges)
        projected_max = max(
            projected_max,
            *(sum(is_sounding(note) for note in notes[start:end])
              for start, end in ranges),
        )
    report = {
        "training_phrases": len(items),
        "phrases_by_split": {
            split: sum(item["split"] == split for item in phrase_sizes)
            for split in ("train", "validation", "test")
        },
        "phrase_max_rows": max_rows,
        "phrase_max_sounding_notes": max_sounding,
        "train_max_sounding_notes": max(
            (item for item in phrase_sizes if item["split"] == "train"),
            key=lambda item: item["sounding"],
        ),
        "source_sections": len(sections),
        "corrected_split_projection": {
            "phrases": projected_count,
            "max_sounding_notes": projected_max,
        },
        "section_max_rows": {"key": max_section_rows[0], **max_section_rows[1]},
        "section_max_sounding_notes": {
            "key": max_section_sounding[0], **max_section_sounding[1],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
