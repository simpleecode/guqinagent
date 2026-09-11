#!/usr/bin/env python3
"""Build phrase-level reference trajectory skeletons from the split manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges
MANIFEST = ROOT / "ABC_J" / "results" / "dataset_split_groups.csv"
OUTPUT_DIR = ROOT / "ABC_J" / "agent_training"
AUDIT_PATH = ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"
TECHNIQUES = (
    "进复", "退复", "泛起", "泛止", "掐起", "带起", "抓起", "爪起", "滔起", "往来",
    "绰", "注", "吟", "猱", "撞", "逗", "撮", "泼", "剌", "进", "退", "复",
)


def load_audit():
    spec = importlib.util.spec_from_file_location("trajectory_pitch_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit()


def sounding(note: dict, tonic_midi: float) -> bool:
    return any(
        AUDIT.parse_jianpu(note.get(field), tonic_midi) is not None
        for field in ("jianpu", "jianpu_alt")
    )


def technique_tokens(text: str) -> list[str]:
    remaining = text
    result: list[str] = []
    for token in sorted(TECHNIQUES, key=len, reverse=True):
        count = remaining.count(token)
        result.extend([token] * count)
        remaining = remaining.replace(token, "")
    return result


def phrase_ranges(notes: list[dict], max_sounding: int, tonic_midi: float) -> list[tuple[int, int]]:
    """Split at sections, then at a barline near the phrase size limit."""
    return split_phrase_ranges(
        notes, max_sounding=max_sounding,
        is_sounding=lambda note: sounding(note, tonic_midi),
    )


def build_score_trajectories(row: dict[str, str], max_sounding: int,
                             context_notes: int) -> list[dict]:
    path = Path(row["final_data_path"]) / "jianpu_jianzi_readable.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata") or {}
    notes = data.get("notes", [])
    tonic_midi = AUDIT.parse_tonic_midi(metadata)
    try:
        audit = AUDIT.audit(data, 50.0)
        detail_by_index = {item["index"]: item for item in audit["details"]}
        pitch_status = "available"
    except ValueError as exc:
        if "open_strings must contain 7 strings" not in str(exc):
            raise
        detail_by_index = {}
        pitch_status = "unavailable_missing_tuning"

    trajectories: list[dict] = []
    for phrase_number, (start, end) in enumerate(
        phrase_ranges(notes, max_sounding, tonic_midi), 1
    ):
        phrase_notes = notes[start:end]
        reference_actions = []
        technique_counts: Counter[str] = Counter()
        matched = compared = targets = 0
        for note in phrase_notes:
            is_sounding = sounding(note, tonic_midi)
            targets += int(is_sounding)
            detail = detail_by_index.get(note.get("index"), {})
            if detail.get("status") in {"matched", "mismatched"}:
                compared += 1
                matched += int(detail["status"] == "matched")
            jianzi = str(note.get("jianzi", ""))
            techniques = technique_tokens(jianzi)
            technique_counts.update(techniques)
            reference_actions.append({
                "source_index": note.get("index"),
                "abc": note.get("abc"),
                "jianpu": note.get("jianpu"),
                "duration": note.get("duration"),
                "lyric": note.get("lyric", ""),
                "jianzi": jianzi,
                "techniques": techniques,
                "pitch_audit": {
                    key: detail[key] for key in (
                        "status", "reason", "pairs", "mean_absolute_cents"
                    ) if key in detail
                },
            })
        phrase_id = f"p{phrase_number:04d}"
        identity = f"{row['score_key']}:{start}:{end}:trajectory-v1"
        trajectory_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        trajectories.append({
            "trajectory_version": "1.0",
            "trajectory_id": trajectory_id,
            "score_family_id": row["leakage_group_id"],
            "score_key": row["score_key"],
            "score_id": row.get("score_id", ""),
            "title": metadata.get("score_title", row.get("candidate_title", "")),
            "phrase_id": phrase_id,
            "split": row["split"],
            "input": {
                "metadata": metadata,
                "event_range": {"start_index": start, "end_index_exclusive": end},
                "left_context": notes[max(0, start - context_notes):start],
                "phrase_notes": [
                    {key: note.get(key) for key in (
                        "index", "abc", "jianpu", "jianpu_alt", "duration", "lyric", "section"
                    )}
                    for note in phrase_notes
                ],
                "right_context": notes[end:min(len(notes), end + context_notes)],
                "baseline_plan": None,
            },
            "steps": [
                {
                    "type": "observation",
                    "content": {
                        "target_events": targets,
                        "pitch_status": pitch_status,
                        "section": phrase_notes[0].get("section") if phrase_notes else None,
                    },
                },
                {
                    "type": "reference_decision",
                    "content": {"actions": reference_actions},
                },
                {
                    "type": "audit",
                    "content": {
                        "pitch_compared": compared,
                        "pitch_matched": matched,
                        "pitch_accuracy_at_50c": round(matched / compared, 6) if compared else None,
                        "pitch_coverage": round(compared / targets, 6) if targets else None,
                    },
                },
            ],
            "reference_plan": {"actions": reference_actions},
            "output_plan": {"actions": reference_actions},
            "rewards": {
                "reference": 1.0,
                "pitch_accuracy_at_50c": round(matched / compared, 6) if compared else None,
                "pitch_coverage": round(compared / targets, 6) if targets else None,
                "technique_count": sum(technique_counts.values()),
                "technique_counts": dict(sorted(technique_counts.items())),
            },
            "provenance": {
                "source": str(path),
                "dataset_manifest": str(MANIFEST),
                "dataset_split_seed": 20260812,
                "builder": "build_reference_agent_trajectories.py",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        })
    return trajectories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-sounding", type=int, default=32)
    parser.add_argument("--context-notes", type=int, default=4)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (args.output_dir / f"reference_trajectories_{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        )
        for split in ("train", "validation", "test")
    }
    counts = Counter()
    score_counts = Counter()
    try:
        for index, row in enumerate(rows, 1):
            trajectories = build_score_trajectories(
                row, args.max_sounding, args.context_notes
            )
            for item in trajectories:
                handles[item["split"]].write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[item["split"]] += 1
            score_counts[row["split"]] += 1
            if index % 25 == 0:
                print(f"built {index}/{len(rows)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "schema_version": "1.0",
        "manifest": str(args.manifest),
        "max_sounding_per_phrase": args.max_sounding,
        "context_notes": args.context_notes,
        "scores": dict(score_counts),
        "trajectories": dict(counts),
    }
    (args.output_dir / "reference_trajectories_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
