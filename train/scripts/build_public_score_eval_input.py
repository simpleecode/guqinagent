#!/usr/bin/env python3
"""Build public two-stage inference input from one mapped readable score.

The resulting JSONL intentionally contains no ``jianzi`` field, references, or
teacher-only state.  It is suitable for ``eval_two_stage_score.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges


def load_audit():
    path = ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"
    spec = importlib.util.spec_from_file_location("public_score_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="mapped/jianpu_jianzi_readable.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sounding", type=int, default=16)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    audit = load_audit()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = dict(source["metadata"])
    score_key = str(metadata["score_key"])
    tonic_midi = audit.parse_tonic_midi(metadata)
    notes: list[dict] = []
    event_index = 0
    for raw in source["notes"]:
        note = {key: raw.get(key) for key in (
            "index", "abc", "jianpu", "jianpu_alt", "duration", "lyric", "section",
            "notation_omitted",
        )}
        is_bar = (str(note.get("abc") or "").strip() == "|"
                  or str(note.get("jianpu") or "").strip() == "|"
                  or note.get("duration") == "小节线")
        note["event_index"] = None if is_bar else event_index
        if not is_bar:
            event_index += 1
        notes.append(note)

    ranges = split_phrase_ranges(
        notes,
        max_sounding=args.max_sounding,
        is_sounding=lambda note: any(
            audit.parse_jianpu(note.get(field), tonic_midi) is not None
            for field in ("jianpu", "jianpu_alt")
        ),
    )
    tuning = metadata.get("tuning") or {}
    rows = []
    for number, (start, end) in enumerate(ranges, 1):
        phrase_id = f"p{number:04d}"
        runtime_item = {
            "trajectory_id": f"{score_key}-{phrase_id}",
            "score_key": score_key,
            "phrase_id": phrase_id,
            "split": args.split,
            "input": {
                "metadata": metadata,
                "normalized_tuning": {
                    "name": str(tuning.get("name") or ""),
                    "open_midi": audit.parse_open_midi(metadata),
                },
                "event_range": {"start": start, "end_exclusive": end},
                "notes_without_jianzi": notes[start:end],
                "phrase_handoff": {
                    "phrase_id": phrase_id,
                    "current_phrase": notes[start:end],
                    "event_range": {"start": start, "end_exclusive": end},
                    "section": dict((notes[start].get("section") or {})),
                },
            },
            "baseline_plan": {"actions": []},
        }
        payload = {
            "schema_version": "agent-eval-input-2.0",
            "sample_id": runtime_item["trajectory_id"],
            "split": args.split,
            "score_key": score_key,
            "phrase_id": phrase_id,
            "runtime_item": runtime_item,
        }
        payload["input_sha256"] = fingerprint(payload)
        rows.append(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"score_key": score_key, "phrases": len(rows),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
