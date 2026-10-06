#!/usr/bin/env python3
"""Build one public two-stage evaluation JSONL from audited modern scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges
from scripts.audit_jianpu_jianzi_pitch import parse_jianpu, parse_tonic_midi, parse_open_midi


def fingerprint(payload: dict) -> str:
    import hashlib
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def rows_for(readable_path: Path, split: str, max_sounding: int) -> list[dict]:
    source = json.loads(readable_path.read_text(encoding="utf-8"))
    metadata = dict(source["metadata"])
    score_key = str(metadata["score_key"])
    tonic_midi = parse_tonic_midi(metadata)
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
        notes, max_sounding=max_sounding,
        is_sounding=lambda note: any(parse_jianpu(note.get(field), tonic_midi) is not None
                                     for field in ("jianpu", "jianpu_alt")),
    )
    tuning = metadata.get("tuning") or {}
    rows = []
    for number, (start, end) in enumerate(ranges, 1):
        phrase_id = f"p{number:04d}"
        runtime_item = {
            "trajectory_id": f"{score_key}-{phrase_id}", "score_key": score_key,
            "phrase_id": phrase_id, "split": split,
            "input": {
                "metadata": metadata,
                "normalized_tuning": {"name": str(tuning.get("name") or ""),
                                      "open_midi": parse_open_midi(metadata)},
                "event_range": {"start": start, "end_exclusive": end},
                "notes_without_jianzi": notes[start:end],
                "phrase_handoff": {"phrase_id": phrase_id, "current_phrase": notes[start:end],
                                   "event_range": {"start": start, "end_exclusive": end},
                                   "section": dict((notes[start].get("section") or {}))},
            },
            "baseline_plan": {"actions": []},
        }
        payload = {"schema_version": "agent-eval-input-2.0",
                   "sample_id": runtime_item["trajectory_id"], "split": split,
                   "score_key": score_key, "phrase_id": phrase_id,
                   "runtime_item": runtime_item}
        payload["input_sha256"] = fingerprint(payload)
        rows.append(payload)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modern-root", type=Path, default=Path("ABC_J/modern"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-key", action="append", required=True)
    parser.add_argument("--split", default="modern")
    parser.add_argument("--max-sounding", type=int, default=16)
    args = parser.parse_args()
    rows = []
    for key in args.score_key:
        readable = args.modern_root / key / "mapped" / "jianpu_jianzi_readable.json"
        if not readable.exists():
            raise SystemExit(f"missing selected mapped score: {readable}")
        rows.extend(rows_for(readable, args.split, args.max_sounding))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                      for row in rows), encoding="utf-8")
    print(json.dumps({"scores": len(args.score_key), "phrases": len(rows),
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
