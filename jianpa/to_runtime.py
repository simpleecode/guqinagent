#!/usr/bin/env python3
"""Convert melody events (MIDI/ABC) into the project's inference inputs.

Outputs, mirroring train/scripts/build_public_score_eval_input.py exactly:
  1. <key>/jianpu_jianzi_readable.json  -- 中间谱面（可直接喂 PDF 出谱工具）
  2. <key>_public.jsonl                  -- eval_two_stage_score.py 的公开输入

The guqin always sounds in 正调 (movable-do, §audit comments): open_midi
stays [48,50,53,55,57,60,62] and "1=X" is only a label; tonic_degree1_midi
pins the do so the project's parse_jianpu round-trips our jianpu text.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges
from scripts.audit_jianpu_jianzi_pitch import MAJOR, parse_jianpu, parse_open_midi

OPEN_STRINGS = [
    {"string": i + 1, "pitch": p, "octave": o, "semitone_offset": 0}
    for i, (p, o) in enumerate(
        (("C", 3), ("D", 3), ("F", 3), ("G", 3), ("A", 3), ("C", 4), ("D", 4)))
]
OPEN_MIDI = [48, 50, 53, 55, 57, 60, 62]

MAJOR_SHARP_DEGREE = {0: "1", 1: "♯1", 2: "2", 3: "♯2", 4: "3", 5: "4", 6: "♯4",
                      7: "5", 8: "♯5", 9: "6", 10: "♯6", 11: "7"}
MINOR_DEGREE = {0: "1", 2: "2", 3: "♭3", 5: "4", 7: "5", 8: "♭6", 10: "♭7", 1: "♯1", 4: "3", 6: "♯4", 9: "6", 11: "7"}
DURATION_BUCKETS = ((2.0, "二分"), (1.0, "四分"), (0.5, "八分"), (0.25, "十六分"), (0.125, "三十二分"))
DOT_ABOVE, DOT_BELOW = "\u0307", "\u0323"
NOTE_LETTERS_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def anchor_tonic_midi(tonic_pc: int) -> int:
    """Tonic do in a stable register: C4-based, kept inside [55, 67)."""
    midi = 60 + tonic_pc
    while midi >= 67:
        midi -= 12
    while midi < 55:
        midi += 12
    return midi


def center_octaves(events):
    """Shift a whole melody by octaves so its median pitch sits in [60, 72).

    正调 is movable-do ("1=X" is only a label), so a uniform octave shift
    keeps the label and degrees identical while pulling the melody toward
    the guqin's range (open strings 48-62).  Graces (zero-length) shift too.
    """
    pitches = sorted(m for _, d, m in events if d > 0)
    if not pitches:
        return list(events)
    median = pitches[len(pitches) // 2]
    shift = 0
    while median + shift >= 72:
        shift -= 12
    while median + shift < 60:
        shift += 12
    if shift == 0:
        return list(events)
    return [(s, d, m + shift) for s, d, m in events]


# Guqin sounding span in 正调: open first string (48) up to the highest
# practical pressed position (fourth hui area on the seventh string).
GUQIN_LOW, GUQIN_HIGH = 48, 79


def fold_outliers(events):
    """Move only the notes outside the instrument's span by octaves.

    After whole-piece centering a few source extremes (cadenza sweeps,
    accompaniment leftovers) can still overshoot; folding just those keeps
    every target pitch actually on the instrument instead of handing the
    eval permanently unplayable notes.
    """
    folded = []
    for start, dur, midi in events:
        while midi < GUQIN_LOW:
            midi += 12
        while midi > GUQIN_HIGH:
            midi -= 12
        folded.append((start, dur, midi))
    return folded


def degree_text(midi: int, tonic_midi: int, mode: str) -> str:
    semitones = midi - tonic_midi
    octave, offset = divmod(semitones, 12) if semitones >= 0 else (semitones // 12, semitones % 12)
    table = MINOR_DEGREE if mode == "minor" else MAJOR_SHARP_DEGREE
    degree = table[offset]
    dots = (DOT_ABOVE * octave) if octave > 0 else (DOT_BELOW * -octave)
    return degree[0] + dots + degree[1:]


def _bucket(quarters: float) -> str:
    for length, name in DURATION_BUCKETS:
        if quarters + 1e-6 >= length:
            return name
    return "三十二分"


def _abc_token(midi: int, quarters: float) -> str:
    letter = NOTE_LETTERS_SHARP[midi % 12]
    token = f"{letter[0]}{'#' if len(letter) > 1 else ''}"
    octave_marks = midi // 12 - 5
    token += "'" * max(0, octave_marks) + "," * max(0, -octave_marks)
    if quarters >= 2:
        token += str(int(round(quarters / 2)))
    elif quarters == 1:
        pass
    elif quarters:
        token += f"/{int(round(1 / quarters))}"
    return token


def build_readable(events, bars, *, title, tonic_pc, mode, beats_per_bar) -> dict:
    """events: [(quarter_pos, quarter_dur, midi)]; graces are zero-dur events."""
    tonic_midi = anchor_tonic_midi(tonic_pc)
    bar_set = sorted({round(b, 4) for b in bars})
    rows = []
    index = 0
    merged = sorted(events, key=lambda e: (round(e[0], 4), -e[2]))
    pos_cursor = 0.0

    def add_row(jianpu, jianpu_alt, duration, abc):
        nonlocal index
        rows.append({
            "index": index, "abc": abc, "jianpu": jianpu,
            "jianpu_alt": jianpu_alt, "duration": duration, "lyric": "",
            "section": {"number": 1, "label": "一", "title": "", "start": index == 0, "marker": "<一>"},
        })
        index += 1

    def add_bar():
        nonlocal index
        rows.append({
            "index": index, "abc": "|", "jianpu": "|", "jianpu_alt": None,
            "duration": "小节线", "lyric": "",
            "section": {"number": 1, "label": "一", "title": "", "start": False, "marker": "<一>"},
        })
        index += 1

    bar_idx = 0
    for event_index, (start, dur, midi) in enumerate(merged):
        while bar_idx < len(bar_set) and bar_set[bar_idx] <= start + 1e-6:
            if bar_set[bar_idx] >= pos_cursor - 1e-6:
                add_bar()
                pos_cursor = bar_set[bar_idx]
            bar_idx += 1
        if dur <= 1e-6:  # grace
            add_row(degree_text(midi, tonic_midi, mode), None, "装饰音", _abc_token(midi, 0.125))
            continue
        if start > pos_cursor + 1e-3:
            gap = start - pos_cursor
            while gap > 1e-3:
                take = min(gap, 1.0)
                add_row("0（休止）", None, _bucket(take), f"z{'' if take == 1 else int(round(take * 2))}")
                gap -= take
            pos_cursor = start
        # top-two chord voices at the same onset
        chord = [m for s, d, m in merged[event_index:event_index + 3]
                 if abs(s - start) < 1e-3][:2]
        main = chord[0] if chord else midi
        alt = chord[1] if len(chord) > 1 else None
        remaining = dur
        first = True
        while remaining > 1e-3:
            bucket_len = next((l for l, n in DURATION_BUCKETS if l <= remaining + 1e-6), 0.125)
            if first:
                add_row(degree_text(main, tonic_midi, mode),
                        degree_text(alt, tonic_midi, mode) if alt is not None else None,
                        _bucket(bucket_len), _abc_token(main, bucket_len))
                first = False
            else:
                add_row("－（延音）", None, "", "-")
            remaining -= bucket_len
            pos_cursor = start + dur - remaining
    metadata = {
        "score_title": title,
        "score_id": 0,
        "score_key": "",
        "meter": "4/4",
        "tonic": f"1={'CDEFGAB'[0]}",  # replaced below
        "tempo": "",
        "tuning": {"name": "正调", "value": [0] * 7, "open_strings": OPEN_STRINGS},
    }
    letters = ("C", "D", "E", "F", "G", "A", "B")
    metadata["tonic"] = f"1={letters[0]}"  # placeholder, caller sets real label
    return {
        "metadata": metadata,
        "tonic_pc": tonic_pc,
        "mode": mode,
        "tonic_degree1_midi": tonic_midi,
        "notes": rows,
    }


def build_runtime(readable: dict, *, score_key: str, split: str = "test") -> list[dict]:
    metadata = dict(readable["metadata"])
    metadata["score_key"] = score_key
    metadata["tonic"] = readable["tonic_label"]
    metadata["tonic_degree1_midi"] = readable["tonic_degree1_midi"]
    tonic_midi = readable["tonic_degree1_midi"]
    notes = []
    event_index = 0
    for row in readable["notes"]:
        note = {key: row.get(key) for key in
                ("index", "abc", "jianpu", "jianpu_alt", "duration", "lyric", "section")}
        is_bar = note.get("duration") == "小节线"
        note["event_index"] = None if is_bar else event_index
        if not is_bar:
            event_index += 1
        notes.append(note)
    ranges = split_phrase_ranges(
        notes, max_sounding=16,
        is_sounding=lambda note: any(
            parse_jianpu(note.get(field), tonic_midi) is not None
            for field in ("jianpu", "jianpu_alt")),
    )
    rows = []
    for number, (start, end) in enumerate(ranges, 1):
        phrase_id = f"p{number:04d}"
        runtime_item = {
            "trajectory_id": f"{score_key}-{phrase_id}",
            "score_key": score_key, "phrase_id": phrase_id, "split": split,
            "input": {
                "metadata": metadata,
                "normalized_tuning": {"name": "正调", "open_midi": list(OPEN_MIDI)},
                "event_range": {"start": start, "end_exclusive": end},
                "notes_without_jianzi": notes[start:end],
                "phrase_handoff": {
                    "phrase_id": phrase_id,
                    "current_phrase": notes[start:end],
                    "event_range": {"start": start, "end_exclusive": end},
                    "section": dict(notes[start].get("section") or {}),
                },
            },
            "baseline_plan": {"actions": []},
        }
        payload = {
            "schema_version": "agent-eval-input-2.0",
            "sample_id": runtime_item["trajectory_id"],
            "split": split, "score_key": score_key, "phrase_id": phrase_id,
            "runtime_item": runtime_item,
        }
        payload["input_sha256"] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        rows.append(payload)
    return rows


def write_outputs(readable: dict, *, score_key: str, out_dir: Path, split: str = "test") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = out_dir / score_key
    score_dir.mkdir(exist_ok=True)
    readable_path = score_dir / "jianpu_jianzi_readable.json"
    readable_path.write_text(json.dumps(readable, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = build_runtime(readable, score_key=score_key, split=split)
    jsonl_path = out_dir / f"{score_key}_public.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8")
    return {"readable": str(readable_path), "public": str(jsonl_path), "phrases": len(rows)}
