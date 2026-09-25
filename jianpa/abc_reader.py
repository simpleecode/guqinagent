#!/usr/bin/env python3
"""Minimal ABC-notation melody reader (single-pass tokenizer).

Supports the subset folk/piano tune sets actually use: headers X/T/L/M/K,
notes with accidental/octave/duration, ties (merged), rests, bar lines,
chords (top two pitches kept), graces (zero-length events). Broken rhythm
and repeats are not expanded.
"""
from __future__ import annotations

import re

NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = (5, 0, 7, 2, 9, 4, 11)
FLAT_ORDER = (11, 4, 9, 2, 7, 0, 5)
MAJOR_KEYS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7}
MINOR_KEYS = {"Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7}
FLAT_KEYS = {"F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6}
TONIC_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# 严格锚定在 pos：意外号? 音名 八度标记? 时值(num/den 或 //简写)?
NOTE_RE = re.compile(r"([\^_=]*)([A-Ga-g])([',]*)(\d+)?(?:/(\d+))?(/+)?")
DURATION_RE = re.compile(r"(\d+)?(?:/(\d+))?")


def _key_info(key_header: str) -> tuple[int, dict[int, int], str]:
    key = key_header.strip().split()[0] if key_header.strip() else "C"
    key = key.replace("♯", "#").replace("♭", "b")
    minor = key.endswith("m") and len(key) > 1
    base = key[:-1] if minor else key
    base = base[0].upper() + base[1:]
    tonic_pc = TONIC_PC.get(base[0], 0)
    table: dict[int, int] = {}
    sharps = MAJOR_KEYS.get(base, MINOR_KEYS.get(key, 0))
    if sharps > 0:
        for pc in SHARP_ORDER[:sharps]:
            table[pc] = +1
    elif sharps < 0:
        for pc in FLAT_ORDER[:-sharps]:
            table[pc] = -1
    else:
        flats = FLAT_KEYS.get(base)
        if flats:
            for pc in FLAT_ORDER[:-flats]:
                table[pc] = -1
    return tonic_pc, table, ("minor" if minor else "major")


def _unit_quarters(text: str) -> float:
    match = re.match(r"1/(\d+)", text.strip())
    if not match:
        return 0.125
    return 4.0 / float(match.group(1))


def _parse_duration(text: str, pos: int, unit: float) -> tuple[float, int]:
    match = DURATION_RE.match(text, pos)
    if not match:
        return unit, pos
    num, den, consumed = match.group(1), match.group(2), match.end() - pos
    if not consumed:
        return unit, pos
    if num and den:
        return unit * float(num) / float(den), match.end()
    if num:
        return unit * float(num), match.end()
    if den:
        return unit / float(den), match.end()
    return unit, match.end()


def _tokenize_note(text: str, pos: int, unit: float, key_table: dict[int, int]):
    match = NOTE_RE.match(text, pos)
    if not match or match.end() == pos:
        return None
    sharp_flat, letter, octave_marks, num, den, slashes = match.groups()
    base_pc = NOTE_BASE[letter.upper()]
    if sharp_flat:
        alter = sharp_flat.count("^") - sharp_flat.count("_")
        pc = (base_pc + alter) % 12
    else:
        pc = (base_pc + key_table.get(base_pc, 0)) % 12
    octave = 5 if letter.islower() else 4
    octave += octave_marks.count("'") - octave_marks.count(",")
    midi = 12 * (octave + 1) + pc
    if num and den:
        quarters = unit * float(num) / float(den)
    elif num:
        quarters = unit * float(num)
    elif den:
        quarters = unit / float(den)
    elif slashes:
        quarters = unit / (2 ** len(slashes))
    else:
        quarters = unit
    return {"midi": midi, "quarters": quarters, "next": match.end()}


def parse_abc(text: str) -> dict:
    headers: dict[str, str] = {}
    body_chunks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if len(stripped) > 2 and stripped[1] == ":" and stripped[0] in "XTLMKQPR" and not body_chunks:
            headers[stripped[0]] = stripped[2:].strip()
            continue
        body_chunks.append(stripped)

    unit = _unit_quarters(headers.get("L", "1/8"))
    tonic_pc, key_table, mode = _key_info(headers.get("K", "C"))
    meter = headers.get("M", "4/4")
    try:
        num, _, den = meter.partition("/")
        beats = float(num) * 4.0 / float(den or "4")
    except ValueError:
        beats = 4.0

    body = " ".join(body_chunks)
    events: list[tuple[float, float, int]] = []
    graces: set[int] = set()
    bars: list[float] = []
    pos = 0.0
    pending_tie = False
    i = 0
    while i < len(body):
        ch = body[i]
        if ch in " \t":
            i += 1
        elif ch == "|":
            j = i + 1
            while j < len(body) and body[j] in "|:][":
                j += 1
            bars.append(pos)
            i = j
        elif ch == ":":
            i += 1
        elif ch == "-":
            pending_tie = True
            i += 1
        elif ch in "(<>" :
            i += 1  # slurs/broken rhythm not expanded
        elif ch in '"!+':
            closer = ch if ch != '"' else '"'
            j = body.find(closer, i + 1)
            i = j + 1 if j > 0 else i + 1
        elif ch in "zZxy":
            length, next_pos = _parse_duration(body, i + 1, unit)
            pos += length if ch in "zZ" else 0.0
            i = next_pos if next_pos > i + 1 else i + 1
        elif ch == "{":
            j = body.find("}", i + 1)
            inner = body[i + 1:j if j > 0 else len(body)]
            k = 0
            while k < len(inner):
                note = _tokenize_note(inner, k, unit, key_table)
                if note is None:
                    k += 1
                    continue
                events.append((pos, 0.0, note["midi"]))
                graces.add(len(events) - 1)
                k = note["next"]
            i = (j + 1) if j > 0 else len(body)
        elif ch == "[":
            j = body.find("]", i + 1)
            inner = body[i + 1:j if j > 0 else len(body)]
            chord: list[dict] = []
            k = 0
            while k < len(inner):
                note = _tokenize_note(inner, k, unit, key_table)
                if note is None:
                    k += 1
                    continue
                chord.append(note)
                k = note["next"]
            if chord:
                length = max(n["quarters"] for n in chord)
                for order, note in enumerate(sorted(chord, key=lambda n: -n["midi"])[:2]):
                    events.append((pos, length, note["midi"]))
                pos += length
            i = (j + 1) if j > 0 else len(body)
        else:
            note = _tokenize_note(body, i, unit, key_table)
            if note is None:
                i += 1
                continue
            if pending_tie and events and events[-1][2] == note["midi"]:
                start, _, midi = events[-1]
                events[-1] = (start, note["quarters"] + (events[-1][1] if events[-1][1] else 0), midi)
            else:
                events.append((pos, note["quarters"], note["midi"]))
            pending_tie = False
            pos += note["quarters"]
            i = note["next"]

    return {
        "title": headers.get("T", ""),
        "events": events,
        "graces": graces,
        "bars": bars,
        "beats_per_bar": beats,
        "tonic_pc": tonic_pc,
        "mode": mode,
    }
