#!/usr/bin/env python3
"""Krumhansl-Schmuckler key detection over a duration-weighted histogram."""
from __future__ import annotations

import math

MAJOR_PROFILE = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR_PROFILE = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

SHARP_LETTERS = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
FLAT_LETTERS = ("C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B")


def _correlation(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


def detect_key(events: list[tuple[float, float, int]]) -> tuple[int, str, float]:
    """events = [(quarter_start, quarter_dur, midi), ...] → (tonic_pc, 'major'|'minor', score)."""
    histogram = [0.0] * 12
    for _, dur, midi in events:
        histogram[midi % 12] += max(dur, 0.125)
    best = (0, "major", -2.0)
    for shift in range(12):
        rotated = histogram[shift:] + histogram[:shift]
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            score = _correlation(rotated, list(profile))
            if score > best[2]:
                best = (shift, mode, score)
    return best


def tonic_label(tonic_pc: int, mode: str) -> str:
    letter = SHARP_LETTERS[tonic_pc] if mode == "major" else FLAT_LETTERS[tonic_pc]
    return f"1={letter}"
