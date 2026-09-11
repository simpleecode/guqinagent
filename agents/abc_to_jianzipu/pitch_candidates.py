from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

from .models import CanonicalScore, PositionCandidate, ScoreEvent


ROOT = Path(__file__).resolve().parents[2]
MAPPER_PATH = ROOT / "skills" / "guqin-pitch-mapper" / "scripts" / "guqin_pitch_mapper.py"
_SPEC = importlib.util.spec_from_file_location("abc_to_jianzipu_pitch_mapper", MAPPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load pitch mapper: {MAPPER_PATH}")
MAPPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MAPPER)


ABC_CHORD_RE = re.compile(r"\[([A-Ga-z][^]]*)\]")
ABC_NOTE_RE = re.compile(r"([\^_=]*)([A-Ga-g])([',]*)")
ABC_LETTER_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MODE_ALIASES = {
    "open": "open",
    "stopped": "stopped",
    "pressed": "stopped",
    "harmonic": "harmonic",
    "散音": "open",
    "按音": "stopped",
    "泛音": "harmonic",
}


def normalize_modes(modes: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize public Chinese labels and reject unknown pitch modes."""
    normalized: list[str] = []
    for value in modes:
        mode = MODE_ALIASES.get(str(value).strip())
        if mode is None:
            raise ValueError(f"unsupported pitch candidate mode: {value}")
        if mode not in normalized:
            normalized.append(mode)
    if not normalized:
        raise ValueError("at least one pitch candidate mode is required")
    return tuple(normalized)


def abc_chord_tones(abc: str) -> list[tuple[str, float]]:
    """(token, midi) for each tone of an ABC chord such as [D,A,]; [] if none.

    Uppercase letters sit at octave 4 (C=60), lowercase one octave higher;
    trailing commas drop and apostrophes raise octaves per ABC convention.
    """
    match = ABC_CHORD_RE.search(abc)
    if not match:
        return []
    tones = []
    for accidentals, letter, octaves in ABC_NOTE_RE.findall(match.group(1)):
        value = 60 + ABC_LETTER_SEMITONES[letter.upper()]
        value += sum(1 if char == "^" else -1 for char in accidentals)
        value += 12 if letter.islower() else 0
        value += sum(-12 if char == "," else 12 for char in octaves)
        tones.append((f"{accidentals}{letter}{octaves}", float(value)))
    return tones


def abc_chord_midis(abc: str) -> list[float]:
    """All pitches of an ABC chord token; [] for non-chords."""
    return [midi for _, midi in abc_chord_tones(abc)]


def _pitch(open_midi: float, hui: float | None, mode: str) -> float:
    if mode == "open" or hui is None:
        return open_midi
    if mode == "harmonic":
        if hui != int(hui):
            raise ValueError("fractional_hui_not_valid_for_harmonic")
        return open_midi + MAPPER.HARMONIC_SEMITONES[int(hui) - 1]
    coordinate = MAPPER.hui_coordinate(hui)
    return open_midi + 12 * math.log2(1 / coordinate)


def _tone_region_cost(hui: float | None, mode: str) -> float:
    if mode == "open":
        return 0.25
    if mode == "harmonic":
        return 0.45
    assert hui is not None
    # The central working region receives a mild prior, not a hard rule.
    cost = abs(hui - 7.5) * 0.12
    if hui < 4:
        cost += (4 - hui) * 0.8
    if hui > 11:
        cost += (hui - 11) * 0.9
    return round(cost, 6)


def candidates_for_event(event: ScoreEvent, open_midi: list[float], *,
                         tolerance_cents: float = 35.0,
                         modes: tuple[str, ...] = ("stopped",),
                         max_candidates: int = 18) -> list[PositionCandidate]:
    if event.midi is None or not event.attack:
        return []
    target = float(event.midi)
    modes = normalize_modes(modes)
    raw: list[PositionCandidate] = []
    sequence = 0
    for string, opened in enumerate(open_midi, 1):
        for mode in modes:
            if mode == "open":
                hui_values: list[float | None] = [None]
            elif mode == "harmonic":
                hui_values = [float(value) for value in range(1, 14)]
            else:
                hui_values = [value / 10 for value in range(10, 131)]
            for hui in hui_values:
                sounding = _pitch(opened, hui, mode)
                cents = (sounding - target) * 100
                if abs(cents) > tolerance_cents:
                    continue
                sequence += 1
                raw.append(PositionCandidate(
                    candidate_id=f"{event.id}-c{sequence:03d}",
                    event_id=event.id,
                    target_midi=target,
                    mode=mode,
                    string=string,
                    hui=hui,
                    sounding_midi=round(sounding, 6),
                    cents_error=round(cents, 3),
                    tone_region_cost=_tone_region_cost(hui, mode),
                    confidence="exact" if abs(cents) <= 10 else "approximate",
                ))
    raw.sort(key=lambda item: (
        round(abs(item.cents_error), 3),
        # 音分差相同时散音优先：空弦最省事且无按弦噪声。
        0 if item.mode == "open" else 1,
        item.tone_region_cost,
        item.string,
        item.hui or 0,
    ))
    selected = raw[:max_candidates]
    # Renumber after pruning so artifacts are compact and deterministic.
    for index, candidate in enumerate(selected, 1):
        candidate.candidate_id = f"{event.id}-c{index:02d}"
    return selected


def generate_candidates(score: CanonicalScore, *, tolerance_cents: float = 35.0,
                        modes: tuple[str, ...] = ("stopped",),
                        max_candidates: int = 18) -> dict[str, list[PositionCandidate]]:
    result: dict[str, list[PositionCandidate]] = {}
    for event in score.events:
        if event.midi is None or not event.attack:
            continue
        result[event.id] = candidates_for_event(
            event, score.open_midi,
            tolerance_cents=tolerance_cents,
            modes=modes,
            max_candidates=max_candidates,
        )
    return result
