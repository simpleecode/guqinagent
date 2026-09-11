from __future__ import annotations

from collections import Counter
from typing import Any

from .models import CanonicalScore
from .pitch_candidates import candidates_for_event


STANDARD_OPEN_MIDI = [48.0, 50.0, 53.0, 55.0, 57.0, 60.0, 62.0]
TUNING_CATALOG: dict[str, list[float]] = {
    "正调": STANDARD_OPEN_MIDI,
    "紧五弦": [48, 50, 53, 55, 59, 60, 62],
    "慢三弦": [48, 50, 51, 55, 57, 60, 62],
    "慢二": [48, 48, 53, 55, 57, 60, 62],
    "紧五慢一": [46, 50, 53, 55, 59, 60, 62],
    "慢一三六": [46, 50, 51, 55, 57, 58, 62],
    "慢一三四六": [46, 50, 51, 53, 57, 58, 62],
    "紧二五": [48, 52, 53, 55, 59, 60, 62],
    "紧二五七": [48, 52, 53, 55, 59, 60, 64],
    "紧一二三四六七": [50, 52, 55, 55, 59, 62, 64],
}


def _pitch_class_distance(left: float, right: float) -> float:
    distance = abs((left - right) % 12)
    return min(distance, 12 - distance)


def score_tuning(score: CanonicalScore, name: str, open_midi: list[float], *,
                 tonic_midi: float | None = None) -> dict[str, Any]:
    sounding = [event for event in score.events if event.attack and event.midi is not None]
    pitch_counts = Counter(float(event.midi) for event in sounding)
    total = max(sum(pitch_counts.values()), 1)
    open_hits = sum(count for pitch, count in pitch_counts.items() if any(
        abs(pitch - opened) <= 0.1 for opened in open_midi
    ))
    harmonic_classes = {round((opened + interval) % 12, 3)
                        for opened in open_midi for interval in (12, 19, 24, 28, 31, 34)}
    harmonic_hits = sum(
        count for pitch, count in pitch_counts.items()
        if round(pitch % 12, 3) in harmonic_classes
    )
    candidate_counts, region_costs = [], []
    for event in sounding:
        candidates = candidates_for_event(
            event, open_midi, tolerance_cents=35.0,
            modes=("stopped",), max_candidates=18,
        )
        candidate_counts.append(len(candidates))
        if candidates:
            region_costs.append(candidates[0].tone_region_cost)
    coverage = sum(bool(value) for value in candidate_counts) / total
    mean_region_cost = sum(region_costs) / len(region_costs) if region_costs else 99.0
    retune_distance = sum(abs(a - b) for a, b in zip(open_midi, STANDARD_OPEN_MIDI))
    important = []
    if tonic_midi is not None:
        important = [tonic_midi % 12, (tonic_midi + 7) % 12]
    important_open = sum(any(
        _pitch_class_distance(opened % 12, pitch_class) <= 0.1 for opened in open_midi
    ) for pitch_class in important)
    important_harmonic = sum(any(
        _pitch_class_distance(candidate, pitch_class) <= 0.1 for candidate in harmonic_classes
    ) for pitch_class in important)
    utility = (3.0 * coverage + 1.6 * open_hits / total + 0.35 * harmonic_hits / total
               + 0.8 * important_open + 0.25 * important_harmonic
               - 0.08 * mean_region_cost - 0.04 * retune_distance)
    return {
        "name": name, "open_midi": [float(value) for value in open_midi],
        "utility": round(utility, 6), "pitch_coverage": round(coverage, 6),
        "open_note_ratio": round(open_hits / total, 6),
        "harmonic_pitch_class_ratio": round(harmonic_hits / total, 6),
        "tonic_dominant_open_count": important_open,
        "tonic_dominant_harmonic_count": important_harmonic,
        "mean_best_region_cost": round(mean_region_cost, 6),
        "retune_semitones": retune_distance, "sounding_events": len(sounding),
    }


def rank_tunings(score: CanonicalScore, *, names: list[str] | None = None,
                 tonic_midi: float | None = None) -> list[dict[str, Any]]:
    selected = names or list(TUNING_CATALOG)
    unknown = [name for name in selected if name not in TUNING_CATALOG]
    if unknown:
        raise ValueError(f"unknown tunings: {unknown}")
    rows = [score_tuning(score, name, TUNING_CATALOG[name], tonic_midi=tonic_midi)
            for name in selected]
    rows.sort(key=lambda row: (-row["utility"], row["retune_semitones"], row["name"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def decide_tuning(score: CanonicalScore, *, tonic_midi: float | None = None,
                  performance_context: str = "solo",
                  open_tone_demand: str = "auto",
                  minimum_retune_gain: float = 0.35) -> dict[str, Any]:
    """First decide whether retuning is justified, then choose an alternative.

    The default is to keep standard tuning. Retuning must provide a material
    resource gain; solo playing and explicit high open-tone demand lower, but
    never remove, that threshold.
    """
    ranked = rank_tunings(score, tonic_midi=tonic_midi)
    standard = next(row for row in ranked if row["name"] == "正调")
    alternative = next((row for row in ranked if row["name"] != "正调"), standard)
    gain = alternative["utility"] - standard["utility"]
    threshold = minimum_retune_gain
    if performance_context == "solo":
        threshold -= 0.08
    elif performance_context == "ensemble":
        threshold += 0.15
    if open_tone_demand == "high":
        threshold -= 0.12
    elif open_tone_demand == "low":
        threshold += 0.12
    threshold = max(threshold, 0.10)
    retune = alternative is not standard and gain >= threshold
    selected = alternative if retune else standard
    return {
        "retune": retune, "selected_name": selected["name"],
        "open_midi": selected["open_midi"], "gain_over_standard": round(gain, 6),
        "required_gain": round(threshold, 6), "standard_assessment": standard,
        "best_alternative": alternative, "ranked_candidates": ranked,
        "performance_context": performance_context, "open_tone_demand": open_tone_demand,
        "reason_codes": (["RETUNE_RESOURCE_GAIN"] if retune else ["KEEP_STANDARD_SUFFICIENT"]),
    }
