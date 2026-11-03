"""Reusable structured-field and set metrics for parsed jianzipu events."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def _valid(value: Any) -> bool:
    return value is not None and value != ""


def field_counts(events: Iterable[dict[str, Any]], field: str) -> Counter:
    """Micro counts for one semantic field, omitting unavailable references."""
    counts: Counter = Counter()
    for row in events:
        ref, pred = row["reference"].get(field), row["prediction"].get(field)
        if not _valid(ref):
            continue
        counts["n"] += 1
        equal = ref == pred
        counts["correct"] += int(equal)
        counts["tp"] += int(equal and _valid(pred))
        counts["fp"] += int(_valid(pred) and not equal)
        counts["fn"] += int(not equal)
    return counts


def resolved_positions(detail: dict[str, Any]) -> list[list[Any]]:
    """Auditor-resolved string/contact pairs, not guessed from surface text.

    An open string has no hui contact; it receives a distinct OPEN label.
    Unresolved/partial compound events are unavailable, never wildcards.
    """
    if detail.get("status") not in {"matched", "mismatched"}:
        return []
    positions = detail.get("resolved_pitches") or []
    if len(positions) != len(detail.get("actual_midis") or []):
        return []
    result = []
    for position in positions:
        string, hui = position.get("string"), position.get("hui")
        if not isinstance(string, int) or not 1 <= string <= 7:
            return []
        if position.get("mode") == "open":
            hui = "OPEN"
        elif hui is None:
            return []
        elif isinstance(hui, (int, float)):
            hui = round(float(hui), 6)
        result.append([string, hui])
    return result


def string_hui_metrics(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Micro F1 of exact (string, hui) sets on reference-resolvable events."""
    counts: Counter = Counter()
    eligible = excluded = 0
    for row in events:
        if not row.get("target_midi"):
            continue
        eligible += 1
        ref = {tuple(p) for p in row["reference"].get("string_hui_positions", [])}
        if not ref:
            excluded += 1
            continue
        pred = {tuple(p) for p in row["prediction"].get("string_hui_positions", [])}
        counts["tp"] += len(ref & pred)
        counts["fp"] += len(pred - ref)
        counts["fn"] += len(ref - pred)
    return {**rates(counts, include_accuracy=False),
            "reference_evaluable_events": eligible - excluded,
            "target_events": eligible, "reference_unresolved_events": excluded,
            "reference_event_coverage": (eligible - excluded) / eligible if eligible else None,
            "definition": "event-local exact string/hui set micro-F1; OPEN for open strings; all resolved compound positions; unresolved references excluded, unresolved predictions empty"}


def set_counts(events: Iterable[dict[str, Any]], field: str) -> Counter:
    counts: Counter = Counter()
    for row in events:
        reference = set(row["reference"].get(field) or [])
        prediction = set(row["prediction"].get(field) or [])
        if not reference and not prediction:
            continue
        counts["tp"] += len(reference & prediction)
        counts["fp"] += len(prediction - reference)
        counts["fn"] += len(reference - prediction)
    return counts


def rates(counts: Counter, *, include_accuracy: bool = True) -> dict[str, float | int | None]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    # A nonempty but disjoint label set scores zero, not an unavailable result.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    result: dict[str, float | int | None] = {"n": counts.get("n", tp + fn), "precision": precision,
                                               "recall": recall, "f1": f1}
    if include_accuracy:
        result["accuracy"] = (counts["correct"] / counts["n"] if counts["n"] else None)
    return result
