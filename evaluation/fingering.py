"""Reusable structured-field and set metrics for parsed jianzipu events."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def _valid(value: Any) -> bool:
    return value is not None and value != ""


def field_counts(events: Iterable[dict[str, Any]], field: str,
                 *, hui_tolerance: float = 0.05) -> Counter:
    """Micro counts for one semantic field, omitting unavailable references."""
    counts: Counter = Counter()
    for row in events:
        ref, pred = row["reference"].get(field), row["prediction"].get(field)
        if not _valid(ref):
            continue
        counts["n"] += 1
        equal = (abs(float(ref) - float(pred)) <= hui_tolerance
                 if field == "hui" and _valid(pred) else ref == pred)
        counts["correct"] += int(equal)
        counts["tp"] += int(equal and _valid(pred))
        counts["fp"] += int(_valid(pred) and not equal)
        counts["fn"] += int(not equal)
    return counts


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
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    result: dict[str, float | int | None] = {"n": counts.get("n", tp + fn), "precision": precision,
                                               "recall": recall, "f1": f1}
    if include_accuracy:
        result["accuracy"] = (counts["correct"] / counts["n"] if counts["n"] else None)
    return result

