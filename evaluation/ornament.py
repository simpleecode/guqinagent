"""Ornament metrics are token-set metrics, never surface-string metrics."""
from collections import Counter

from .fingering import rates, set_counts


def ornament_metrics(events):
    return rates(set_counts(events, "ornaments"), include_accuracy=False)


def technique_usage(events):
    """Per-technique event incidence for prediction and reference.

    One technique is counted at most once per event: ``吟吟`` in malformed
    source text must not inflate its usage compared with a normal ``吟``.
    """
    predicted, reference = Counter(), Counter()
    event_count = len(events)
    for event in events:
        predicted.update(set(event["prediction"].get("ornaments") or []))
        reference.update(set(event["reference"].get("ornaments") or []))
    names = sorted(set(predicted) | set(reference))
    return {
        "events": event_count,
        "by_technique": {
            name: {
                "prediction_events": predicted[name],
                "prediction_rate": predicted[name] / event_count if event_count else None,
                "reference_events": reference[name],
                "reference_rate": reference[name] / event_count if event_count else None,
                "event_count_delta": predicted[name] - reference[name],
            }
            for name in names
        },
    }
