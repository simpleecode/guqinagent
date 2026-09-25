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
    # ``events`` is the unified, source-indexed note population used by this
    # metric.  Counting sets here makes both the per-technique rates and the
    # density denominator consistently "valid note events", rather than text
    # occurrences.
    event_count = len(events)
    predicted_with_any = 0
    reference_with_any = 0
    for event in events:
        prediction_techniques = set(event["prediction"].get("ornaments") or [])
        reference_techniques = set(event["reference"].get("ornaments") or [])
        predicted.update(prediction_techniques)
        reference.update(reference_techniques)
        predicted_with_any += bool(prediction_techniques)
        reference_with_any += bool(reference_techniques)
    names = sorted(set(predicted) | set(reference))
    return {
        "events": event_count,
        "by_technique": {
            name: {
                "prediction_events": predicted[name],
                "prediction_rate": predicted[name] / event_count if event_count else None,
                "reference_events": reference[name],
                "reference_rate": reference[name] / event_count if event_count else None,
                "rate_delta": ((predicted[name] - reference[name]) / event_count
                               if event_count else None),
            }
            for name in names
        },
        "technique_density": {
            "prediction_events": predicted_with_any,
            "reference_events": reference_with_any,
            "prediction_rate": predicted_with_any / event_count if event_count else None,
            "reference_rate": reference_with_any / event_count if event_count else None,
            "rate_delta": ((predicted_with_any - reference_with_any) / event_count
                           if event_count else None),
        },
    }
