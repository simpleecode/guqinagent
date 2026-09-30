"""Ornament metrics are token-set metrics, never surface-string metrics."""
from collections import Counter
import math

from .fingering import rates, set_counts


def ornament_metrics(events):
    return rates(set_counts(events, "ornaments"), include_accuracy=False)


def technique_usage(events, *, field="ornaments"):
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
        prediction_techniques = set(event["prediction"].get(field) or [])
        reference_techniques = set(event["reference"].get(field) or [])
        predicted.update(prediction_techniques)
        reference.update(reference_techniques)
        predicted_with_any += bool(prediction_techniques)
        reference_with_any += bool(reference_techniques)
    names = sorted(set(predicted) | set(reference))
    predicted_total = sum(predicted.values())
    reference_total = sum(reference.values())
    prediction_distribution = {
        name: predicted[name] / predicted_total if predicted_total else None
        for name in names
    }
    reference_distribution = {
        name: reference[name] / reference_total if reference_total else None
        for name in names
    }
    distribution_similarity = None
    if predicted_total and reference_total:
        distribution_similarity = 1 - 0.5 * sum(
            abs(prediction_distribution[name] - reference_distribution[name])
            for name in names
        )

    # Effective vocabulary size: exp(Shannon entropy), a length-normalized
    # alternative to raw distinct-name count. The denominator here is total
    # technique incidences; raw per-event density is reported separately.
    def diversity(counter, event_total):
        total = sum(counter.values())
        if not total:
            return {"unique_techniques": len(counter), "entropy_nats": None,
                    "effective_techniques": None}
        entropy = -sum((count / total) * math.log(count / total)
                       for count in counter.values())
        effective = math.exp(entropy)
        return {"unique_techniques": len(counter), "entropy_nats": entropy,
                "effective_techniques": effective}

    return {
        "events": event_count,
        "distribution_similarity": distribution_similarity,
        "distribution_similarity_definition": "1 - total_variation_distance",
        "prediction_distribution": prediction_distribution,
        "reference_distribution": reference_distribution,
        "technique_diversity": {
            "prediction": diversity(predicted, event_count),
            "reference": diversity(reference, event_count),
        },
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
