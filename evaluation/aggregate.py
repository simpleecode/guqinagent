"""Metric aggregation for structured GuqinAgent evaluation events."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .fingering import field_counts, rates
from .ornament import ornament_metrics, technique_usage
from .pitch import paired_cents


TONE_TYPES = ("stopped", "open", "harmonic")


def final_warning_stats(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Count phrases whose final edit_plan preview still contains warnings."""
    total = 0
    warned = 0
    for prediction in predictions:
        total += 1
        stage = prediction.get("guqinizer") or prediction.get("base") or {}
        last_text = ""
        for trace_item in stage.get("trace") or []:
            for result in trace_item.get("tool_results") or []:
                if result.get("name") == "edit_plan":
                    payload = result.get("result") or {}
                    if isinstance(payload, dict):
                        nested = payload.get("result") if isinstance(payload.get("result"), dict) else payload
                        last_text = str(nested.get("text") or "")
        if ":warning:" in last_text:
            warned += 1
    return {
        "phrases": total,
        "warning_phrases": warned,
        "warning_rate": warned / total if total else None,
        "definition": "phrases whose final edit_plan preview contains at least one :warning:",
    }


def tone_type_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare replay-derived 按/散/泛 state and its corpus distribution."""
    comparable = [event for event in events
                  if event["reference"].get("tone_type") in TONE_TYPES]
    accuracy = rates(field_counts(comparable, "tone_type"))
    reference_counts = Counter(event["reference"]["tone_type"] for event in comparable)
    prediction_counts = Counter(event["prediction"].get("tone_type") for event in comparable
                                if event["prediction"].get("tone_type") in TONE_TYPES)
    total_reference = sum(reference_counts.values())
    total_prediction = sum(prediction_counts.values())
    reference_distribution = {
        mode: reference_counts[mode] / total_reference if total_reference else None
        for mode in TONE_TYPES
    }
    prediction_distribution = {
        mode: prediction_counts[mode] / total_prediction if total_prediction else None
        for mode in TONE_TYPES
    }
    similarity = None
    if total_reference and total_prediction:
        similarity = 1 - 0.5 * sum(
            abs(reference_distribution[mode] - prediction_distribution[mode])
            for mode in TONE_TYPES
        )
    return {
        **accuracy,
        "reference_counts": dict(reference_counts),
        "prediction_counts": dict(prediction_counts),
        "reference_distribution": reference_distribution,
        "prediction_distribution": prediction_distribution,
        "distribution_similarity": similarity,
        "distribution_similarity_definition": "1 - total_variation_distance",
    }


def metrics(events: list[dict[str, Any]], violations: list[dict[str, Any]],
            tolerance_cents: float) -> dict[str, Any]:
    cents = [value for event in events for value in paired_cents(event.get("pitch_audit") or {})]
    pitch_events = [event for event in events
                    if (event.get("pitch_audit") or {}).get("status") in {"matched", "mismatched"}]
    matched = sum((event["pitch_audit"] or {}).get("status") == "matched" for event in pitch_events)
    result = {
        "pitch_accuracy_at_50_cents": matched / len(pitch_events) if pitch_events else None,
        "pitch_mae_cents": sum(cents) / len(cents) if cents else None,
        "pitch_evaluable_events": len(pitch_events),
        "pitch_evaluable_pairs": len(cents),
        "string": rates(field_counts(events, "string")),
        "left_hand_fingering": rates(field_counts(events, "left_finger")),
        "right_hand_fingering": rates(field_counts(events, "right_finger")),
        "tone_type": tone_type_metrics(events),
        "ornament": ornament_metrics(events),
        # One paper-facing technique statistic: count named, playable actions
        # from final notation.  This includes 撮 exactly once, whether it also
        # has an ornament-like semantic role, and excludes state markers such
        # as 泛起/泛止.  Ornament P/R/F1 above remains an event-level semantic
        # correctness metric, not a competing technique-distribution metric.
        "performance_technique_usage": technique_usage(
            events, field="fingering_techniques"
        ),
        "rule_violation_rate": len({(v["piece_id"], v["phrase_id"], v["event_id"]) for v in violations}) /
                               len(pitch_events) if pitch_events else None,
        "rule_violations": len(violations),
        "pitch_tolerance_cents": tolerance_cents,
    }
    return result


def per_piece(events: list[dict[str, Any]], violations: list[dict[str, Any]],
              tolerance_cents: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault((event["piece_id"], event["phrase_id"]), []).append(event)
    result = []
    for (piece_id, phrase_id), group in sorted(grouped.items()):
        own = [v for v in violations if v["piece_id"] == piece_id and v["phrase_id"] == phrase_id]
        result.append({"piece_id": piece_id, "phrase_id": phrase_id,
                       "metrics": metrics(group, own, tolerance_cents)})
    return result
