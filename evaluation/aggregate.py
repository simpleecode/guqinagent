"""Metric aggregation for structured GuqinAgent evaluation events."""
from __future__ import annotations

from collections import Counter
from typing import Any

from .fingering import field_counts, rates
from .ornament import ornament_metrics, technique_usage
from .pitch import paired_cents


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
        "hui": rates(field_counts(events, "hui")),
        "left_hand_fingering": rates(field_counts(events, "left_finger")),
        "right_hand_fingering": rates(field_counts(events, "right_finger")),
        "ornament": ornament_metrics(events),
        "technique_usage": technique_usage(events),
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
