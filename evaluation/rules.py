"""Only deterministic, parser-backed final-plan violations."""
from __future__ import annotations

from typing import Any


PARSER_FAILURES = {
    "string_out_of_range", "missing_hui", "harmonic_hui_missing",
    "hui_outside_not_valid_for_harmonic", "fractional_hui_not_valid_for_harmonic",
    "ambiguous_multiple_strings", "current_stopped_position_missing",
}


def violations(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for event in events:
        audit = event.get("pitch_audit") or {}
        reason = audit.get("reason")
        if audit.get("status") == "mismatched":
            result.append(_row(event, "pitch_mismatch", audit,
                               "可解析减字的实得音高与该事件简谱目标相差超过容差。"))
        elif reason in PARSER_FAILURES:
            result.append(_row(event, "unplayable_or_unresolvable_jianzi", audit,
                               "减字触发确定性的解析/演奏状态错误。"))
    return result


def _row(event: dict[str, Any], kind: str, audit: dict[str, Any], explanation: str) -> dict[str, Any]:
    return {"piece_id": event["piece_id"], "phrase_id": event["phrase_id"],
            "event_id": event["event_id"], "violation_type": kind,
            "relevant_state_event": {"prediction_text": event["prediction_text"],
                                     "target_midi": event["target_midi"], "audit": audit},
            "explanation": explanation}
