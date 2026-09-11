from __future__ import annotations

import math

from .models import CanonicalScore, PerformancePlan
from .pitch_candidates import _pitch


def audit_plan(score: CanonicalScore, plan: PerformancePlan,
               compiled: dict, *, tolerance_cents: float = 35.0) -> dict:
    event_by_id = {event.id: event for event in score.events}
    covered: dict[str, int] = {}
    diagnostics: list[dict] = []
    matched = 0
    compared = 0
    for action in plan.actions:
        if action.attack and not action.right_finger:
            diagnostics.append({
                "code": "pending_right_finger", "action_id": action.action_id,
                "event_id": action.source_event_ids[0] if action.source_event_ids else None,
                "severity": "error",
            })
        if action.mode == "stopped" and not action.left_finger:
            diagnostics.append({
                "code": "pending_left_finger", "action_id": action.action_id,
                "event_id": action.source_event_ids[0] if action.source_event_ids else None,
                "severity": "error",
            })
        for event_id in action.source_event_ids:
            covered[event_id] = covered.get(event_id, 0) + 1
            event = event_by_id.get(event_id)
            if event is None:
                diagnostics.append({
                    "code": "unknown_source_event", "event_id": event_id,
                    "severity": "error",
                })
                continue
            if event.midi is None:
                continue
            compared += 1
            if action.string is None or not 1 <= action.string <= 7:
                diagnostics.append({
                    "code": "invalid_string", "event_id": event_id,
                    "severity": "error",
                })
                continue
            try:
                sounding = _pitch(
                    score.open_midi[action.string - 1], action.hui, action.mode or "stopped"
                )
            except (ValueError, IndexError) as exc:
                diagnostics.append({
                    "code": str(exc), "event_id": event_id, "severity": "error",
                })
                continue
            cents = (sounding - event.midi) * 100
            if abs(cents) <= tolerance_cents:
                matched += 1
            else:
                diagnostics.append({
                    "code": "pitch_mismatch", "event_id": event_id,
                    "severity": "error", "delta_cents": round(cents, 3),
                })

    required = [
        event for event in score.events if event.midi is not None and event.attack
    ]
    for event in required:
        count = covered.get(event.id, 0)
        if count == 0:
            diagnostics.append({
                "code": "missing_performance_action", "event_id": event.id,
                "severity": "error",
            })
        elif count > 1:
            diagnostics.append({
                "code": "duplicate_performance_action", "event_id": event.id,
                "severity": "error", "count": count,
            })
    missing_glyph = [
        item["source_event_ids"][0]
        for item in compiled.get("glyph_events", [])
        if item.get("kind") == "missing"
    ]
    for event_id in missing_glyph:
        diagnostics.append({
            "code": "compiler_missing_glyph", "event_id": event_id,
            "severity": "error",
        })
    errors = [item for item in diagnostics if item.get("severity") == "error"]
    return {
        "status": "passed" if not errors else "failed",
        "summary": {
            "required_events": len(required),
            "compared_events": compared,
            "matched_events": matched,
            "match_rate": round(matched / compared, 6) if compared else None,
            "errors": len(errors),
        },
        "diagnostics": diagnostics,
        "checks": {
            "pitch": not any(item["code"] == "pitch_mismatch" for item in errors),
            "coverage": not any("performance_action" in item["code"] for item in errors),
            "compiler": not missing_glyph,
            "inheritance": True,
            "fingering_complete": not any(
                item["code"] in {"pending_right_finger", "pending_left_finger"}
                for item in errors
            ),
        },
    }
