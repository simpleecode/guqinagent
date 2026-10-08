"""Per-phrase scoring on top of the sealed structured-event metrics.

Scoring protocol
----------------
A phrase is evaluated by replaying the phrase-spanning parse state through the
*full reference prefix* of the same score (every preceding phrase enters the
replay with prediction = reference, hence a perfect prefix) followed by the
candidate's prediction; only the current phrase's events are kept.  This
reuses ``structured_events_for_score`` so inherited 散/泛/走手 state is handled
by the same state machine as the sealed evaluation.

Residual reference noise
------------------------
Some train-half references are themselves pitch-inconsistent (the eligible
export tolerates up to ~50% unmatched rows).  ``data.py`` precomputes, per
phrase, the event ids the *reference itself* fails (``audit.pitch_mismatch_event_ids``
/ ``audit.violation_event_ids``).  Pitch and playability components — and the
mismatch feedback shown to the reflection model — are computed over the
remaining events only, so a prediction equal to the reference scores exactly
1.0 and reflection never chases unfixable events.  Tone/fingering/ornament
stay over all events (the reference is consistent there by construction).

The composite is a weight-renormalized average over the available components;
phrases with no evaluable component at all score a neutral 0.5.  Feedback is
deterministic and contains no reference notation text.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.aggregate import metrics  # noqa: E402
from evaluation.parser import structured_events_for_score  # noqa: E402
from evaluation.rules import violations  # noqa: E402

from baseline.gepa_guqin.compat import action_text  # noqa: E402

DEFAULT_WEIGHTS: dict[str, float] = {
    "pitch": 0.45,
    "tone_type": 0.15,
    "fingering": 0.15,
    "ornament": 0.10,
    "playability": 0.15,
}


def _row_order_key(row: dict[str, Any]):
    indexes = [int(note["index"]) for note in
               (row.get("input") or {}).get("notes_without_jianzi") or []
               if note.get("index") is not None]
    return (min(indexes) if indexes else -1, str(row.get("phrase_id") or ""))


def _reference_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "runtime_item": {"input": row["input"]},
        "reference": {"actions": row["reference_plan"]["actions"]},
    }


def _perfect_prediction(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_key": row.get("score_key"), "phrase_id": row.get("phrase_id"),
        "jianzi_rows": [
            [int(action["source_index"]), action_text(action)]
            for action in row["reference_plan"]["actions"]
            if action.get("source_index") is not None
        ],
    }


def _score_prefix(historical: dict[tuple[str, str], Any],
                  example: dict[str, Any]) -> list[dict[str, Any]]:
    """Reference rows of the same score strictly before the current phrase."""
    current_key = _row_order_key({
        "input": example["runtime_item"]["input"],
        "phrase_id": example["phrase_id"],
    })
    prefix = [row for row in historical.values()
              if str(row.get("score_key")) == str(example["score_key"])
              and _row_order_key(row) < current_key]
    prefix.sort(key=_row_order_key)
    return prefix


def _avg(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def replay_events(example: dict[str, Any], prediction: dict[str, Any],
                  historical: dict[tuple[str, str], Any], *,
                  tolerance_cents: float = 50.0):
    """Full-prefix replay events (current phrase only) + violations + audit."""
    samples = [(_perfect_prediction(row), _reference_row(row))
               for row in _score_prefix(historical, example)]
    reference_row = {
        "runtime_item": {"input": example["runtime_item"]["input"]},
        "reference": {"actions": example["reference"]["actions"]},
    }
    samples.append((prediction, reference_row))
    events, audit = structured_events_for_score(samples, tolerance_cents)
    events = [event for event in events
              if str(event.get("phrase_id")) == str(example["phrase_id"])]
    return events, violations(events), audit


def reference_self_audit(example: dict[str, Any],
                         historical: dict[tuple[str, str], Any], *,
                         tolerance_cents: float = 50.0) -> dict[str, Any]:
    """Events the sealed reference itself fails on this phrase.

    Used by ``data.py`` to precompute per-phrase exclusion sets so the metric
    never blames the model for reference-inherent pitch misses.
    """
    prediction = _perfect_prediction({
        "score_key": example["score_key"], "phrase_id": example["phrase_id"],
        "reference_plan": {"actions": example["reference"]["actions"]},
    })
    events, event_violations, _ = replay_events(
        example, prediction, historical, tolerance_cents=tolerance_cents)
    return {
        "pitch_mismatch_event_ids": sorted(
            int(event["event_id"]) for event in events
            if (event.get("pitch_audit") or {}).get("status") == "mismatched"),
        "violation_event_ids": sorted(
            {int(violation["event_id"]) for violation in event_violations}),
    }


def phrase_report(example: dict[str, Any], prediction: dict[str, Any],
                  historical: dict[tuple[str, str], Any], *,
                  tolerance_cents: float = 50.0,
                  weights: dict[str, float] | None = None,
                  warnings_text: str = "") -> dict[str, Any]:
    """Score one prepared example against its sealed reference actions."""
    weights = weights or DEFAULT_WEIGHTS
    events, event_violations, _ = replay_events(
        example, prediction, historical, tolerance_cents=tolerance_cents)

    audit_info = example.get("audit") or {}
    excluded = set(int(i) for i in audit_info.get("pitch_mismatch_event_ids") or [])
    excluded |= set(int(i) for i in audit_info.get("violation_event_ids") or [])
    clean_events = [event for event in events if int(event["event_id"]) not in excluded]
    clean_violations = [violation for violation in event_violations
                        if int(violation["event_id"]) not in excluded]

    summary = metrics(events, event_violations, tolerance_cents)
    clean_summary = metrics(clean_events, clean_violations, tolerance_cents)

    pitch = clean_summary["pitch_accuracy_at_50_cents"]
    tone = (summary["tone_type"] or {}).get("accuracy")
    left_f1 = (summary["left_hand_fingering"] or {}).get("f1")
    right_f1 = (summary["right_hand_fingering"] or {}).get("f1")
    fingering = _avg([left_f1, right_f1])
    ornament = (summary["ornament"] or {}).get("f1")
    violation_rate = clean_summary["rule_violation_rate"]
    playability = (1.0 - violation_rate) if violation_rate is not None else None

    components = {"pitch": pitch, "tone_type": tone, "fingering": fingering,
                  "ornament": ornament, "playability": playability}
    available = [(weights[name], value) for name, value in components.items()
                 if value is not None and weights.get(name)]
    total_weight = sum(weight for weight, _ in available) or 0.0
    score = (sum(weight * value for weight, value in available) / total_weight
             if total_weight else None)

    mismatched = sorted(int(event["event_id"]) for event in clean_events
                        if (event.get("pitch_audit") or {}).get("status") == "mismatched")
    tone_confused = sum(1 for event in events
                        if event["reference"].get("tone_type") in ("stopped", "open", "harmonic")
                        and event["prediction"].get("tone_type")
                        != event["reference"].get("tone_type"))
    left_missing = sum(1 for event in events
                       if event["reference"].get("left_finger")
                       and not event["prediction"].get("left_finger"))
    right_missing = sum(1 for event in events
                        if event["reference"].get("right_finger")
                        and not event["prediction"].get("right_finger"))
    pitch_evaluable = clean_summary.get("pitch_evaluable_events") or 0

    parts = []
    if pitch is not None:
        parts.append(f"音高匹配率 {pitch:.3f}（可评 {pitch_evaluable} 个音）"
                     + (f"；音高不匹配音序：{('、'.join(str(i) for i in mismatched))}"
                        if mismatched else ""))
    if tone is not None:
        parts.append(f"按/散/泛准确率 {tone:.3f}（{tone_confused} 个音色类型不一致）")
    if left_f1 is not None or right_f1 is not None:
        parts.append(f"左手指法 F1 {left_f1 if left_f1 is not None else 'N/A'}"
                     f"（缺失 {left_missing}）、右手指法 F1 {right_f1 if right_f1 is not None else 'N/A'}"
                     f"（缺失 {right_missing}）")
    if ornament is not None:
        parts.append(f"装饰技法集合 F1 {ornament:.3f}")
    if playability is not None:
        parts.append(f"可演奏性（1-违规率） {playability:.3f}（违规 {len(clean_violations)} 处）")
    if warnings_text:
        parts.append(f"最终预览警告：{warnings_text}")
    feedback = "；".join(parts) or "本段没有可评估的演奏事件"

    objective = {
        "pitch_accuracy": float(pitch) if pitch is not None else 0.0,
        "tone_type_accuracy": float(tone) if tone is not None else 0.0,
        "left_hand_f1": float(left_f1) if left_f1 is not None else 0.0,
        "right_hand_f1": float(right_f1) if right_f1 is not None else 0.0,
        "ornament_f1": float(ornament) if ornament is not None else 0.0,
        "playability": float(playability) if playability is not None else 0.0,
    }
    return {
        "score": score,
        "objective_scores": objective,
        "feedback": feedback,
        "metrics": {
            "events": len(events),
            "excluded_reference_failed_events": len(excluded),
            "components": components,
            "pitch_evaluable_events": pitch_evaluable,
            "rule_violations": len(clean_violations),
        },
    }
