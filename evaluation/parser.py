"""Prediction/reference adapters to a single structured event schema."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agents.abc_to_jianzipu.reference_parser import AUDIT, parse_reference_actions

from .pitch import audit_plan, target_midis


def final_actions(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer the final Guqinizer plan; support one-stage ``jianzi_rows`` too."""
    stage = prediction.get("guqinizer") or prediction.get("base") or {}
    actions = ((stage.get("plan") or {}).get("actions") or [])
    if actions:
        return actions
    return [{"source_index": row[0], "jianzi_text": row[1]}
            for row in prediction.get("jianzi_rows") or [] if len(row) >= 2]


def _notes_for_plan(runtime_item: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    payload = deepcopy(runtime_item["input"])
    by_index = {int(action["source_index"]): str(action.get("jianzi_text") or action.get("text") or "")
                for action in actions if action.get("source_index") is not None}
    notes = []
    for note in payload.get("notes_without_jianzi") or []:
        copied = dict(note)
        if copied.get("index") is not None:
            copied["jianzi"] = by_index.get(int(copied["index"]), "")
        notes.append(copied)
    return {"metadata": payload["metadata"], "open_midi":
            (payload.get("normalized_tuning") or {}).get("open_midi"), "notes": notes}


def _semantic_action_map(plan_data: dict[str, Any], audit: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Use the existing state-aware reference parser for both sides."""
    return {int(row.source_index): row.to_dict()
            for row in parse_reference_actions(plan_data, audit)}


def structured_events(prediction: dict[str, Any], reference: dict[str, Any],
                      tolerance_cents: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = reference["runtime_item"]
    pred_data = _notes_for_plan(runtime, final_actions(prediction))
    pred_audit = audit_plan(pred_data, tolerance_cents)
    pred_semantics = _semantic_action_map(pred_data, pred_audit)
    ref_semantics = {int(row["source_index"]): dict(row)
                     for row in reference.get("reference", {}).get("actions") or []}
    audit_by_index = {int(row["index"]): row for row in pred_audit["details"]
                      if row.get("index") is not None}
    events = []
    for note in pred_data["notes"]:
        if note.get("index") is None:
            continue
        source_index = int(note["index"])
        expected = target_midis(note, pred_data["metadata"])
        if not expected and source_index not in ref_semantics and source_index not in pred_semantics:
            continue
        pred = pred_semantics.get(source_index, {})
        ref = ref_semantics.get(source_index, {})
        detail = audit_by_index.get(source_index, {})
        events.append({
            "piece_id": prediction.get("score_key") or runtime.get("score_key"),
            "phrase_id": prediction.get("phrase_id") or runtime.get("phrase_id"),
            "event_id": source_index,
            "target_midi": expected or None,
            "prediction": _event_view(pred),
            "reference": _event_view(ref),
            "prediction_text": str(note.get("jianzi") or ""),
            "reference_text": str(ref.get("jianzi_text") or ref.get("text") or ""),
            "pitch_audit": detail,
        })
    return events, pred_audit


def _event_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "string": row.get("string"), "hui": row.get("hui"),
        "left_finger": row.get("left_finger"), "right_finger": row.get("right_finger"),
        "tone_type": row.get("mode"), "ornaments": list(row.get("techniques") or []),
        "pre_attack_ornaments": list(row.get("pre_attack_techniques") or []),
        "compound_gesture": row.get("compound_gesture"), "attack": row.get("attack"),
        "inherited_fields": list(row.get("inherited_fields") or []),
    }

