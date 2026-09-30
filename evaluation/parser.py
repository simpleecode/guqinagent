"""Prediction/reference adapters to a single structured event schema."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from agents.abc_to_jianzipu.reference_parser import AUDIT, parse_reference_actions

from .pitch import audit_plan, target_midis
from .techniques import extract_fingering_techniques


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
            "prediction": _event_view(pred, note.get("jianzi")),
            "reference": _event_view(ref, ref.get("jianzi_text") or ref.get("text")),
            "prediction_text": str(note.get("jianzi") or ""),
            "reference_text": str(ref.get("jianzi_text") or ref.get("text") or ""),
            "pitch_audit": detail,
        })
    return events, pred_audit


def structured_events_for_score(samples: list[tuple[dict[str, Any], dict[str, Any]]],
                                tolerance_cents: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate phrase predictions after replaying the complete score.

    A phrase-local parse loses inherited ``散`` and ``泛起`` state at its
    first event.  Assemble the phrases in source-index order and pass the
    complete prediction and annotation through the same audit/parser state
    machine before selecting the events belonging to this evaluation run.
    """
    ordered = sorted(samples, key=lambda pair: min(
        (int(note["index"]) for note in pair[1]["runtime_item"]["input"].get("notes_without_jianzi") or []
         if note.get("index") is not None), default=-1
    ))
    if not ordered:
        return [], {"details": []}

    pred_notes: list[dict[str, Any]] = []
    ref_notes: list[dict[str, Any]] = []
    reference_text_by_index: dict[int, str] = {}
    event_owner: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    metadata = ordered[0][1]["runtime_item"]["input"]["metadata"]
    open_midi = (ordered[0][1]["runtime_item"]["input"].get("normalized_tuning") or {}).get("open_midi")

    for prediction, reference in ordered:
        runtime = reference["runtime_item"]
        input_data = runtime["input"]
        predicted_actions = {
            int(action["source_index"]): str(action.get("jianzi_text") or action.get("text") or "")
            for action in final_actions(prediction) if action.get("source_index") is not None
        }
        reference_actions = {
            int(action["source_index"]): str(action.get("jianzi_text") or action.get("text") or "")
            for action in reference.get("reference", {}).get("actions") or []
            if action.get("source_index") is not None
        }
        for note in input_data.get("notes_without_jianzi") or []:
            if note.get("index") is None:
                continue
            index = int(note["index"])
            pred_note, ref_note = dict(note), dict(note)
            pred_note["jianzi"] = predicted_actions.get(index, "")
            ref_note["jianzi"] = reference_actions.get(index, "")
            pred_notes.append(pred_note)
            ref_notes.append(ref_note)
            reference_text_by_index[index] = ref_note["jianzi"]
            event_owner[index] = (prediction, reference)

    pred_notes.sort(key=lambda note: int(note["index"]))
    ref_notes.sort(key=lambda note: int(note["index"]))
    pred_data = {"metadata": metadata, "open_midi": open_midi, "notes": pred_notes}
    ref_data = {"metadata": metadata, "open_midi": open_midi, "notes": ref_notes}
    pred_audit = audit_plan(pred_data, tolerance_cents)
    ref_audit = audit_plan(ref_data, tolerance_cents)
    pred_semantics = _semantic_action_map(pred_data, pred_audit)
    ref_semantics = _semantic_action_map(ref_data, ref_audit)
    audit_by_index = {int(row["index"]): row for row in pred_audit["details"]
                      if row.get("index") is not None}

    events = []
    for note in pred_notes:
        source_index = int(note["index"])
        prediction, reference = event_owner[source_index]
        expected = target_midis(note, metadata)
        pred, ref = pred_semantics.get(source_index, {}), ref_semantics.get(source_index, {})
        if not expected and not pred and not ref:
            continue
        events.append({
            "piece_id": prediction.get("score_key") or reference["runtime_item"].get("score_key"),
            "phrase_id": prediction.get("phrase_id") or reference["runtime_item"].get("phrase_id"),
            "event_id": source_index,
            "target_midi": expected or None,
            "prediction": _event_view(pred, note.get("jianzi")),
            "reference": _event_view(ref, reference_text_by_index.get(source_index, "")),
            "prediction_text": str(note.get("jianzi") or ""),
            "reference_text": reference_text_by_index.get(source_index, ""),
            "pitch_audit": audit_by_index.get(source_index, {}),
        })
    return events, pred_audit


def _event_view(row: dict[str, Any], notation_text: Any = "") -> dict[str, Any]:
    return {
        "string": row.get("string"), "hui": row.get("hui"),
        "left_finger": row.get("left_finger"), "right_finger": row.get("right_finger"),
        "tone_type": row.get("mode"), "ornaments": list(row.get("techniques") or []),
        "pre_attack_ornaments": list(row.get("pre_attack_techniques") or []),
        "fingering_techniques": [item["name"] for item in
                                 extract_fingering_techniques(notation_text)],
        "compound_gesture": row.get("compound_gesture"), "attack": row.get("attack"),
        "inherited_fields": list(row.get("inherited_fields") or []),
    }
