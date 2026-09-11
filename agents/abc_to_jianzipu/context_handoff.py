from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


BOUNDARY_FIELDS = (
    "mode", "string", "hui", "left_finger", "right_finger", "attack",
)


def action_state(action: dict[str, Any] | None) -> dict[str, Any]:
    if not action:
        return {
            "sound_mode": None, "active_string": None, "active_hui": None,
            "left_finger": None, "right_finger": None, "attack": None,
            "techniques": [], "source_index": None,
        }
    return {
        "sound_mode": action.get("mode"), "active_string": action.get("string"),
        "active_hui": action.get("hui"), "left_finger": action.get("left_finger"),
        "right_finger": action.get("right_finger"), "attack": action.get("attack"),
        "techniques": list(action.get("techniques") or []),
        "source_index": action.get("source_index"),
    }


def boundary_state(actions: Iterable[dict[str, Any]], *, phrase_id: str | None = None,
                   confidence: str = "generated") -> dict[str, Any]:
    ordered = sorted(actions, key=lambda action: int(action.get("source_index", -1)))
    sounding = [action for action in ordered if action.get("attack")]
    last = sounding[-1] if sounding else (ordered[-1] if ordered else None)
    state = action_state(last)
    state.update({
        "source_phrase": phrase_id, "confidence": confidence,
        "harmonic_scope": state["sound_mode"] == "harmonic",
        "resonance_active": bool(last and (last.get("attack") or last.get("techniques"))),
        "pending_slide": next((technique for technique in reversed(state["techniques"])
                               if technique in {"上", "下", "进", "退", "绰", "注"}), None),
    })
    return state


def compact_note(note: dict[str, Any]) -> dict[str, Any]:
    compact = {key: note.get(key) for key in
               ("index", "event_index", "abc", "jianpu", "jianpu_alt", "duration", "lyric", "section")}
    # 再作物化的省略标记必须随音符传递，否则渲染层无法给出占位减字。
    if note.get("notation_omitted"):
        compact["notation_omitted"] = True
    return compact


def _note_line(note: dict[str, Any]) -> str:
    index = note.get("index", note.get("id", "?"))
    pitch = note.get("jianpu") or note.get("abc") or "休止"
    duration = note.get("duration") or note.get("duration_ticks") or "?"
    return f"{index}｜{pitch}｜{duration}"


def _action_line(action: dict[str, Any]) -> str:
    source = action.get("source_index") or ",".join(action.get("source_event_ids") or []) or "?"
    mode = {"open": "散音", "harmonic": "泛音", "stopped": "按音"}.get(
        action.get("mode"), action.get("mode") or "待定"
    )
    position = f'{action.get("string", "?")}弦'
    if action.get("hui") is not None:
        position += f'{action["hui"]}徽'
    fingers = "".join(filter(None, (action.get("left_finger"), action.get("right_finger"))))
    techniques = "+".join(action.get("techniques") or [])
    detail = " ".join(part for part in (mode, position, fingers, techniques) if part)
    return f"{source}｜{detail}"


def render_context_source(current_phrase: list[dict[str, Any]], *,
                          previous_phrase: dict[str, Any] | None = None,
                          older_context_refs: list[dict[str, str]] | None = None) -> str:
    lines: list[str] = []
    for item in older_context_refs or []:
        lines.append(
            f'… <更早段：{item["phrase_id"]}，'
            f'expand={item["context_ref"]}>'
        )
    if previous_phrase:
        lines.append(f'<前一段：{previous_phrase["phrase_id"]}，已确认，只读>')
        actions = previous_phrase.get("actions") or []
        source = actions if actions else previous_phrase.get("notes", [])
        renderer = _action_line if actions else _note_line
        lines.extend(renderer(item) for item in source)
        lines.append("</前一段>")
    lines.append("<当前段：规划中，可编辑>")
    lines.extend(_note_line(note) for note in current_phrase)
    lines.append("</当前段>")
    return "\n".join(lines)


def build_phrase_handoff(
    *, phrase_id: str, notes: list[dict[str, Any]], start: int, end: int,
    previous_exit_state: dict[str, Any] | None = None,
    previous_phrase: dict[str, Any] | None = None,
    older_context_refs: list[dict[str, str]] | None = None,
    next_phrase_id: str | None = None,
    context_notes: int = 3,
) -> dict[str, Any]:
    current = notes[start:end]
    current_section = (current[0].get("section") or {}) if current else {}
    previous_notes = (previous_phrase or {}).get("notes") or []
    previous_section = (previous_notes[-1].get("section") or {}) if previous_notes else {}
    section_boundary = bool(
        previous_phrase and current_section.get("number") != previous_section.get("number")
    )
    current_phrase = [compact_note(note) for note in current]
    handoff = {
        "schema_version": "phrase-handoff-2.0", "phrase_id": phrase_id,
        "next_phrase_id": next_phrase_id,
        "section": {
            "number": current_section.get("number"), "label": current_section.get("label"),
            "title": current_section.get("title"), "marker": current_section.get("marker"),
            "boundary_from_previous": section_boundary,
        },
        "current_phrase": current_phrase,
        "context_source": render_context_source(
            current_phrase, previous_phrase=previous_phrase,
            older_context_refs=older_context_refs,
        ),
        "constraints": {
            "previous_phrase_readonly": True,
            "only_current_phrase_is_editable": True,
            "section_boundary_does_not_reset_state": True,
        },
    }
    if previous_phrase:
        handoff["previous_phrase"] = deepcopy(previous_phrase)
    if older_context_refs:
        handoff["older_context_refs"] = deepcopy(older_context_refs)
    return handoff


def attach_inferred_handoffs(items: list[dict[str, Any]], *, context_notes: int = 3) -> list[dict[str, Any]]:
    """Attach sequential teacher-forced handoffs to one score's trajectories."""
    if not items:
        return items
    items = sorted(items, key=lambda item: item["input"]["event_range"]["start"])
    # Reconstruct the score-level public note stream from non-overlapping phrases.
    notes = []
    for item in items:
        notes.extend(item["input"]["notes_without_jianzi"])
    notes.sort(key=lambda note: int(note["index"]))
    position_by_index = {int(note["index"]): position for position, note in enumerate(notes)}
    previous_exit = None
    previous_item = None
    for position, item in enumerate(items):
        source_range = item["input"]["event_range"]
        phrase_notes = item["input"]["notes_without_jianzi"]
        start = position_by_index[int(phrase_notes[0]["index"])] if phrase_notes else 0
        end = start + len(phrase_notes)
        next_id = items[position + 1]["phrase_id"] if position + 1 < len(items) else None
        previous_phrase = None
        if previous_item is not None:
            previous_phrase = {
                "phrase_id": previous_item["phrase_id"],
                "status": "confirmed_readonly",
                "notes": deepcopy(previous_item["input"]["notes_without_jianzi"]),
                "actions": deepcopy(previous_item["reference_plan"]["actions"]),
            }
        older_refs = []
        if position > 1:
            older_refs = [
                {
                    "phrase_id": older["phrase_id"],
                    "context_ref": (
                        f'context://score/{item.get("score_key", "unknown")}/'
                        f'phrases/{older["phrase_id"]}'
                    ),
                }
                for older in items[:position - 1]
            ]
        handoff = build_phrase_handoff(
            phrase_id=item["phrase_id"], notes=notes, start=start, end=end,
            previous_exit_state=previous_exit, previous_phrase=previous_phrase,
            older_context_refs=older_refs, next_phrase_id=next_id,
            context_notes=context_notes,
        )
        # The preceding reference translation is legitimate teacher-forced
        # history, not an answer for the current phrase.
        references = item["reference_plan"]["actions"]
        exit_state = boundary_state(references, phrase_id=item["phrase_id"],
                                    confidence="verified")
        handoff["exit_state_target"] = exit_state
        handoff["event_range"] = deepcopy(source_range)
        item["input"]["phrase_handoff"] = handoff
        previous_exit = exit_state
        previous_item = item
    return items
