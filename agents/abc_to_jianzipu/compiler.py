from __future__ import annotations

from .jianzi_renderer import hui_label, render_jianzi_text
from .models import CanonicalScore, PerformancePlan


def compile_plan(score: CanonicalScore, plan: PerformancePlan) -> dict:
    action_by_event = {
        source_id: action
        for action in plan.actions
        for source_id in action.source_event_ids
    }
    section_by_event = {
        section.start_event: section for section in score.sections if section.start_event
    }
    glyph_events = []
    tokens: list[str] = []
    for event in score.events:
        section = section_by_event.get(event.id)
        if section:
            tokens.append(section.marker)
        action = action_by_event.get(event.id)
        if event.kind == "rest":
            tokens.append("[—]")
            glyph_events.append({
                "source_event_ids": [event.id], "kind": "rest", "text": "—",
            })
            continue
        if action is None:
            tokens.append("[?]")
            glyph_events.append({
                "source_event_ids": [event.id], "kind": "missing", "text": "?",
                "diagnostics": ["missing_performance_action"],
            })
            continue
        pending = []
        if action.attack and not action.right_finger:
            pending.append("右手")
        if action.mode == "stopped" and not action.left_finger:
            pending.append("左手")
        if pending:
            position = (
                f"{action.string}弦{hui_label(action.hui)}"
                if action.mode != "open" else f"散音{action.string}弦"
            )
            text = f"待定：{'/'.join(pending)}；{position}"
        else:
            text = render_jianzi_text(
                action, omitted_placeholder="无（由于是再作部分，省略）"
            )
        if text:
            tokens.append(f"[{text}]")
        glyph_events.append({
            "source_event_ids": list(action.source_event_ids),
            "kind": "pending" if pending else action.kind,
            "text": text,
            "components": {
                "mode": action.mode,
                "left_finger": action.left_finger,
                "hui": action.hui,
                "right_finger": action.right_finger,
                "string": action.string,
                "attack": action.attack,
                "attack_source_id": action.attack_source_id,
                "techniques": list(action.techniques),
                "pre_attack_techniques": list(action.pre_attack_techniques),
                "jianzi_text": action.jianzi_text,
            },
            "inheritance": {
                "left_position": "pending" if action.mode == "stopped" and not action.left_finger else "explicit",
                "right_finger": "pending" if action.attack and not action.right_finger else "explicit",
                "mode": "explicit",
            },
            "diagnostics": [
                f"pending_{field}_fingering" for field in pending
            ],
        })
    lines = [" ".join(tokens[start:start + 16]) for start in range(0, len(tokens), 16)]
    return {
        "schema_version": "1.0",
        "glyph_events": glyph_events,
        "jianzipu": "\n".join(lines),
    }
