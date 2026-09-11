from __future__ import annotations

import importlib.util
from pathlib import Path

from .models import CanonicalScore, Phrase, ScoreEvent
from .pitch_candidates import generate_candidates
from .route_planner import routes_to_plan, top_k_routes


ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"
_SPEC = importlib.util.spec_from_file_location("inverse_baseline_audit", AUDIT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(AUDIT)

DURATION_TICKS = {
    "六十四分": 60, "三十二分": 120, "十六分": 240, "八分": 480,
    "四分": 960, "二分": 1920, "全音": 3840,
}


def _ticks(text: str) -> int:
    return next((value for label, value in DURATION_TICKS.items() if label in text), 960)


def build_phrase_baseline(data: dict, start: int, end: int, *,
                          tolerance_cents: float = 50.0) -> dict:
    """Build an answer-blind stopped-note route from numbered notation only."""
    metadata = data["metadata"]
    tonic_midi = AUDIT.parse_tonic_midi(metadata)
    open_midi = AUDIT.parse_open_midi(metadata)
    source_notes = data.get("notes", [])[start:end]
    events: list[ScoreEvent] = []
    source_index_by_event: dict[str, int] = {}
    onset = 0
    bar = 1
    for note in source_notes:
        abc = str(note.get("abc", ""))
        if abc.strip() == "|":
            bar += 1
            continue
        pitches = [
            value for value in (
                AUDIT.parse_jianpu(note.get("jianpu"), tonic_midi),
                AUDIT.parse_jianpu(note.get("jianpu_alt"), tonic_midi),
            ) if value is not None
        ]
        duration = _ticks(str(note.get("duration", "")))
        if not pitches:
            onset += duration
            continue
        event_id = f"src-{int(note['index']):05d}"
        events.append(ScoreEvent(
            id=event_id, kind="note", abc=abc,
            pitches_midi=[round(max(pitches))], onset_ticks=onset,
            duration_ticks=duration, bar=bar, beat=1.0, attack=True,
            section_id="sec-1", phrase_id="p0001",
        ))
        source_index_by_event[event_id] = int(note["index"])
        onset += duration
    score = CanonicalScore(
        title=str(metadata.get("score_title", "")), key=str(metadata.get("tonic", "")),
        meter=str(metadata.get("meter", "1/4")), unit_note_length="1/16",
        tempo=metadata.get("tempo"), tuning_name=str(metadata.get("tuning", {}).get("name", "")),
        open_midi=open_midi, sections=[],
        phrases=[Phrase("p0001", "sec-1", [event.id for event in events],
                        events[0].bar if events else 1, events[-1].bar if events else 1)],
        events=events, source_abc="",
    )
    if not events:
        return {"actions": [], "diagnostics": ["no_sounding_events"]}
    candidates = generate_candidates(
        score, tolerance_cents=tolerance_cents, modes=("stopped",),
        max_candidates=14,
    )
    routes = top_k_routes(score, "p0001", candidates, top_k=1, states_per_candidate=3)
    if not routes:
        missing = [event.id for event in events if not candidates.get(event.id)]
        return {"actions": [], "diagnostics": ["baseline_route_missing", *missing]}
    candidate_index = {
        candidate.candidate_id: candidate
        for items in candidates.values() for candidate in items
    }
    plan = routes_to_plan(routes[:1], candidate_index)
    actions = []
    omitted_by_index = {
        int(note["index"]): bool(note.get("notation_omitted"))
        for note in source_notes if note.get("index") is not None
    }
    for action in plan.actions:
        event_id = action.source_event_ids[0]
        source_index = source_index_by_event[event_id]
        actions.append({
            "source_index": source_index, "mode": action.mode, "string": action.string,
            "hui": action.hui, "left_finger": action.left_finger,
            "right_finger": action.right_finger,
            "compound_gesture": None,
            "techniques": [], "pre_attack_techniques": [], "attack": action.attack,
            "notation_omitted": omitted_by_index.get(source_index, False),
            "jianzi_text": None,
            "evidence": list(action.rule_evidence),
        })
    return {
        "actions": actions,
        "route": routes[0].to_dict(),
        "candidate_counts": {event_id: len(items) for event_id, items in candidates.items()},
        "diagnostics": list(plan.diagnostics),
    }
