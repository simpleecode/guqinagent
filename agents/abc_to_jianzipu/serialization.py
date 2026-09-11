from __future__ import annotations

from .models import (
    CanonicalScore,
    PerformanceAction,
    PerformancePlan,
    Phrase,
    PositionCandidate,
    Route,
    ScoreEvent,
    Section,
)


def score_from_dict(data: dict) -> CanonicalScore:
    return CanonicalScore(
        title=data["title"],
        key=data["key"],
        meter=data["meter"],
        unit_note_length=data["unit_note_length"],
        tempo=data.get("tempo"),
        tuning_name=data["tuning_name"],
        open_midi=list(data["open_midi"]),
        sections=[Section(**item) for item in data.get("sections", [])],
        phrases=[Phrase(**item) for item in data.get("phrases", [])],
        events=[ScoreEvent(**item) for item in data.get("events", [])],
        source_abc=data.get("source_abc", ""),
        diagnostics=list(data.get("diagnostics", [])),
        schema_version=data.get("schema_version", "1.0"),
    )


def candidate_from_dict(data: dict) -> PositionCandidate:
    return PositionCandidate(**data)


def route_from_dict(data: dict) -> Route:
    return Route(**data)


def plan_from_dict(data: dict) -> PerformancePlan:
    return PerformancePlan(
        actions=[PerformanceAction(**item) for item in data.get("actions", [])],
        selected_routes=[route_from_dict(item) for item in data.get("selected_routes", [])],
        total_cost=float(data.get("total_cost", 0.0)),
        diagnostics=list(data.get("diagnostics", [])),
        schema_version=data.get("schema_version", "1.0"),
    )
