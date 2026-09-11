from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .abc_parser import parse_abc
from .auditor import audit_plan
from .compiler import compile_plan
from .pitch_candidates import generate_candidates
from .route_planner import (
    merge_phrase_routes, merge_phrase_routes_report, routes_to_plan, top_k_routes,
)
from .tuning_planner import decide_tuning, rank_tunings


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """A versioned deterministic capability exposed to graph agents."""

    name: str
    version: str
    handler: Callable[..., Any]
    description: str


class SkillRegistry:
    """Explicit registry that prevents agents from calling arbitrary code."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            raise ValueError(f"duplicate skill: {spec.name}")
        self._skills[spec.name] = spec

    def call(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        try:
            spec = self._skills[name]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {name}") from exc
        return spec.handler(*args, **kwargs)

    def versions(self) -> dict[str, str]:
        return {name: spec.version for name, spec in sorted(self._skills.items())}

    def describe(self) -> list[dict[str, str]]:
        return [
            {"name": spec.name, "version": spec.version, "description": spec.description}
            for spec in sorted(self._skills.values(), key=lambda item: item.name)
        ]


def default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    for spec in (
        SkillSpec("abc.parse", "1.0.0", parse_abc, "Parse ABC into CanonicalScore."),
        SkillSpec(
            "guqin.rank_tunings", "1.0.0", rank_tunings,
            "Rank common tunqin tunings for the current melody.",
        ),
        SkillSpec(
            "guqin.decide_tuning", "1.0.0", decide_tuning,
            "Decide whether retuning is justified before selecting an alternative.",
        ),
        SkillSpec(
            "guqin.pitch_candidates", "1.0.0", generate_candidates,
            "Enumerate playable guqin positions for score events.",
        ),
        SkillSpec(
            "guqin.phrase_routes", "1.0.0", top_k_routes,
            "Generate top-k fingering routes for one phrase.",
        ),
        SkillSpec(
            "guqin.merge_routes", "1.0.0", merge_phrase_routes,
            "Choose phrase routes with boundary costs.",
        ),
        SkillSpec(
            "guqin.merge_routes_report", "1.0.0", merge_phrase_routes_report,
            "Choose phrase routes and report local and boundary costs.",
        ),
        SkillSpec(
            "guqin.routes_to_plan", "1.0.0", routes_to_plan,
            "Convert selected routes to a baseline performance plan.",
        ),
        SkillSpec(
            "jianzipu.compile", "1.0.0", compile_plan,
            "Compile the performance plan into auditable jianzipu IR.",
        ),
        SkillSpec(
            "jianzipu.audit", "1.0.0", audit_plan,
            "Check pitch, coverage, and compiler completeness.",
        ),
    ):
        registry.register(spec)
    return registry
