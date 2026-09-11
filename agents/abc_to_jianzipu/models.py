from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class Section:
    id: str
    label: str
    title: str
    start_event: str | None = None
    marker: str = ""


@dataclass(slots=True)
class ScoreEvent:
    id: str
    kind: str
    abc: str
    pitches_midi: list[int]
    onset_ticks: int
    duration_ticks: int
    bar: int
    beat: float
    attack: bool = True
    section_id: str | None = None
    phrase_id: str | None = None
    lyric: str = ""
    diagnostics: list[str] = field(default_factory=list)

    @property
    def midi(self) -> int | None:
        """Return the melody pitch used by the monophonic MVP.

        Chords are preserved in ``pitches_midi``.  Until a multi-voice planner
        is added, the highest pitch is treated as the melody and the reduction
        is recorded as a diagnostic by the parser.
        """
        return max(self.pitches_midi) if self.pitches_midi else None


@dataclass(slots=True)
class Phrase:
    id: str
    section_id: str | None
    event_ids: list[str]
    start_bar: int
    end_bar: int


@dataclass(slots=True)
class CanonicalScore:
    title: str
    key: str
    meter: str
    unit_note_length: str
    tempo: int | None
    tuning_name: str
    open_midi: list[float]
    sections: list[Section]
    phrases: list[Phrase]
    events: list[ScoreEvent]
    source_abc: str
    diagnostics: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PositionCandidate:
    candidate_id: str
    event_id: str
    target_midi: float
    mode: str
    string: int
    hui: float | None
    sounding_midi: float
    cents_error: float
    tone_region_cost: float
    confidence: str
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Route:
    phrase_id: str
    rank: int
    candidate_ids: list[str]
    event_ids: list[str]
    total_cost: float
    node_cost: float
    transition_cost: float
    entry_candidate_id: str | None
    exit_candidate_id: str | None
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformanceAction:
    action_id: str
    kind: str
    source_event_ids: list[str]
    mode: str | None = None
    string: int | None = None
    hui: float | None = None
    left_finger: str | None = None
    right_finger: str | None = None
    attack: bool = True
    attack_source_id: str | None = None
    # 双弦技法（撮／泼／剌）的第二根弦；mode2/hui2 缺省继承主位。
    string2: int | None = None
    mode2: str | None = None
    hui2: float | None = None
    compound_gesture: str | None = None
    techniques: list[str] = field(default_factory=list)
    pre_attack_techniques: list[str] = field(default_factory=list)
    confidence: str = "inferred"
    notation_omitted: bool = False
    jianzi_text: str | None = None
    rule_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformancePlan:
    actions: list[PerformanceAction]
    selected_routes: list[Route]
    total_cost: float
    diagnostics: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskEnvelope:
    job_id: str
    task_id: str
    agent_role: str
    payload: dict[str, Any]
    expected_schema: str
    ruleset_version: str = "transplant-course-1.0"
    skill_versions: dict[str, str] = field(default_factory=dict)
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskResult:
    task_id: str
    agent_role: str
    status: str
    output: dict[str, Any]
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
