from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ConfidenceClass = Literal["verified", "weak", "conflicting", "unusable"]
PatchType = Literal[
    "REPOSITION", "CHANGE_FINGER", "CHANGE_MODE", "ADD_TECHNIQUE",
    "REMOVE_TECHNIQUE", "CHANGE_ATTACK", "SET_COMPOUND_GESTURE",
    "SET_JIANZI_TEXT", "NO_OP",
]


@dataclass(slots=True)
class ReferenceAction:
    source_index: int
    text: str
    attack: bool
    mode: str | None
    string: int | None
    hui: float | None
    left_finger: str | None
    right_finger: str | None
    # 双弦技法（撮／泼／剌）的第二根弦：mode2/hui2 缺省时继承主位。
    string2: int | None = None
    mode2: str | None = None
    hui2: float | None = None
    # 多声／跨弦复合指法保留为原子语义，不拆成普通撮、剌等子串。
    compound_gesture: str | None = None
    jianzi_text: str | None = None
    techniques: list[str] = field(default_factory=list)
    # attack=true 时，位于取音主体之前的技法，如“绰大指五徽六分挑6弦”。
    pre_attack_techniques: list[str] = field(default_factory=list)
    explicit_fields: list[str] = field(default_factory=list)
    inherited_fields: list[str] = field(default_factory=list)
    confidence_class: ConfidenceClass = "weak"
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PlanPatch:
    patch_id: str
    patch_type: PatchType
    source_index: int
    before: dict[str, Any]
    after: dict[str, Any]
    confidence_class: ConfidenceClass
    confidence: float
    evidence: list[str] = field(default_factory=list)
    required_tools: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
