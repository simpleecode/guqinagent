"""Thin adapters over the repository's authoritative pitch audit."""
from __future__ import annotations

from typing import Any

from agents.abc_to_jianzipu.reference_parser import AUDIT


def target_midis(note: dict[str, Any], metadata: dict[str, Any]) -> list[float]:
    """Return all notated target tones (a chord can have two) in MIDI."""
    tonic = AUDIT.parse_tonic_midi(metadata)
    return [value for value in (
        AUDIT.parse_jianpu(note.get("jianpu"), tonic),
        AUDIT.parse_jianpu(note.get("jianpu_alt"), tonic),
    ) if value is not None]


def audit_plan(data: dict[str, Any], tolerance_cents: float) -> dict[str, Any]:
    """Delegate pitch/state parsing to ``audit_jianpu_jianzi_pitch``."""
    return AUDIT.audit(data, tolerance_cents)


def paired_cents(detail: dict[str, Any]) -> list[float]:
    return [float(pair["absolute_cents"]) for pair in detail.get("pairs") or []]

