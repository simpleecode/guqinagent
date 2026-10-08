"""Fixed Complex Fingering Handbook: appended to every system prompt.

The handbook is generated once by ``build_handbook.py`` (filtered to the
techniques present in the training-half reference annotations) and is then a
read-only artifact.  GEPA evolves only the strategy prompt;
``compose_system`` stitches candidate + handbook so that every consumer —
GEPA rollouts, the plain seed-prompt baseline ("Strong ReAct + Handbook")
and the final sealed-test run — sees exactly the same handbook.
"""
from __future__ import annotations

from pathlib import Path

HANDBOOK_PATH = Path(__file__).resolve().parent / "knowledge" / "complex_fingering_handbook.md"

FIXED_COMPLEX_FINGERING_HANDBOOK = HANDBOOK_PATH.read_text(encoding="utf-8").strip()


def compose_system(candidate_text: str) -> str:
    """GEPA candidate + fixed handbook; the handbook never enters mutation."""
    return candidate_text.rstrip() + "\n\n" + FIXED_COMPLEX_FINGERING_HANDBOOK
