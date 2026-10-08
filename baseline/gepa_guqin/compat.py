"""Field compatibility for the renderfix action schema.

The 2026-10-07 re-split stores the rendered jianzi surface in ``text`` while
``jianzi_text`` may be ``None``; older paths (prompt renderer, handoff,
prediction records) still read ``jianzi_text``.  These helpers normalise
actions at the boundaries so every downstream consumer sees the legacy field.
"""
from __future__ import annotations

from typing import Any


def action_text(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    return str(action.get("jianzi_text") or action.get("text") or "")


def materialize_action(action: dict[str, Any]) -> dict[str, Any]:
    copied = dict(action)
    if not str(copied.get("jianzi_text") or "").strip() and copied.get("text") is not None:
        copied["jianzi_text"] = copied["text"]
    return copied


def materialize_actions(actions: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [materialize_action(action) for action in actions or []]
