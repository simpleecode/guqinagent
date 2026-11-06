"""In-process form of the repository's sealed exact-cell scorer.

The definitions deliberately match
``guqinagent/train/scripts/score_agent_predictions.py``.  It is used by the
prompt experiment runner so candidate selection cannot drift back to the old
weighted GEPA proxy score.
"""
from __future__ import annotations

from typing import Any
from train.scripts.score_agent_predictions import canonical


def action_map(actions: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(action["source_index"]): str(action.get("text") or action.get("jianzi_text") or "")
        for action in actions if action.get("source_index") is not None
    }


def score_prediction(reference_actions: list[dict[str, Any]], prediction: dict[str, Any]) -> dict[str, Any]:
    """Return the three official scorer quantities for one phrase prediction."""
    reference = action_map(reference_actions)
    generated = {
        int(cell[0]): str(cell[1])
        for cell in prediction.get("jianzi_rows") or []
        if isinstance(cell, (list, tuple)) and len(cell) >= 2
    }
    indices = set(reference) | set(generated)
    exact = sum(canonical(reference.get(index)) == canonical(generated.get(index)) for index in indices)
    nonempty_reference = {index for index, text in reference.items() if canonical(text)}
    nonempty_generated = {index for index, text in generated.items() if canonical(text)}
    return {
        "rows_compared": len(indices),
        "exact_cells": exact,
        "reference_nonempty": len(nonempty_reference),
        "predicted_nonempty": len(nonempty_generated),
        "nonempty_hits": len(nonempty_reference & nonempty_generated),
        "exact_cell_rate": exact / len(indices) if indices else None,
        "nonempty_recall": (len(nonempty_reference & nonempty_generated) / len(nonempty_reference)
                            if nonempty_reference else None),
    }
