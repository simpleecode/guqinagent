"""Qwen3.5 generation boundaries shared by local evaluation entry points."""
from __future__ import annotations

from typing import Any


TURN_END_MARKERS = ("<|im_end|>", "<|endoftext|>")


def qwen35_eos_token_ids(tokenizer: Any) -> list[int]:
    """Return all single-token assistant-turn terminators known by the tokenizer."""
    ids: list[int] = []
    configured = tokenizer.eos_token_id
    if isinstance(configured, int):
        ids.append(configured)
    elif configured:
        ids.extend(int(value) for value in configured)
    for marker in TURN_END_MARKERS:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise RuntimeError("Qwen3.5 tokenizer exposes no usable EOS token id")
    return ids


def trim_qwen35_assistant_turn(text: str) -> str:
    """Discard any hallucinated role continuation after the first assistant turn."""
    positions = [position for marker in TURN_END_MARKERS if (position := text.find(marker)) >= 0]
    if not positions:
        return text.strip()
    return text[: min(positions)].strip()
