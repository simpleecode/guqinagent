from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"


@lru_cache(maxsize=2)
def load_qwen35_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    local_snapshot = snapshot_download(model_id, local_files_only=True)
    return AutoTokenizer.from_pretrained(local_snapshot, local_files_only=True)


def qwen_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style declarations to Qwen's function-tool shape."""
    return [{
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": deepcopy(tool.get("input_schema") or {"type": "object"}),
        },
    } for tool in tools]


def qwen_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize stored tool-call arguments for the official Qwen template."""
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                function["arguments"] = json.loads(arguments)
    return normalized


def serialize_qwen35(sample: dict[str, Any], *, tokenizer=None,
                     model_id: str = DEFAULT_MODEL_ID) -> str:
    """Apply the selected model's official chat template without tokenizing."""
    if tokenizer is None:
        tokenizer = load_qwen35_tokenizer(model_id)
    return tokenizer.apply_chat_template(
        qwen_messages(sample.get("messages") or []),
        tools=qwen_tools(sample.get("tools") or []),
        tokenize=False,
        add_generation_prompt=False,
    )
