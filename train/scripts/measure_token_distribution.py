#!/usr/bin/env python3
"""Measure Qwen3.5 chat-template token lengths for exported trajectories."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoProcessor


def normalize_tools(tools: object) -> list[dict] | None:
    """Accept both raw agent-message tools and LLaMA-Factory OpenAI tools."""
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    if not isinstance(tools, list):
        return None
    normalized: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(tool)
            continue
        if not tool.get("name"):
            continue
        normalized.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
            },
        })
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    # Qwen3.5's multimodal processor owns the chat template.  Calling the
    # tokenizer's template directly can return a malformed/near-empty result
    # (and previously produced the bogus constant length 2).
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    tokenizer = processor.tokenizer
    metadata_path = args.metadata or args.input.with_name(args.input.stem + "_metadata.jsonl")
    metadata = []
    if metadata_path.exists():
        metadata = [json.loads(line) for line in metadata_path.open(encoding="utf-8") if line.strip()]
    lengths: list[int] = []
    by_stage: dict[str, list[int]] = defaultdict(list)
    maximum: tuple[int, int, dict] | None = None
    for line_number, line in enumerate(args.input.open(encoding="utf-8"), 1):
        row = json.loads(line)
        tools = normalize_tools(row.get("tools"))
        rendered = processor.apply_chat_template(
            row["messages"], tools=tools, tokenize=False, add_generation_prompt=False,
        )
        length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        lengths.append(length)
        stage = (metadata[line_number - 1].get("agent_stage")
                 if line_number <= len(metadata) else row.get("agent_stage", "unknown"))
        by_stage[str(stage)].append(length)
        if maximum is None or length > maximum[0]:
            maximum = (length, line_number, row.get("metadata") or {})
    lengths.sort()
    def quantile(q: float) -> float:
        position = (len(lengths) - 1) * q
        low, high = int(position), min(int(position) + 1, len(lengths) - 1)
        return lengths[low] + (lengths[high] - lengths[low]) * (position - low)
    def summary(values: list[int]) -> dict[str, float | int]:
        values = sorted(values)
        def q_local(q: float) -> float:
            position = (len(values) - 1) * q
            low, high = int(position), min(int(position) + 1, len(values) - 1)
            return values[low] + (values[high] - values[low]) * (position - low)
        bins = {
            "<=2048": sum(value <= 2048 for value in values),
            "2049-4096": sum(2048 < value <= 4096 for value in values),
            "4097-6144": sum(4096 < value <= 6144 for value in values),
            "6145-8192": sum(6144 < value <= 8192 for value in values),
            ">8192": sum(value > 8192 for value in values),
        }
        return {"rows": len(values), "p50": q_local(.5), "p95": q_local(.95),
                "p99": q_local(.99), "max": values[-1],
                "over_2048": sum(value > 2048 for value in values), "bins": bins}
    report = {
        "rows": len(lengths), "min": lengths[0],
        "p50": quantile(.50), "p90": quantile(.90),
        "p95": quantile(.95), "p99": quantile(.99), "max": maximum[0],
        "over_2048": sum(value > 2048 for value in lengths),
        "over_8192": sum(value > 8192 for value in lengths),
        "over_16384": sum(value > 16384 for value in lengths),
        "bins": {
            "<=2048": sum(value <= 2048 for value in lengths),
            "2049-4096": sum(2048 < value <= 4096 for value in lengths),
            "4097-6144": sum(4096 < value <= 6144 for value in lengths),
            "6145-8192": sum(6144 < value <= 8192 for value in lengths),
            ">8192": sum(value > 8192 for value in lengths),
        },
        "max_line": maximum[1], "max_metadata": maximum[2],
        "by_stage": {stage: summary(values) for stage, values in by_stage.items()},
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
