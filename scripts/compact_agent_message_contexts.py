#!/usr/bin/env python3
"""Rewrite existing message artifacts to the compact on-demand history format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact_input(value: dict) -> dict:
    handoff = value.pop("phrase_handoff", None)
    if not isinstance(handoff, dict):
        return value
    older = handoff.get("older_context_refs") or []
    value["phrase_context"] = {
        "current_phrase_id": handoff.get("phrase_id"),
        "current_phrase": handoff.get("current_phrase") or value.get("notes") or [],
        "previous_phrase": handoff.get("previous_phrase"),
        "older_history": {
            "available": bool(older),
            "phrase_count": len(older),
            "instruction": (
                "默认不加载；需要时先调用 list_context，再用 expand_context 展开一个 phrase。"
                if older else "无更早 phrase"
            ),
        },
    }
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_rows = []
    for line in (args.input_dir / "messages_train.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        for message in row.get("messages", []):
            if message.get("role") != "user" or not isinstance(message.get("content"), str):
                continue
            try:
                payload = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(payload.get("input"), dict):
                payload["input"] = compact_input(payload["input"])
                message["content"] = json.dumps(payload, ensure_ascii=False)
        public_rows.append(row)
    with (args.output_dir / "messages_train.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for row in public_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name in ("teacher_trajectory_audit.jsonl", "fingering_intermediates.jsonl",
                 "generation_report.json"):
        source = args.input_dir / name
        if source.exists():
            (args.output_dir / name).write_bytes(source.read_bytes())
    print(json.dumps({"rows": len(public_rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
