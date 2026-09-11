#!/usr/bin/env python3
"""Validate the schema-stable full-trajectory SFT export before server upload."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


BANNED = ("GQS", "教师提示", "私有参考", "参考谱面", "参考谱", "系统提示", "teacher-only", "private answer")
TOOL_CALL = re.compile(r"<tool_call>\s*<function=[^\s<>]+.*?</function>\s*</tool_call>", re.S)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    data_path = args.input_dir / "guqin_agent_train.jsonl"
    meta_path = args.input_dir / "guqin_agent_train_metadata.jsonl"
    errors: list[str] = []
    by_stage = Counter()
    assistant_turns = 0
    tool_call_turns = 0
    data_rows = 0
    with data_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            data_rows += 1
            if set(row) != {"messages", "tools"} or not isinstance(row["tools"], str):
                errors.append(f"{line_number}: unstable top-level schema")
                continue
            try:
                json.loads(row["tools"])
            except json.JSONDecodeError:
                errors.append(f"{line_number}: tools is not JSON")
            messages = row["messages"]
            roles = [message.get("role") for message in messages]
            if len(messages) < 3 or roles[:2] != ["system", "user"] or roles[-1] != "assistant":
                errors.append(f"{line_number}: invalid SFT conversation boundary")
                continue
            aligned_roles = []
            pending_tools = False
            for role in roles[1:]:
                if role == "tool":
                    pending_tools = True
                    continue
                if pending_tools:
                    aligned_roles.append("tool")
                    pending_tools = False
                aligned_roles.append(role)
            if pending_tools:
                aligned_roles.append("tool")
            if any(
                role not in ({"user", "tool"} if index % 2 == 0 else {"assistant"})
                for index, role in enumerate(aligned_roles)
            ):
                errors.append(f"{line_number}: roles violate OpenAI converter alternation")
                continue
            row_tool_calls = 0
            for message in messages:
                if message.get("role") == "assistant":
                    assistant_turns += 1
                    content = str(message.get("content") or "").lower()
                    matches = TOOL_CALL.findall(content)
                    row_tool_calls += len(matches)
                    tool_call_turns += len(matches)
                    if any(term.lower() in content for term in BANNED):
                        errors.append(f"{line_number}: assistant private-source term")
                        break
                if set(message) != {"role", "content"}:
                    errors.append(f"{line_number}: messages must contain only role/content after preprocessing")
                    break
            # A direct assistant answer is a valid supervised no-op trajectory:
            # it teaches the model to stop without manufacturing an edit.
    meta_rows = 0
    with meta_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            meta_rows += 1
            by_stage[str(row.get("agent_stage"))] += 1
            if not row.get("training_trajectory_id") or not row.get("source_sample_id"):
                errors.append(f"metadata:{line_number}: missing stable IDs")
    if data_rows != meta_rows:
        errors.append(f"row count mismatch: data={data_rows}, metadata={meta_rows}")
    report = {
        "schema_version": "guqin-agent-sft-export-validation-2.0",
        "valid": not errors,
        "rows": data_rows,
        "by_stage": dict(by_stage),
        "assistant_turns": assistant_turns,
        "tool_call_blocks": tool_call_turns,
        "errors": errors[:100],
    }
    (args.input_dir / "sft_export_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
