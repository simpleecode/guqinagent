#!/usr/bin/env python3
"""Validate public Agent messages and separation from teacher-private answers."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_PUBLIC_KEYS = ("teacher_private", "reference_plan", "reference_actions", "target_patches")
BANNED_ASSISTANT_PHRASES = ("reference action", "reference plan", "teacher-only",
                            "private answer", "参考答案", "标注答案", "私有答案")
BANNED_PUBLIC_REFERENCE_FEEDBACK = (
    "reference_jianzi_text_mismatch",
    "nonempty_reference_jianzi_dropped",
)
BANNED_PUBLIC_REFERENCE_TERMS = ("GQS", "教师私有", "最终标注")
BANNED_SURFACE_PATTERNS = (
    (re.compile(r"\[无[1-7]弦"), "open-string surface contains literal 无"),
    (re.compile(r"\[(?:大指|名指|食指|中指|跪指)[1-7]弦[^\]]*徽"),
     "non-open surface puts string before hui"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ABC_J/agent_training/messages")
    args = parser.parse_args()
    public_path = args.input_dir / "messages_train.jsonl"
    private_path = args.input_dir / "teacher_trajectory_audit.jsonl"
    errors, ids, stages = [], set(), Counter()
    private_ids = set()
    with private_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            private_ids.add(item["sample_id"])
            if "teacher_private" not in item:
                errors.append(f"private:{line_number}: missing teacher_private")
    with public_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            item = json.loads(line)
            sample_id = item["sample_id"]
            if sample_id in ids:
                errors.append(f"public:{line_number}: duplicate sample_id")
            ids.add(sample_id)
            stages[item["agent_stage"]] += 1
            serialized = json.dumps(item, ensure_ascii=False)
            for banned in BANNED_PUBLIC_KEYS:
                if f'"{banned}"' in serialized:
                    errors.append(f"public:{line_number}: leaked {banned}")
            for code in BANNED_PUBLIC_REFERENCE_FEEDBACK:
                if code in serialized:
                    errors.append(
                        f"public:{line_number}: leaked teacher-reference feedback: {code}"
                    )
            for term in BANNED_PUBLIC_REFERENCE_TERMS:
                if term.lower() in serialized.lower():
                    errors.append(
                        f"public:{line_number}: mentions teacher-private source: {term}"
                    )
            for pattern, description in BANNED_SURFACE_PATTERNS:
                if pattern.search(serialized):
                    errors.append(f"public:{line_number}: invalid jianzi surface: {description}")
            messages = item.get("messages") or []
            declared_tools = {tool.get("name") for tool in item.get("tools") or []}
            if len(messages) < 3 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
                errors.append(f"public:{line_number}: invalid conversation prefix")
                continue
            terminal_role = messages[-1].get("role")
            termination_kind = (item.get("termination") or {}).get("kind")
            if terminal_role != "assistant":
                errors.append(f"public:{line_number}: invalid terminal message")
            pending = {}
            returned = set()
            for index, message in enumerate(messages):
                role = message.get("role")
                if role == "assistant":
                    assistant_text = str(message.get("content") or "").lower()
                    for phrase in BANNED_ASSISTANT_PHRASES:
                        if phrase.lower() in assistant_text:
                            errors.append(
                                f"public:{line_number}:{index}: semantic private leakage: {phrase}"
                            )
                for call in message.get("tool_calls") or []:
                    call_id = call.get("id")
                    if role != "assistant" or not call_id or call_id in pending:
                        errors.append(f"public:{line_number}:{index}: invalid tool call")
                    pending[call_id] = index
                    tool_name = (call.get("function") or {}).get("name")
                    if str(item.get("schema_version", "")).startswith("agent-messages-2.") and tool_name not in declared_tools:
                        errors.append(f"public:{line_number}:{index}: undeclared tool {tool_name}")
                    try:
                        json.loads(call["function"]["arguments"])
                    except Exception:
                        errors.append(f"public:{line_number}:{index}: invalid arguments JSON")
                if role == "tool":
                    call_id = message.get("tool_call_id")
                    if call_id not in pending or call_id in returned:
                        errors.append(f"public:{line_number}:{index}: orphan/duplicate tool result")
                    elif any(messages[position].get("role") != "tool"
                             for position in range(pending[call_id] + 1, index)):
                        errors.append(f"public:{line_number}:{index}: tool result left its call batch")
                    returned.add(call_id)
                    try:
                        json.loads(message["content"])
                    except Exception:
                        errors.append(f"public:{line_number}:{index}: invalid tool result JSON")
            if set(pending) != returned:
                errors.append(f"public:{line_number}: calls without results")
            if termination_kind == "accepted_edit_plan" and not pending:
                errors.append(f"public:{line_number}: accepted_edit_plan trajectory has no tool calls")
            verification = item.get("verification") or {}
            if not all(verification.values()):
                errors.append(f"public:{line_number}: failed verification flags")
    if ids != private_ids:
        errors.append(f"public/private id mismatch: public={len(ids)} private={len(private_ids)}")
    report = {"schema_version": "agent-messages-validation-1.0", "valid": not errors,
              "message_trajectories": len(ids), "by_stage": dict(stages),
              "public_private_ids_match": ids == private_ids, "errors": errors[:200]}
    (args.input_dir / "messages_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
