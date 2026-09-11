#!/usr/bin/env python3
"""Remove two redundant legacy headings from accepted public user prompts.

Only exact boilerplate in ``role=user`` messages is changed.  Assistant
reasoning, tool calls/results, private audits, and score tables are untouched.
Writes atomically and leaves a compact audit report beside the JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TASK = re.compile(r"^任务｜(?:fingering_agent|guqinization)\r?\n")
REQUIREMENT = "要求｜通过 edit_plan.jianzi_rows 提交减字文字。"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def clean(content: str) -> tuple[str, bool, bool]:
    result, removed_task = TASK.subn("", content, count=1)
    suffix = "\n\n" + REQUIREMENT
    removed_requirement = result.endswith(suffix)
    if removed_requirement:
        result = result[:-len(suffix)]
    return result, bool(removed_task), removed_requirement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.input_dir / "messages_train.jsonl"
    before = digest(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows = users = task_count = requirement_count = changed = 0
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            for message in row.get("messages") or []:
                if message.get("role") != "user":
                    continue
                users += 1
                value, removed_task, removed_requirement = clean(str(message.get("content") or ""))
                if removed_task or removed_requirement:
                    message["content"] = value
                    changed += 1
                    task_count += int(removed_task)
                    requirement_count += int(removed_requirement)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    report = {
        "schema_version": "legacy-user-prompt-cleanup-1.0",
        "messages_path": str(path), "rows": rows, "user_messages": users,
        "changed_user_messages": changed, "removed_task_headers": task_count,
        "removed_requirement_lines": requirement_count,
        "before_sha256": before, "after_sha256": digest(path),
    }
    (args.input_dir / "legacy_user_prompt_cleanup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
