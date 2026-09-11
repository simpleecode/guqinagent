#!/usr/bin/env python3
"""Clean failed unknown-tool calls from a public teacher-message copy.

The generator records a malformed ``null`` call followed by its failed tool
response before the successful retry.  Those failed calls are useful in the
private audit, but are not valid supervised tool targets for LLaMA-Factory.
This script copies the public JSONL rows and removes only such paired calls;
the source directory and private audit remain untouched.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    for source in args.input_dir.iterdir():
        if source.is_file() and source.name != "messages_train.jsonl":
            shutil.copy2(source, args.output_dir / source.name)

    rows = changed_rows = removed_calls = removed_results = 0
    source_path = args.input_dir / "messages_train.jsonl"
    output_path = args.output_dir / "messages_train.jsonl"
    with source_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            messages = row.get("messages") or []
            cleaned = []
            index = 0
            row_changed = False
            while index < len(messages):
                message = messages[index]
                calls = message.get("tool_calls") or []
                null_calls = [
                    call for call in calls
                    if str((call.get("function") or {}).get("name") or "") == "null"
                ]
                if message.get("role") == "assistant" and null_calls:
                    failed_result = (
                        messages[index + 1]
                        if index + 1 < len(messages)
                        else None
                    )
                    failed_ids = {str(call.get("id")) for call in null_calls}
                    failed_text = str((failed_result or {}).get("content") or "")
                    is_failed_pair = (
                        failed_result is not None
                        and failed_result.get("role") == "tool"
                        and str(failed_result.get("tool_call_id")) in failed_ids
                        and "unknown tool: null" in failed_text
                    )
                    if is_failed_pair:
                        kept_calls = [
                            call for call in calls
                            if str((call.get("function") or {}).get("name") or "") != "null"
                        ]
                        if kept_calls:
                            updated = dict(message)
                            updated["tool_calls"] = kept_calls
                            cleaned.append(updated)
                        # If no valid call remains, retain the reasoning-only
                        # assistant; the exporter will merge it with the next
                        # assistant tool-call turn.
                        else:
                            updated = dict(message)
                            updated.pop("tool_calls", None)
                            cleaned.append(updated)
                        removed_calls += len(null_calls)
                        removed_results += 1
                        row_changed = True
                        index += 2
                        continue
                cleaned.append(message)
                index += 1
            if row_changed:
                row["messages"] = cleaned
                changed_rows += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "teacher-public-invalid-tool-cleanup-1.0",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "rows": rows,
        "changed_rows": changed_rows,
        "removed_null_tool_calls": removed_calls,
        "removed_failed_tool_results": removed_results,
    }
    (args.output_dir / "invalid_tool_cleanup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
