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


def clean_messages(messages, declared_tools):
    """Remove only undeclared calls with an explicit matching unknown-tool error.

    Batched tool calls may have several consecutive result messages; the
    rejected call need not be the first result. Successful calls are untouched.
    """
    remove_ids = set()
    for index, message in enumerate(messages):
        if message.get('role') != 'assistant':
            continue
        results = {}
        for result in messages[index + 1:]:
            if result.get('role') != 'tool':
                break
            key = str(result.get('tool_call_id'))
            if key in results:
                raise ValueError(f'Duplicate tool receipt: {key}')
            results[key] = result
        for call in message.get('tool_calls') or []:
            name = str((call.get('function') or {}).get('name') or '')
            key = str(call.get('id'))
            if name in declared_tools or key not in results:
                continue
            try:
                result = json.loads(str(results[key].get('content') or ''))
            except json.JSONDecodeError:
                continue
            if result.get('ok') is False and f'unknown tool: {name}' in str(result.get('error', '')):
                remove_ids.add(key)
    cleaned = []
    for message in messages:
        if message.get('role') == 'tool' and str(message.get('tool_call_id')) in remove_ids:
            continue
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            kept = [c for c in message['tool_calls'] if str(c.get('id')) not in remove_ids]
            if len(kept) != len(message['tool_calls']):
                message = dict(message)
                if kept:
                    message['tool_calls'] = kept
                else:
                    message.pop('tool_calls')
        cleaned.append(message)
    return cleaned, sorted(remove_ids)


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
            declared_tools = {
                str(tool.get("name") or "") for tool in (row.get("tools") or [])
            }
            cleaned, removed = clean_messages(messages, declared_tools)
            if removed:
                row["messages"] = cleaned
                changed_rows += 1
                removed_calls += len(removed)
                removed_results += len(removed)
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                target.write(line)

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
