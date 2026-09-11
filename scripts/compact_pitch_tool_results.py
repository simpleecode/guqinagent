#!/usr/bin/env python3
"""Compact already-recorded get_pitch_candidates tool-result tables.

Only the human-readable ``result.text`` is rewritten. Tool arguments,
candidate objects, audits, and all non-pitch messages remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OLD_HEADER = "序号｜方式｜弦徽｜实得MIDI｜误差｜可信度"
NEW_HEADER = "序号｜方式｜弦徽｜实得MIDI｜可信度"


def split_sources(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "（":
            depth += 1
        elif char == "）":
            depth = max(0, depth - 1)
        elif char == "、" and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def compact_source_header(line: str) -> str:
    match = re.match(r"^(目标音高｜MIDI\s+[^｜]+｜来源｜)(.+)$", line)
    if not match:
        return line
    indices: list[str] = []
    jianpu: list[str] = []
    for part in split_sources(match.group(2)):
        source = re.match(r"^(\d+)(?:（简谱(.+)）)?$", part)
        if not source:
            source = re.match(r"^(\d+)", part)
        if not source:
            continue
        indices.append(source.group(1))
        if source.lastindex and source.lastindex >= 2 and source.group(2):
            value = source.group(2).strip()
            if value not in jianpu:
                jianpu.append(value)
    if not indices:
        return line
    result = match.group(1) + "、".join(dict.fromkeys(indices))
    if jianpu:
        result += "｜简谱｜" + "、".join(jianpu)
    return result


def compact_candidate_row(line: str) -> str:
    fields = line.split("｜")
    # A compacted source header can also contain six full-width separators:
    # ``目标音高｜MIDI …｜来源｜…｜简谱｜…``.  Only numeric first-column
    # rows are candidate rows; otherwise the header itself would be mistaken
    # for a candidate and ``451｜简谱｜1（饰）`` would become ``451.0｜1（饰）``.
    if len(fields) != 6 or not fields[0].strip().isdigit():
        return line
    rank, mode, position, midi, _error, confidence = fields
    try:
        midi_text = f"{float(midi):.1f}"
    except ValueError:
        return line
    return "｜".join((rank, mode, position, midi_text, confidence))


def compact_text(text: str) -> tuple[str, bool]:
    changed = False
    output: list[str] = []
    for line in text.splitlines():
        updated = line
        if updated == OLD_HEADER:
            updated = NEW_HEADER
        elif updated.startswith("目标音高｜MIDI ") and "｜来源｜" in updated:
            updated = compact_source_header(updated)
        elif updated.startswith("目标｜") or updated.startswith("指定音高｜"):
            # Target-only queries have no source list, but their candidate rows
            # still use the same verbose table format.
            pass
        compacted_row = compact_candidate_row(updated)
        if compacted_row != updated:
            updated = compacted_row
        changed |= updated != line
        output.append(updated)
    result = "\n".join(output)
    if text.endswith("\n"):
        result += "\n"
    return result, changed


def rewrite_row(row: dict) -> bool:
    changed = False
    for message in row.get("messages") or []:
        if message.get("role") != "tool" or not isinstance(message.get("content"), str):
            continue
        try:
            payload = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        result = payload.get("result")
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            continue
        compacted, did_change = compact_text(result["text"])
        if did_change:
            result["text"] = compacted
            message["content"] = json.dumps(payload, ensure_ascii=False)
            changed = True
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    changed_rows = 0
    changed_messages = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row_changed_messages = 0
            for message in row.get("messages") or []:
                if message.get("role") != "tool" or not isinstance(message.get("content"), str):
                    continue
                try:
                    payload = json.loads(message["content"])
                except json.JSONDecodeError:
                    continue
                result = payload.get("result")
                if isinstance(result, dict) and isinstance(result.get("text"), str):
                    _, did_change = compact_text(result["text"])
                    row_changed_messages += int(did_change)
            if rewrite_row(row):
                changed_rows += 1
                changed_messages += row_changed_messages
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "input": str(args.input), "output": str(args.output),
        "changed_rows": changed_rows, "changed_tool_messages": changed_messages,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
