#!/usr/bin/env python3
"""Repair legacy public prompts that exposed current-section placeholders in read-only sections.

Only the exact ``[减字待填写]`` surface inside ``【只读...】`` blocks is changed
to ``[空]``.  Editable/current sections and all structured fields are untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def repair(text: str) -> tuple[str, int, int]:
    lines = text.splitlines(keepends=True)
    readonly = False
    changed = 0
    barlines = 0
    out: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("【只读"):
            readonly = True
        elif stripped.startswith("【当前段") or stripped.startswith("更早段"):
            readonly = False
        if readonly and "[减字待填写]" in line:
            line = line.replace("[减字待填写]", "[空]")
            changed += 1
        # A barline is a structural event, not an empty jianzi value.  Keep a
        # visible marker in both read-only and current phrase tables.
        if "｜|｜|｜小节线｜" in line and not line.rstrip().endswith("[小节线]"):
            line = line.rstrip("\r\n") + "[小节线]\n"
            barlines += 1
        out.append(line)
    return "".join(out), changed, barlines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    changes = 0
    barlines = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            for message in row.get("messages") or []:
                if message.get("role") == "user":
                    message["content"], count, bars = repair(str(message.get("content") or ""))
                    changes += count
                    barlines += bars
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    print(json.dumps({"rows": rows, "replaced_readonly_placeholders": changes,
                      "barline_markers_added": barlines}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
