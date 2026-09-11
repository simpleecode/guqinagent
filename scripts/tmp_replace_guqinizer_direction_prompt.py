#!/usr/bin/env python3
"""Replace the historical Guqinizer slide-direction wording in JSONL traces.

This is an intentionally small, one-off migration: it rewrites only the two
known legacy sentence bodies and leaves all other JSON content untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

OLD_BODIES = (
    "绰上表示左手从较低徽位向较高徽位滑行，注下表示从较高徽位向较低徽位滑行。",
    "绰上是左手由较低徽位滑向较高徽位，注下是由较高徽位滑向较低徽位。",
    # A few archived teacher responses quoted the same sentence without its
    # final punctuation; migrate those quotations as well.
    "绰上表示左手从较低徽位向较高徽位滑行，注下表示从较高徽位向较低徽位",
    "绰上是左手由较低徽位滑向较高徽位，注下是由较高徽位滑向较低徽位",
    "绰上表示左手从较低徽位向较高徽位滑行",
    "注下表示从较高徽位向较低徽位滑行",
    "绰上是左手由较低徽位滑向较高徽位",
    "注下是由较高徽位滑向较低徽位",
)
NEW_BODY = (
    "绰表示左手由低音位置滑向本位，注表示左手由高音位置滑向本位"
    "（如绰上七徽：由八徽方向滑至七徽；注下七徽：由六徽方向滑至七徽）。"
)


def rewrite_jsonl(path: Path) -> int:
    temporary = path.with_suffix(path.suffix + ".direction.tmp")
    replacements = 0
    with path.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if line.strip():
                # JSON round-trip keeps the JSONL valid and avoids accidental
                # replacement in binary/formatting artifacts.
                value = json.loads(line)
                text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                before = text
                for old in OLD_BODIES:
                    text = text.replace(old, NEW_BODY)
                replacements += sum(before.count(old) for old in OLD_BODIES)
                target.write(text + "\n")
            else:
                target.write(line)
    os.replace(temporary, path)
    return replacements


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    total = 0
    reports = []
    for path in args.paths:
        count = rewrite_jsonl(path)
        total += count
        reports.append({"path": str(path), "replacements": count})
    print(json.dumps({"total_replacements": total, "files": reports}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
