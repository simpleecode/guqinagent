#!/usr/bin/env python3
"""Extract phrase IDs whose public prompt still uses the legacy harmonic wording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LEGACY_TEXT = "泛音区间｜当前段开始时是"


def has_legacy_prompt(row: dict) -> bool:
    return any(
        message.get("role") == "user" and LEGACY_TEXT in str(message.get("content") or "")
        for message in row.get("messages") or []
    )


def phrase_id(row: dict) -> str:
    stage = str(row["agent_stage"])
    return str(row["sample_id"]).removesuffix(f"-{stage}-teacher-tools")


def write_ids(path: Path, ids: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in sorted(ids)), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fingering-output", type=Path, required=True)
    parser.add_argument("--guqinizer-output", type=Path, required=True)
    args = parser.parse_args()

    fingering_ids: set[str] = set()
    guqinizer_ids: set[str] = set()
    for line in args.input.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if not has_legacy_prompt(row):
            continue
        if row.get("agent_stage") == "fingering_agent":
            fingering_ids.add(phrase_id(row))
        elif row.get("agent_stage") == "guqinization":
            guqinizer_ids.add(phrase_id(row))

    write_ids(args.fingering_output, fingering_ids)
    write_ids(args.guqinizer_output, guqinizer_ids)
    print(json.dumps({
        "legacy_text": LEGACY_TEXT,
        "fingering_phrase_count": len(fingering_ids),
        "guqinizer_phrase_count": len(guqinizer_ids),
        "guqinizer_subset_of_fingering": guqinizer_ids <= fingering_ids,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
