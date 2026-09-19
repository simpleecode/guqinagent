#!/usr/bin/env python3
"""Audit stopped/open and double-stopped ``撮`` spellings in teacher plans.

This is deliberately a conservative text audit.  It reports combinations that
need a player's review rather than declaring every double stop unplayable.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = ROOT / (
    "ABC_J/agent_training/"
    "messages_glm_full_two_stage_harmonicstatefix_lipitchfix_"
    "p0063harmonicwarning_statewarning38_harmonicparsefix_walkpitchfix_"
    "20260918/teacher_trajectory_audit.jsonl"
)

_LEFT_FINGER = re.compile(r"(大指|食指|中指|名指)")
_STRING = re.compile(r"([一二三四五六七1-7])弦")
_HUI = re.compile(r"(徽外半|徽外|[一二三四五六七八九十]+徽(?:[一二三四五六七八九十]+分)?)")


def component_info(text: str) -> dict[str, str | None]:
    """Extract only explicit information from one parenthesised 撮 component."""
    finger = _LEFT_FINGER.search(text)
    string = _STRING.search(text)
    hui = _HUI.search(text)
    return {
        "text": text.strip(),
        "mode": "stopped" if "按音" in text else "open" if "散" in text else None,
        "left_finger": finger.group(1) if finger else None,
        "string": string.group(1) if string else None,
        "hui": hui.group(1) if hui else None,
    }


def cuo_components(text: str) -> list[dict[str, str | None]]:
    match = re.search(r"(?:小|大|反)?撮（([^）]+)）", text)
    if not match:
        return []
    # Current corpus uses ＋; accept common punctuation so the report remains
    # useful for newly generated examples too.
    return [component_info(part) for part in re.split(r"[＋+]", match.group(1))]


def inspect_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    private = record.get("teacher_private") or {}
    actions = (private.get("accepted_plan") or {}).get("actions") or []
    rows = []
    for action in actions:
        text = str(action.get("jianzi_text") or "")
        if "撮（" not in text:
            continue
        components = cuo_components(text)
        modes = [part["mode"] for part in components]
        if len(components) == 2 and modes.count("stopped") == 1 and modes.count("open") == 1:
            kind = "stopped_plus_open"
        elif len(components) == 2 and modes.count("stopped") == 2:
            kind = "double_stopped"
        else:
            kind = "other_or_unparsed"
        flags: list[str] = []
        if kind == "double_stopped":
            stopped = [part for part in components if part["mode"] == "stopped"]
            explicit = [part["left_finger"] for part in stopped if part["left_finger"]]
            if len(explicit) < 2:
                flags.append("double_stop_missing_explicit_second_left_finger")
            elif len(set(explicit)) < 2:
                flags.append("double_stop_reuses_one_explicit_left_finger")
            # A player-supplied, narrowly scoped known-infeasible pattern.
            strings = {part["string"]: part for part in stopped}
            if (strings.get("一") or strings.get("1")) and (strings.get("五") or strings.get("5")):
                first = strings.get("一") or strings.get("1")
                fifth = strings.get("五") or strings.get("5")
                if (first["left_finger"] == "大指" and first["hui"] == "九徽"
                        and fifth["hui"] == "七徽"):
                    flags.append("known_unreachable_thumb_1st_9hui__5th_7hui")
        rows.append({
            "sample_id": record.get("sample_id"),
            "stage": record.get("agent_stage") or str(record.get("sample_id", "")).split("-")[-3],
            "score_key": (record.get("provenance") or {}).get("score_key"),
            "source_index": action.get("source_index"),
            "jianzi_text": text,
            "kind": kind,
            "components": components,
            "review_flags": flags,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_samples = 0
    with args.audit.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                total_samples += 1
                rows.extend(inspect_record(json.loads(line)))

    by_kind = Counter(row["kind"] for row in rows)
    by_stage_kind = Counter((row["stage"], row["kind"]) for row in rows)
    stopped_open_fingers = Counter(
        row["components"][0]["left_finger"]
        for row in rows
        if row["kind"] == "stopped_plus_open"
        and row["components"][0]["mode"] == "stopped"
        and row["components"][0]["left_finger"]
    )
    # Components can appear in either order; count the stopped component.
    stopped_open_fingers = Counter(
        next(part["left_finger"] for part in row["components"] if part["mode"] == "stopped")
        for row in rows
        if row["kind"] == "stopped_plus_open"
        and any(part["left_finger"] for part in row["components"] if part["mode"] == "stopped")
    )
    flagged = [row for row in rows if row["review_flags"]]
    summary = {
        "audit": str(args.audit),
        "teacher_trajectory_samples": total_samples,
        "cuo_rows": len(rows),
        "by_kind": dict(sorted(by_kind.items())),
        "by_stage_and_kind": {
            f"{stage}:{kind}": count
            for (stage, kind), count in sorted(by_stage_kind.items())
        },
        "stopped_plus_open_left_finger": dict(stopped_open_fingers.most_common()),
        "double_stopped_review_candidates": len(flagged),
        "flag_counts": dict(Counter(flag for row in flagged for flag in row["review_flags"])),
        "notes": [
            "Double-stopped 撮 is not automatically classified as invalid.",
            "Flags only identify missing/duplicated explicit left fingers and the player-specified 1st-string ninth-hui thumb plus 5th-string seventh-hui pattern.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, values in (("all_cuo_rows.jsonl", rows), ("double_stopped_review_candidates.jsonl", flagged)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
