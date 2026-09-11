#!/usr/bin/env python3
"""Audit no-op Guqinizer phrases against the canonical GQS and base plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text(action: dict | None) -> str:
    if not action:
        return ""
    value = action.get("jianzi_text")
    if value is None:
        value = action.get("jianzi")
    return str(value or "").strip()


def canon(value: str) -> str:
    return "".join(str(value).split()).translate(str.maketrans("一二三四五六七", "1234567"))


def gqs_notes(path: Path) -> list[dict]:
    notes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("音｜"):
            cells = json.loads(line.split("｜", 1)[1])
            notes.append({"index": cells[0], "jianzi": cells[6], "jianzi_text": cells[7]})
    return notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noop", type=Path, required=True)
    parser.add_argument("--intermediates", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--gqs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    noop = rows(args.noop)
    ids = [item["provenance"]["source_trajectory_id"] for item in noop]
    intermediates = {item["trajectory_id"]: item for item in rows(args.intermediates)}
    references = {item["trajectory_id"]: item for item in rows(args.references)}
    result = []
    for trajectory_id in ids:
        item = references[trajectory_id]
        score_key = item["score_key"]
        notes = gqs_notes(args.gqs_dir / score_key / "teacher.gqs")
        by_index = {int(note["index"]): note for note in notes}
        event = item["input"]["event_range"]
        start = int(event.get("start_index", event.get("start")))
        end = int(event.get("end_index_exclusive", event.get("end_exclusive")))
        indices = list(range(start, end))
        target = {index: text(by_index.get(index)) for index in indices}
        base_actions = {
            int(action["source_index"]): text(action)
            for action in intermediates[trajectory_id].get("plan", {}).get("actions", [])
        }
        nonempty = [index for index, value in target.items() if value]
        comparable = [index for index in indices if index in base_actions and (target[index] or base_actions[index])]
        mismatches = [index for index in comparable if canon(target[index]) != canon(base_actions[index])]
        prior = [text(by_index.get(index)) for index in range(0, start)]
        repeat_markers = [value for value in prior if "再作" in value]
        result.append({
            "trajectory_id": trajectory_id,
            "score_key": score_key,
            "phrase_id": item["phrase_id"],
            "start_index": start,
            "end_index_exclusive": end,
            "rows": len(indices),
            "actual_nonempty": len(nonempty),
            "actual_nonempty_indices": nonempty,
            "base_nonempty": sum(bool(value) for value in base_actions.values()),
            "base_target_mismatches": len(mismatches),
            "mismatch_indices": mismatches,
            "prior_repeat_markers": repeat_markers[-5:],
            "all_actual_empty": not nonempty,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    print(json.dumps({
        "total": len(result),
        "actual_all_empty": sum(item["all_actual_empty"] for item in result),
        "actual_nonempty": sum(not item["all_actual_empty"] for item in result),
        "base_target_exact": sum(item["base_target_mismatches"] == 0 for item in result),
        "base_target_mismatch": sum(item["base_target_mismatches"] > 0 for item in result),
        "empty_with_repeat_marker": sum(bool(item["all_actual_empty"] and item["prior_repeat_markers"]) for item in result),
        "mismatch_histogram": Counter(item["base_target_mismatches"] for item in result),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
