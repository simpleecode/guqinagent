#!/usr/bin/env python3
"""Classify no-op phrases using corrected repeat-semantics references."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def text(action: dict | None) -> str:
    if not action:
        return ""
    return str(action.get("jianzi_text", action.get("jianzi", "")) or "").strip()


def canon(value: str) -> str:
    return "".join(value.split()).translate(str.maketrans("一二三四五六七", "1234567"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noop", type=Path, required=True)
    parser.add_argument("--intermediates", type=Path, required=True)
    parser.add_argument("--references-v9", type=Path, required=True)
    parser.add_argument("--references-fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = [row["provenance"]["source_trajectory_id"] for row in read(args.noop)]
    refs = {row["trajectory_id"]: row for row in read(args.references_v9)}
    fallback = {row["trajectory_id"]: row for row in read(args.references_fallback)}
    intermediates = {row["trajectory_id"]: row for row in read(args.intermediates)}
    out = []
    for tid in ids:
        item = refs.get(tid) or fallback.get(tid)
        if item is None:
            out.append({"trajectory_id": tid, "missing_reference": True, "all_actual_empty": True, "prior_repeat_markers": []})
            continue
        actions = item.get("reference_plan", {}).get("actions", [])
        target = {int(a["source_index"]): text(a) for a in actions}
        base = {int(a["source_index"]): text(a) for a in intermediates[tid].get("plan", {}).get("actions", [])}
        # Compare the complete phrase index domain.  Previously only the
        # intersection was compared, so a reference with omitted rows could
        # falsely look like a no-op even when Base contained whole runs of text.
        note_indices = {
            int(note["index"])
            for note in (item.get("input", {}).get("notes_without_jianzi") or [])
            if note.get("index") is not None
        }
        all_indices = note_indices | set(target) | set(base)
        mismatch = [
            index for index in sorted(all_indices)
            if canon(target.get(index, "")) != canon(base.get(index, ""))
        ]
        markers = item.get("provenance", {}).get("repeat_materialization", {}).get("markers", {}) or {}
        out.append({
            "trajectory_id": tid, "score_key": item.get("score_key"), "phrase_id": item.get("phrase_id"),
            "rows": len(all_indices), "actual_nonempty": sum(bool(value) for value in target.values()),
            "base_nonempty": sum(bool(value) for value in base.values()), "base_target_mismatches": len(mismatch),
            "mismatch_indices": mismatch, "prior_repeat_markers": sorted(markers),
            "all_actual_empty": not any(target.values()), "reference_source": "v9" if tid in refs else "fallback",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "total": len(out), "all_empty": sum(x.get("all_actual_empty", False) for x in out),
        "all_empty_with_repeat": sum(bool(x.get("all_actual_empty") and x.get("prior_repeat_markers")) for x in out),
        "all_empty_without_repeat": sum(bool(x.get("all_actual_empty") and not x.get("prior_repeat_markers")) for x in out),
        "nonempty": sum(not x.get("all_actual_empty", False) for x in out),
        "missing_reference": sum(x.get("missing_reference", False) for x in out),
        "mismatch_histogram": Counter(x.get("base_target_mismatches") for x in out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
