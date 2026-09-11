#!/usr/bin/env python3
"""Compare notation_omitted rows between two inferred trajectory trees."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_tree(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(root.glob("inferred_trajectories_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["trajectory_id"])] = row
    return rows


def omitted(row: dict) -> set[int]:
    return {
        int(note["index"])
        for note in (row.get("input", {}).get("notes_without_jianzi") or [])
        if note.get("notation_omitted")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ids-out", type=Path,
                        help="also write changed trajectory IDs, one per line")
    args = parser.parse_args()
    old, new = load_tree(args.old), load_tree(args.new)
    changed = []
    for trajectory_id, row in new.items():
        before = omitted(old[trajectory_id]) if trajectory_id in old else set()
        after = omitted(row)
        old_range = (old.get(trajectory_id, {}).get("input", {}).get("event_range")
                     if trajectory_id in old else None)
        new_range = row.get("input", {}).get("event_range")
        old_indices = [int(n["index"]) for n in (old.get(trajectory_id, {})
                                                    .get("input", {})
                                                    .get("notes_without_jianzi") or [])]
        new_indices = [int(n["index"]) for n in (row.get("input", {})
                                                    .get("notes_without_jianzi") or [])]
        if before != after or old_range != new_range or old_indices != new_indices:
            changed.append({
                "trajectory_id": trajectory_id,
                "score_key": row.get("score_key"),
                "before": sorted(before),
                "after": sorted(after),
                "added": sorted(after - before),
                "removed": sorted(before - after),
                "event_range_changed": old_range != new_range,
                "note_indices_changed": old_indices != new_indices,
            })
    report = {
        "old_rows": len(old), "new_rows": len(new),
        "changed_rows": len(changed), "changed": changed,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.ids_out:
        args.ids_out.write_text(
            "\n".join(item["trajectory_id"] for item in changed) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
