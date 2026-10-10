#!/usr/bin/env python3
"""Plan an incremental teacher-trajectory rebuild after a split/render update.

Existing trajectories are reusable only when their phrase remains in the new
training selection.  This writes disjoint phrase-ID lists for reuse, new
generation, and removal, plus a manifest that records inference changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.open(encoding="utf-8") if line.strip()}


def read_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            rows[str(row["trajectory_id"])] = row
    return rows


def stable_payload(row: dict) -> str:
    value = dict(row)
    value.pop("split", None)
    value.pop("provenance", None)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_ids(path: Path, values: set[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in sorted(values)), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-selected-ids", type=Path, required=True)
    parser.add_argument("--new-selected-ids", type=Path, required=True)
    parser.add_argument("--old-inferred", type=Path, required=True)
    parser.add_argument("--new-inferred", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    old_ids = read_ids(args.old_selected_ids)
    new_ids = read_ids(args.new_selected_ids)
    old_rows = read_rows(args.old_inferred)
    new_rows = read_rows(args.new_inferred)
    missing = (old_ids - set(old_rows)) | (new_ids - set(new_rows))
    if missing:
        raise ValueError(f"selected IDs missing inferred rows: {sorted(missing)[:10]}")

    reusable = old_ids & new_ids
    newly_required = new_ids - old_ids
    removed = old_ids - new_ids
    changed = {
        phrase_id for phrase_id in reusable
        if stable_payload(old_rows[phrase_id]) != stable_payload(new_rows[phrase_id])
    }

    args.output_dir.mkdir(parents=True)
    write_ids(args.output_dir / "reuse_phrase_ids.txt", reusable)
    write_ids(args.output_dir / "new_teacher_phrase_ids.txt", newly_required)
    write_ids(args.output_dir / "removed_from_train_phrase_ids.txt", removed)
    write_ids(args.output_dir / "reused_phrase_ids_with_inference_change.txt", changed)
    manifest = {
        "schema_version": "incremental-teacher-rebuild-1.0",
        "old_selected_count": len(old_ids),
        "new_selected_count": len(new_ids),
        "reuse_count": len(reusable),
        "new_teacher_generation_count": len(newly_required),
        "removed_from_train_count": len(removed),
        "reused_phrase_inference_changed_count": len(changed),
        "policy": {
            "reuse": "retain existing accepted teacher trajectories for phrases still in train",
            "new": "generate both teacher stages for phrases newly entering train",
            "removed": "omit phrases moved to test from the next training export",
            "changed": "report render/inference changes; do not force regeneration by itself",
        },
        "inputs": {key: str(getattr(args, key)) for key in (
            "old_selected_ids", "new_selected_ids", "old_inferred", "new_inferred",
        )},
    }
    (args.output_dir / "rebuild_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
