#!/usr/bin/env python3
"""Validate trajectory JSONL files and detect split leakage."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "ABC_J" / "agent_training"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--prefix", default="reference_trajectories")
    args = parser.parse_args()
    ids: set[str] = set()
    families: dict[str, set[str]] = defaultdict(set)
    scores: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    errors: list[str] = []
    for split in ("train", "validation", "test"):
        path = args.input_dir / f"{args.prefix}_{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: {exc}")
                    continue
                common = {
                    "trajectory_version", "trajectory_id", "score_family_id",
                    "score_key", "phrase_id", "split", "input", "reference_plan",
                    "provenance",
                }
                if args.prefix == "inferred_trajectories":
                    required = common | {
                        "baseline_plan", "canonical_trajectory", "patches",
                        "unresolved_differences", "quality",
                    }
                else:
                    required = common | {"steps", "output_plan", "rewards"}
                missing = required - set(item)
                if missing:
                    errors.append(f"{path}:{line_number}: missing {sorted(missing)}")
                if item.get("split") != split:
                    errors.append(f"{path}:{line_number}: split mismatch")
                trajectory_id = item.get("trajectory_id")
                if trajectory_id in ids:
                    errors.append(f"duplicate trajectory_id: {trajectory_id}")
                ids.add(trajectory_id)
                families[item["score_family_id"]].add(split)
                scores[item["score_key"]].add(split)
                counts[split] += 1
    leaking_families = {
        family: sorted(splits) for family, splits in families.items() if len(splits) > 1
    }
    leaking_scores = {
        score: sorted(splits) for score, splits in scores.items() if len(splits) > 1
    }
    if leaking_families:
        errors.append(f"families in multiple splits: {leaking_families}")
    if leaking_scores:
        errors.append(f"scores in multiple splits: {leaking_scores}")
    report = {
        "valid": not errors,
        "trajectories": dict(counts),
        "unique_trajectory_ids": len(ids),
        "families": len(families),
        "scores": len(scores),
        "leaking_families": leaking_families,
        "leaking_scores": leaking_scores,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
