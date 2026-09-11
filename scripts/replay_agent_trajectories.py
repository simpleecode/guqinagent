#!/usr/bin/env python3
"""Replay inferred patches and emit a machine-readable acceptance report."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.trajectory_replay import (  # noqa: E402
    compare_replay_to_patch_targets, replay_patches,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ABC_J/agent_training/inferred")
    parser.add_argument("--output", type=Path, default=ROOT / "ABC_J/agent_training/replay_report.json")
    parser.add_argument("--max-failures", type=int, default=100)
    args = parser.parse_args()
    counts = Counter()
    failure_codes = Counter()
    failures = []
    by_split = {}
    for split in ("train", "validation", "test"):
        split_counts = Counter()
        path = args.input_dir / f"inferred_trajectories_{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                item = json.loads(line)
                counts["trajectories"] += 1
                split_counts["trajectories"] += 1
                replay = replay_patches(item["baseline_plan"], item["patches"])
                mismatches = compare_replay_to_patch_targets(
                    replay.actions, item["patches"]
                ) if replay.valid else []
                accepted = replay.valid and not mismatches
                key = "accepted" if accepted else "rejected"
                counts[key] += 1
                split_counts[key] += 1
                if not accepted:
                    problems = [*replay.errors, *mismatches]
                    failure_codes.update(problem["code"] for problem in problems)
                    if len(failures) < args.max_failures:
                        failures.append({
                            "trajectory_id": item["trajectory_id"], "split": split,
                            "line": line_number,
                            "problems": problems[:20],
                        })
        by_split[split] = dict(split_counts)
    report = {
        "schema_version": "replay-1.0", "valid": counts["rejected"] == 0,
        "counts": dict(counts), "by_split": by_split,
        "failure_codes": dict(failure_codes), "sample_failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
