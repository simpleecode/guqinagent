"""Recompute and print deterministic violation details for a saved pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import prompt_candidate_paired_experiment as experiment
from baseline.gepa_guqin.metric import replay_events


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = experiment.select_rows("train", len(result["sample_ids"]), result["seed"])
    examples = {item["sample_id"]: item for item in map(experiment.prepare, rows)}
    output = []
    for row in result["rows"]:
        if not row.get("prediction") or not row.get("rule_violations"):
            continue
        example = examples[row["sample_id"]]
        events, violations, _ = replay_events(example, row["prediction"], {})
        excluded = set(example.get("audit", {}).get("pitch_mismatch_event_ids") or [])
        excluded |= set(example.get("audit", {}).get("violation_event_ids") or [])
        clean = [violation for violation in violations if int(violation["event_id"]) not in excluded]
        output.append({"sample_id": row["sample_id"], "arm": row["arm"],
                       "score": row["score"], "violations": clean})
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
