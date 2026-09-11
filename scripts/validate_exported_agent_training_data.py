#!/usr/bin/env python3
"""Validate exported SFT views for identity, split, tuning, and answer leakage."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ABC_J/agent_training/exported")
    args = parser.parse_args()
    ids, families = set(), defaultdict(set)
    counts, errors = Counter(), []
    patterns_by_split = {
        "train": {
            "full_trajectory": "sft_full_trajectory_train.jsonl",
            "patch": "sft_patch_train.jsonl",
            "direct_generation": "sft_direct_generation_train.jsonl",
            "tuning_decision": "sft_tuning_decision_train.jsonl",
        },
        "validation": {"evaluation_pairs": "evaluation_pairs_validation.jsonl"},
        "test": {"evaluation_pairs": "evaluation_pairs_test.jsonl"},
    }
    for split, patterns in patterns_by_split.items():
        for view, pattern in patterns.items():
            path = args.input_dir / pattern.format(split=split)
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    item = json.loads(line)
                    identity = (view, item["sample_id"])
                    if identity in ids:
                        errors.append(f"duplicate {identity}")
                    ids.add(identity)
                    provenance = item.get("provenance") or {}
                    actual_split = provenance.get("split", item.get("split"))
                    family = provenance.get("score_family_id", item.get("score_family_id"))
                    if actual_split != split:
                        errors.append(f"{path}:{line_number}: split mismatch")
                    families[family].add(split)
                    tuning = (item.get("input") or {}).get("tuning")
                    if view not in {"tuning_decision", "evaluation_pairs"}:
                        if not tuning or len(tuning.get("open_midi") or []) != 7:
                            errors.append(f"{path}:{line_number}: invalid tuning")
                        serialized_input = json.dumps(item["input"], ensure_ascii=False)
                        if '"jianzi"' in serialized_input or '"reference_plan"' in serialized_input:
                            errors.append(f"{path}:{line_number}: answer leakage")
                    if view == "evaluation_pairs" and item.get("training_allowed") is not False:
                        errors.append(f"{path}:{line_number}: evaluation pair permits training")
                    counts[f"{split}.{view}"] += 1
    leakage = {family: sorted(splits) for family, splits in families.items()
               if family and len(splits) > 1}
    if leakage:
        errors.append(f"families across splits: {leakage}")
    report = {"schema_version": "sft-validation-1.0", "valid": not errors,
              "counts": dict(counts), "unique_view_sample_ids": len(ids),
              "families": len([key for key in families if key]),
              "leaking_families": leakage, "errors": errors[:200]}
    output = args.input_dir / "agent_training_validation_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
