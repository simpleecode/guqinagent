#!/usr/bin/env python3
"""Export replay-verified full, patch, and direct-generation SFT views."""
from __future__ import annotations

import argparse
import hashlib
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


def tuning_context(item: dict) -> dict:
    tuning = item["input"].get("normalized_tuning")
    if tuning:
        return tuning
    metadata = item["input"]["metadata"]
    rows = (metadata.get("tuning") or {}).get("open_strings") or []
    if len(rows) != 7:
        raise ValueError("trajectory has no valid seven-string tuning")
    note_pc = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
               "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
               "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
    open_midi = []
    for expected, row in enumerate(rows, 1):
        if int(row.get("string", expected)) != expected:
            raise ValueError("unordered tuning strings")
        pitch = str(row["pitch"]).upper().replace("♯", "#").replace("♭", "B")
        # These fields already encode the sounding open-string pitch.  The
        # offset is descriptive provenance, not another transposition to add.
        open_midi.append(12 * (int(row["octave"]) + 1) + note_pc[pitch])
    payload = json.dumps(open_midi, separators=(",", ":"))
    return {"name": str((metadata.get("tuning") or {}).get("name") or ""),
            "open_midi": open_midi,
            "fingerprint": hashlib.sha256(payload.encode()).hexdigest()[:16]}


def common_input(item: dict, tuning: dict) -> dict:
    return {
        "metadata": {key: value for key, value in item["input"]["metadata"].items()
                     if key != "tuning"},
        "tuning": tuning, "event_range": item["input"]["event_range"],
        "notes": item["input"]["notes_without_jianzi"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ABC_J/agent_training/inferred")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ABC_J/agent_training/exported")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        ("train", view): (args.output_dir / f"sft_{view}_train.jsonl").open(
            "w", encoding="utf-8", newline="\n")
        for view in ("full_trajectory", "patch", "direct_generation")
    }
    evaluation_handles = {
        split: (args.output_dir / f"evaluation_pairs_{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n")
        for split in ("validation", "test")
    }
    counts, patch_types, rejected = Counter(), Counter(), []
    try:
        for split in ("train", "validation", "test"):
            path = args.input_dir / f"inferred_trajectories_{split}.jsonl"
            with path.open(encoding="utf-8") as source:
                for line in source:
                    item = json.loads(line)
                    supervised = set(item["quality"]["supervised_patch_ids"])
                    verified = [patch for patch in item["patches"] if patch["patch_id"] in supervised]
                    replay = replay_patches(item["baseline_plan"], verified, patch_ids=supervised)
                    problems = [*replay.errors,
                                *compare_replay_to_patch_targets(replay.actions, verified)]
                    if problems:
                        rejected.append({"trajectory_id": item["trajectory_id"],
                                         "problems": problems[:10]})
                        continue
                    tuning = tuning_context(item)
                    inputs = common_input(item, tuning)
                    provenance = {
                        "source_trajectory_id": item["trajectory_id"],
                        "score_key": item["score_key"], "score_family_id": item["score_family_id"],
                        "split": split, "trajectory_version": item["trajectory_version"],
                        "export_version": "sft-1.0", "tuning_fingerprint": tuning["fingerprint"],
                    }
                    if split != "train":
                        evaluation = {
                            "schema_version": "agent-eval-pair-1.0",
                            "sample_id": item["trajectory_id"], "split": split,
                            "score_key": item["score_key"],
                            "score_family_id": item["score_family_id"],
                            "input": inputs,
                            "reference": {
                                "actions": item["reference_plan"]["actions"],
                                "masked_source_indices": item["quality"]["masked_source_indices"],
                            },
                            "metrics": ["pitch", "playability", "continuity", "yun_proxy"],
                            "training_allowed": False,
                            "provenance": provenance,
                        }
                        evaluation_handles[split].write(json.dumps(evaluation, ensure_ascii=False) + "\n")
                        counts[f"{split}.evaluation_pairs"] += 1
                        continue
                    full = {
                        "schema_version": "sft-full-1.0", "sample_id": item["trajectory_id"],
                        "task": "revise_baseline_with_tools", "input": {
                            **inputs, "baseline_plan": item["baseline_plan"]},
                        "output": {"patches": verified, "final_actions": replay.actions},
                        "supervised_patch_ids": sorted(supervised), "provenance": provenance,
                    }
                    direct = {
                        "schema_version": "sft-direct-1.0", "sample_id": item["trajectory_id"],
                        "task": "generate_jianzipu_plan", "input": inputs,
                        "output": {"actions": replay.actions}, "provenance": provenance,
                    }
                    handles[split, "full_trajectory"].write(json.dumps(full, ensure_ascii=False) + "\n")
                    handles[split, "direct_generation"].write(json.dumps(direct, ensure_ascii=False) + "\n")
                    counts[f"{split}.full_trajectory"] += 1
                    counts[f"{split}.direct_generation"] += 1
                    baseline_by_index = {int(action["source_index"]): action
                                         for action in item["baseline_plan"].get("actions", [])}
                    note_by_index = {int(note["index"]): note for note in inputs["notes"]}
                    for patch in verified:
                        index = int(patch["source_index"])
                        sample = {
                            "schema_version": "sft-patch-1.0",
                            "sample_id": f"{item['trajectory_id']}-{patch['patch_id']}",
                            "task": "propose_local_plan_patch",
                            "input": {"tuning": tuning, "target_note": note_by_index.get(index),
                                      "baseline_action": baseline_by_index.get(index),
                                      "local_notes": [note_by_index[key] for key in sorted(note_by_index)
                                                      if index - 2 <= key <= index + 2]},
                            "output": {"patch": patch}, "provenance": provenance,
                        }
                        handles[split, "patch"].write(json.dumps(sample, ensure_ascii=False) + "\n")
                        counts[f"{split}.patch"] += 1
                        patch_types[f"{split}.{patch['patch_type']}"] += 1
    finally:
        for handle in handles.values():
            handle.close()
        for handle in evaluation_handles.values():
            handle.close()
    report = {"schema_version": "sft-export-1.0", "valid": not rejected,
              "counts": dict(counts), "patch_types": dict(patch_types),
              "rejected": rejected[:100], "replay_required": True,
              "loss_policy": "verified_patches_only",
              "split_policy": "train_expanded; validation_and_test_evaluation_only"}
    (args.output_dir / "agent_training_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
