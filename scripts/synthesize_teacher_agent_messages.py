#!/usr/bin/env python3
"""Synthesize multi-stage teacher-forced Agent messages from verified trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.teacher_trajectory import (  # noqa: E402
    STAGE_PATCH_TYPES, synthesize_audit, synthesize_stage,
)
from agents.abc_to_jianzipu.trajectory_replay import replay_patches  # noqa: E402


def fingering_complete(plan: dict) -> bool:
    return all(
        (not action.get("attack") or bool(action.get("right_finger")))
        and (action.get("mode") != "stopped" or bool(action.get("left_finger")))
        for action in plan.get("actions", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "ABC_J/agent_training/inferred/inferred_trajectories_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ABC_J/agent_training/messages")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_path = args.output_dir / "messages_train.jsonl"
    private_path = args.output_dir / "teacher_trajectory_audit.jsonl"
    counts, failures = Counter(), []
    with (args.input.open(encoding="utf-8") as source,
          public_path.open("w", encoding="utf-8", newline="\n") as public_out,
          private_path.open("w", encoding="utf-8", newline="\n") as private_out):
        selected = 0
        for line in source:
            item = json.loads(line)
            selected += 1
            if args.limit is not None and selected > args.limit:
                break
            supervised = set(item["quality"]["supervised_patch_ids"])
            verified = [patch for patch in item["patches"] if patch["patch_id"] in supervised]
            records = []
            stage_item = deepcopy(item)
            for stage, kinds in STAGE_PATCH_TYPES.items():
                patches = [patch for patch in verified if patch["patch_type"] in kinds]
                if stage == "guqinization" and not fingering_complete(stage_item["baseline_plan"]):
                    continue
                if patches:
                    records.append(synthesize_stage(stage_item, stage, patches))
                    replay = replay_patches(stage_item["baseline_plan"], patches)
                    if not replay.valid:
                        failures.append(f"{item['trajectory_id']}:{stage}:stage_replay_failed")
                        break
                    stage_item["baseline_plan"] = {"actions": replay.actions}
            records.append(synthesize_audit(item, verified))
            for public, private in records:
                if not all(public["verification"].values()):
                    failures.append(public["sample_id"])
                    continue
                public_out.write(json.dumps(public, ensure_ascii=False) + "\n")
                private_out.write(json.dumps(private, ensure_ascii=False) + "\n")
                counts[public["agent_stage"]] += 1
    report = {"schema_version": "agent-messages-export-1.0", "valid": not failures,
              "source_phrases": selected if args.limit is None else min(selected, args.limit),
              "message_trajectories": sum(counts.values()), "by_stage": dict(counts),
              "failures": failures[:100], "public_output": str(public_path),
              "private_audit_output": str(private_path)}
    (args.output_dir / "messages_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
