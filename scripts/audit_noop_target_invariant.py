#!/usr/bin/env python3
"""Fail closed when a Guqinizer no-op still has written-text targets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ABC_J.scripts.generate_teacher_tool_trajectories import infer_jianzi_text_patches  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="source JSONL; may be repeated for legacy/source overlays")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    public = read_jsonl(args.messages_dir / "messages_train.jsonl")
    audits = {row["sample_id"]: row for row in read_jsonl(
        args.messages_dir / "teacher_trajectory_audit.jsonl")}
    sources = {}
    for source_path in args.source:
        for row in read_jsonl(source_path):
            # The first source is authoritative; later files are only
            # legacy/supplemental overlays for IDs absent from it.
            sources.setdefault(row["trajectory_id"], row)
    fingering = {}
    for sample in public:
        sid = str(sample.get("sample_id") or "")
        suffix = "-fingering_agent-teacher-tools"
        if sid.endswith(suffix):
            fingering[sid[:-len(suffix)]] = sample
    violations = []
    no_ops = 0
    for sample in public:
        if sample.get("agent_stage") != "guqinization":
            continue
        if (sample.get("termination") or {}).get("kind") != "no_changes":
            continue
        no_ops += 1
        tid = (sample.get("provenance") or {}).get("source_trajectory_id")
        source = sources.get(tid)
        base_sample = fingering.get(tid)
        base_audit = audits.get(base_sample.get("sample_id")) if base_sample else None
        base_plan = ((base_audit or {}).get("teacher_private") or {}).get("accepted_plan")
        if not source or not isinstance(base_plan, dict):
            violations.append({"trajectory_id": tid, "reason": "missing_source_or_base_plan"})
            continue
        targets = infer_jianzi_text_patches(
            base_plan, source.get("reference_plan", {}).get("actions", [])
        ).get("patches", [])
        if targets:
            violations.append({
                "trajectory_id": tid,
                "reason": "no_changes_with_surface_targets",
                "source_indices": [int(p["source_index"]) for p in targets],
            })
    report = {"no_op_count": no_ops, "violations": violations,
              "valid": not violations}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
