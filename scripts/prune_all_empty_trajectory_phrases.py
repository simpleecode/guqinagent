#!/usr/bin/env python3
"""Copy a trajectory corpus while removing both stages of all-empty phrases."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_teacher_tool_trajectories as generator


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def trajectory_id(row: dict) -> str | None:
    if row.get("trajectory_id"):
        return str(row["trajectory_id"])
    sample_id = str(row.get("sample_id") or "")
    for stage in ("fingering_agent", "guqinization"):
        suffix = f"-{stage}-teacher-tools"
        if sample_id.endswith(suffix):
            return sample_id[:-len(suffix)]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    sources = {row["trajectory_id"]: row for row in rows(args.source)}
    removed = {
        key for key, item in sources.items()
        if generator.is_all_empty_reference_phrase(item)
    }
    removed_present = {
        value for row in rows(args.input / "messages_train.jsonl")
        if (value := trajectory_id(row)) in removed
    }
    counts: dict[str, dict[str, int]] = {}
    jsonl_names = {
        "messages_train.jsonl", "teacher_trajectory_audit.jsonl",
        "fingering_intermediates.jsonl", "checkpoint.jsonl",
        "teacher_rejected_io.jsonl",
    }
    for source_path in args.input.iterdir():
        target = args.output / source_path.name
        if source_path.name not in jsonl_names or not source_path.is_file():
            if source_path.is_file():
                shutil.copy2(source_path, target)
            continue
        source_rows = rows(source_path)
        kept = [row for row in source_rows if trajectory_id(row) not in removed]
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
            encoding="utf-8",
        )
        counts[source_path.name] = {
            "before": len(source_rows), "after": len(kept),
            "removed": len(source_rows) - len(kept),
        }
    report = {
        "schema_version": "all-empty-phrase-prune-1.0",
        "input": str(args.input), "source": str(args.source),
        "all_empty_source_ids": len(removed),
        "all_empty_ids_present_in_input": len(removed_present),
        "removed_ids": sorted(removed), "files": counts,
    }
    (args.output / "all_empty_prune_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    generation_report_path = args.output / "generation_report.json"
    if generation_report_path.exists():
        generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
        public_rows = rows(args.output / "messages_train.jsonl")
        by_stage: dict[str, int] = {}
        for row in public_rows:
            stage = str(row.get("agent_stage") or "unknown")
            by_stage[stage] = by_stage.get(stage, 0) + 1
        generation_report["accepted"] = len(public_rows)
        generation_report["by_stage"] = by_stage
        generation_report["all_empty_pruned"] = len(removed_present)
        generation_report["all_empty_prune_report"] = str(
            args.output / "all_empty_prune_report.json"
        )
        generation_report_path.write_text(
            json.dumps(generation_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
