#!/usr/bin/env python3
"""Run generate_teacher_tool_trajectories.py in parallel shards, then merge.

Each worker is a full generator process that receives --shard-count/--shard-index
and writes into <output-dir>/shard_<k>. After all workers exit, the shard files
are concatenated into the output dir root and the per-shard generation reports
are summed into one generation_report.json.

Usage (repo root, guqin-agent env):

  python scripts/run_teacher_generation_parallel.py \
      --workers 4 \
      --output-dir ABC_J/agent_training/messages_pilot_v1 \
      -- --basic-intermediate --limit 40 \
         --max-tool-rounds 24 --max-attempts 2

Everything after `--` is passed through to the generator verbatim. Pass pilot
IDs via repeated --trajectory-id flags, or let shards interleave the full pool.
--merge-only re-runs just the merge step (e.g. after re-running one shard by
hand); --dry-run prints the planned worker commands without executing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "ABC_J" / "scripts" / "generate_teacher_tool_trajectories.py"
MERGE_FILES = ("messages_train.jsonl", "teacher_trajectory_audit.jsonl",
               "fingering_intermediates.jsonl", "teacher_rejected_io.jsonl")


def merge_reports(report_paths: list[Path]) -> dict:
    merged: dict = {}
    by_stage: Counter = Counter()
    failures: list[dict] = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not merged:
            merged = {"schema_version": report.get("schema_version"),
                      "model": report.get("model"), "workflow": report.get("workflow")}
        merged["source_phrases"] = merged.get("source_phrases", 0) + int(report.get("source_phrases", 0))
        merged["accepted"] = merged.get("accepted", 0) + int(report.get("accepted", 0))
        merged["rejected"] = merged.get("rejected", 0) + int(report.get("rejected", 0))
        by_stage.update(report.get("by_stage") or {})
        failures.extend(report.get("failures") or [])
    merged["by_stage"] = dict(by_stage)
    merged["failures"] = failures
    return merged


def merge(output_dir: Path, workers: int) -> None:
    shard_dirs = [output_dir / f"shard_{index:02d}" for index in range(workers)]
    existing = [d for d in shard_dirs if d.is_dir()]
    if not existing:
        raise SystemExit(f"no shard directories found under {output_dir}")
    for name in MERGE_FILES:
        target = output_dir / name
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing {target}; delete it first")
        with target.open("w", encoding="utf-8", newline="\n") as out:
            for shard in existing:
                source = shard / name
                if not source.exists():
                    continue
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.write(line + "\n")
    report_paths = [shard / "generation_report.json"
                    for shard in existing if (shard / "generation_report.json").exists()]
    merged = merge_reports(report_paths)
    (output_dir / "generation_report.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merged {len(existing)} shards: accepted {merged['accepted']}, "
          f"rejected {merged['rejected']}, by_stage {merged['by_stage']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merge-only", action="store_true",
                        help="only merge existing shard outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("generator_args", nargs=argparse.REMAINDER,
                        help="arguments passed verbatim to the generator (after --)")
    args = parser.parse_args()
    generator_args = [arg for arg in args.generator_args if arg != "--"]

    if args.merge_only:
        merge(args.output_dir, args.workers)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    for index in range(args.workers):
        shard_dir = args.output_dir / f"shard_{index:02d}"
        command = [sys.executable, str(GENERATOR), *generator_args,
                   "--shard-count", str(args.workers), "--shard-index", str(index),
                   "--output-dir", str(shard_dir)]
        commands.append((shard_dir, command))

    if args.dry_run:
        for _, command in commands:
            print(" ".join(command))
        return 0

    processes = []
    for shard_dir, command in commands:
        shard_dir.mkdir(parents=True, exist_ok=True)
        log = (shard_dir / "worker.log").open("w", encoding="utf-8")
        processes.append((shard_dir, log,
                          subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)))
    exit_codes = []
    for shard_dir, log, process in processes:
        code = process.wait()
        log.close()
        exit_codes.append((shard_dir.name, code))
    failed = [(name, code) for name, code in exit_codes if code != 0]
    for name, code in exit_codes:
        print(f"{name}: exit {code}")
    if failed:
        print("some shards failed; fix them and re-run with --merge-only, "
              "or inspect their worker.log files")
        return 1
    merge(args.output_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
