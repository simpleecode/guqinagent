#!/usr/bin/env python3
"""Run disjoint teacher-generation shards and merge them without duplicate IDs.

Each worker receives the full source file (so historical context remains
available) plus a small trajectory-id file. Worker output directories are
isolated; only this coordinator writes the shared output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "ABC_J" / "scripts" / "generate_teacher_tool_trajectories.py"
FILES = ("messages_train.jsonl", "teacher_trajectory_audit.jsonl",
         "fingering_intermediates.jsonl", "teacher_rejected_io.jsonl",
         "checkpoint.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_unique(base: Path, shard_paths: list[Path], key: str) -> int:
    existing = read_jsonl(base)
    seen = {str(row.get(key)) for row in existing if row.get(key) is not None}
    added = 0
    with base.open("a", encoding="utf-8", newline="\n") as handle:
        for shard in shard_paths:
            for row in read_jsonl(shard):
                value = row.get(key)
                if value is None or str(value) in seen:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add(str(value))
                added += 1
    return added


def merge_checkpoint(base: Path, shard_paths: list[Path]) -> int:
    existing = read_jsonl(base)
    seen = {str(row.get("trajectory_id")) for row in existing
            if row.get("trajectory_id") is not None}
    added = 0
    with base.open("a", encoding="utf-8", newline="\n") as handle:
        for shard in shard_paths:
            latest: dict[str, dict] = {}
            for row in read_jsonl(shard):
                if row.get("trajectory_id") is not None:
                    latest[str(row["trajectory_id"])] = row
            for trajectory_id, row in latest.items():
                if trajectory_id in seen:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add(trajectory_id)
                added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, required=True,
                        help="total source phrase target, including already completed IDs")
    parser.add_argument("--trajectory-id-file", type=Path,
                        help="newline-delimited IDs to rerun; full input is still supplied for context")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--max-tool-rounds", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--allow-private-reasoning-leakage", action="store_true")
    parser.add_argument("--stage", choices=("fingering_agent", "guqinization"))
    parser.add_argument("--intermediate-input", type=Path)
    parser.add_argument("--include-guqinizer-no-op", action="store_true")
    parser.add_argument("--score-shard-count", type=int,
                        help="partition source rows by stable score hash so workers do not each load the full corpus")
    parser.add_argument("--shard-by-trajectory-id", action="store_true",
                        help="balance selected IDs across workers while every worker retains full score context")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    score_shard_count = args.score_shard_count or args.workers
    if score_shard_count < 1 or score_shard_count != args.workers:
        raise SystemExit("--score-shard-count must equal --workers when using parallel workers")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(args.input)
    source_ids = [str(row["trajectory_id"]) for row in source_rows]
    base_checkpoint = args.output_dir / "checkpoint.jsonl"
    completed = {str(row.get("trajectory_id")) for row in read_jsonl(base_checkpoint)
                 if row.get("trajectory_id") is not None}
    remaining = [trajectory_id for trajectory_id in source_ids if trajectory_id not in completed]
    if args.trajectory_id_file:
        wanted = {line.strip() for line in args.trajectory_id_file.read_text(encoding="utf-8").splitlines()
                  if line.strip()}
        remaining = [trajectory_id for trajectory_id in remaining if trajectory_id in wanted]
    remaining = remaining[:max(0, args.target_count - len(completed))]
    if not remaining:
        print(json.dumps({"remaining": 0, "message": "没有需要并行处理的片段"}, ensure_ascii=False))
        return 0

    parallel_root = args.output_dir / ".parallel_workers"
    parallel_root.mkdir(parents=True, exist_ok=True)
    def bucket(value: str) -> int:
        digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % score_shard_count

    source_by_id = {str(row["trajectory_id"]): row for row in source_rows}
    shards: list[list[str]] = [[] for _ in range(args.workers)]
    for trajectory_id in remaining:
        shard_index = bucket(trajectory_id if args.shard_by_trajectory_id else
                             source_by_id[trajectory_id].get("score_key", ""))
        shards[shard_index].append(trajectory_id)

    processes: list[tuple[int, subprocess.Popen, Path]] = []
    for index, ids in enumerate(shards):
        if not ids:
            continue
        worker_dir = parallel_root / f"worker_{index:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        id_file = worker_dir / "trajectory_ids.txt"
        id_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
        log_path = worker_dir / "worker.log"
        log_handle = log_path.open("a", encoding="utf-8", newline="\n")
        command = [sys.executable, str(GENERATOR),
                   "--input", str(args.input),
                   "--output-dir", str(worker_dir / "output"),
                   "--trajectory-id-file", str(id_file),
                   "--limit", str(len(ids)),
                   "--model", args.model,
                   "--max-tool-rounds", str(args.max_tool_rounds),
                   "--max-attempts", str(args.max_attempts),
                   "--min-interval", str(args.min_interval),
                   "--resume"]
        if not args.shard_by_trajectory_id:
            command.extend(("--score-shard-count", str(score_shard_count),
                            "--score-shard-index", str(index)))
        if args.allow_private_reasoning_leakage:
            command.append("--allow-private-reasoning-leakage")
        if args.stage:
            command.extend(("--stage", args.stage))
        if args.intermediate_input:
            command.extend(("--intermediate-input", str(args.intermediate_input)))
        if args.include_guqinizer_no_op:
            command.append("--include-guqinizer-no-op")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle,
                                   stderr=subprocess.STDOUT)
        processes.append((index, process, worker_dir))
        print(f"started worker {index}: {len(ids)} trajectories", flush=True)

    statuses = []
    for index, process, worker_dir in processes:
        code = process.wait()
        statuses.append(code)
        print(f"worker {index} exited with {code}", flush=True)

    shard_outputs = [worker_dir / "output" for _, _, worker_dir in processes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merge_unique(args.output_dir / "messages_train.jsonl",
                 [path / "messages_train.jsonl" for path in shard_outputs], "sample_id")
    merge_unique(args.output_dir / "teacher_trajectory_audit.jsonl",
                 [path / "teacher_trajectory_audit.jsonl" for path in shard_outputs], "sample_id")
    merge_unique(args.output_dir / "fingering_intermediates.jsonl",
                 [path / "fingering_intermediates.jsonl" for path in shard_outputs], "trajectory_id")
    # Rejected traces are attempt-level audit records; preserve them all.
    rejected_path = args.output_dir / "teacher_rejected_io.jsonl"
    with rejected_path.open("a", encoding="utf-8", newline="\n") as handle:
        for path in shard_outputs:
            for row in read_jsonl(path / "teacher_rejected_io.jsonl"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    merge_checkpoint(args.output_dir / "checkpoint.jsonl",
                     [path / "checkpoint.jsonl" for path in shard_outputs])

    public = read_jsonl(args.output_dir / "messages_train.jsonl")
    checkpoints = read_jsonl(args.output_dir / "checkpoint.jsonl")
    latest_checkpoint = {str(row.get("trajectory_id")): row for row in checkpoints
                         if row.get("trajectory_id") is not None}
    failures: list[dict] = []
    for path in shard_outputs:
        report_path = path / "generation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            failures.extend(report.get("failures") or [])
    failure_by_stage: dict[tuple[str, str], dict] = {}
    for failure in failures:
        key = (str(failure.get("trajectory_id")), str(failure.get("stage")))
        failure_by_stage[key] = failure
    report = {
        "schema_version": "teacher-real-tools-report-1.0",
        "model": args.model,
        "workflow": "basic_intermediate_to_annotation_parallel",
        "raw_reasoning_leakage_allowed": bool(args.allow_private_reasoning_leakage),
        "workers": args.workers,
        "min_interval": args.min_interval,
        "source_phrases": len(latest_checkpoint),
        "accepted": len(public),
        "by_stage": dict(Counter(row.get("agent_stage") for row in public)),
        "quarantined": 0,
        "quarantine": [],
        "rejected": len(failure_by_stage),
        "failures": list(failure_by_stage.values()),
        "worker_exit_codes": statuses,
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"remaining_requested": len(remaining),
                      "checkpoint_unique": len(latest_checkpoint),
                      "accepted": len(public),
                      "failures": len(failure_by_stage),
                      "worker_exit_codes": statuses}, ensure_ascii=False, indent=2))
    return 0 if all(code == 0 for code in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
