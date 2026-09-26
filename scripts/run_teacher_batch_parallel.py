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
TWO_STAGE_GENERATOR = ROOT / "ABC_J" / "scripts" / "generate_teacher_tool_trajectories.py"
SINGLE_STAGE_GENERATOR = ROOT / "ABC_J" / "scripts" / "generate_single_stage_teacher_trajectories.py"
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
    # Checkpoints are state records, unlike the public trajectories: a retry
    # must be able to supersede an earlier ``attempted_with_failure`` record.
    # Keep the append-only audit trail and let readers use the final record per
    # trajectory id.
    latest_existing = {
        str(row["trajectory_id"]): row for row in existing
        if row.get("trajectory_id") is not None
    }
    updated = 0
    with base.open("a", encoding="utf-8", newline="\n") as handle:
        for shard in shard_paths:
            latest: dict[str, dict] = {}
            for row in read_jsonl(shard):
                if row.get("trajectory_id") is not None:
                    latest[str(row["trajectory_id"])] = row
            for trajectory_id, row in latest.items():
                if latest_existing.get(trajectory_id) == row:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                latest_existing[trajectory_id] = row
                updated += 1
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int,
                        help="total source phrase target, including already completed IDs")
    parser.add_argument("--merge-only", action="store_true",
                        help="merge existing worker checkpoint outputs without starting workers")
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry only root checkpoint rows whose latest status is attempted_with_failure")
    parser.add_argument("--worker-prefix", default="worker",
                        help="directory prefix for this worker run; use a new prefix for an isolated retry pass")
    parser.add_argument("--trajectory-id-file", type=Path,
                        help="newline-delimited IDs to rerun; full input is still supplied for context")
    parser.add_argument("--score-key", action="append",
                        help="limit work to one or more complete score keys")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument("--max-tool-rounds", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--allow-private-reasoning-leakage", action="store_true")
    parser.add_argument("--stage", choices=("fingering_agent", "guqinization", "single_stage"))
    parser.add_argument("--intermediate-input", type=Path)
    parser.add_argument("--include-guqinizer-no-op", action="store_true")
    parser.add_argument("--score-shard-count", type=int,
                        help="partition source rows by stable score hash so workers do not each load the full corpus")
    parser.add_argument("--shard-by-trajectory-id", action="store_true",
                        help="balance selected IDs across workers while every worker retains full score context")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if not args.merge_only and (args.input is None or args.target_count is None):
        raise SystemExit("--input and --target-count are required unless --merge-only is used")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        parallel_root = args.output_dir / ".parallel_workers"
        shard_outputs = sorted(
            path / "output" for path in parallel_root.glob(f"{args.worker_prefix}_*")
            if (path / "output").is_dir()
        )
        if not shard_outputs:
            raise SystemExit(f"no worker outputs under {parallel_root}")
        counts = {
            "messages_train": merge_unique(
                args.output_dir / "messages_train.jsonl",
                [path / "messages_train.jsonl" for path in shard_outputs], "sample_id"),
            "teacher_trajectory_audit": merge_unique(
                args.output_dir / "teacher_trajectory_audit.jsonl",
                [path / "teacher_trajectory_audit.jsonl" for path in shard_outputs], "sample_id"),
            "fingering_intermediates": merge_unique(
                args.output_dir / "fingering_intermediates.jsonl",
                [path / "fingering_intermediates.jsonl" for path in shard_outputs], "trajectory_id"),
            "checkpoint": merge_checkpoint(
                args.output_dir / "checkpoint.jsonl",
                [path / "checkpoint.jsonl" for path in shard_outputs]),
        }
        print(json.dumps({"merge_only": True, "workers": len(shard_outputs),
                          "added": counts}, ensure_ascii=False))
        return 0
    score_shard_count = args.score_shard_count or args.workers
    if score_shard_count < 1 or score_shard_count != args.workers:
        raise SystemExit("--score-shard-count must equal --workers when using parallel workers")

    source_rows = read_jsonl(args.input)
    if args.score_key:
        selected_scores = set(args.score_key)
        source_rows = [row for row in source_rows
                       if str(row.get("score_key") or "") in selected_scores]
    source_ids = [str(row["trajectory_id"]) for row in source_rows]
    base_checkpoint = args.output_dir / "checkpoint.jsonl"
    checkpoint_rows = read_jsonl(base_checkpoint)
    latest_checkpoint = {
        str(row["trajectory_id"]): row for row in checkpoint_rows
        if row.get("trajectory_id") is not None
    }
    if args.retry_failed:
        remaining = [trajectory_id for trajectory_id in source_ids
                     if latest_checkpoint.get(trajectory_id, {}).get("status") == "attempted_with_failure"]
        completed = set(source_ids) - set(remaining)
    else:
        completed = set(latest_checkpoint)
        remaining = [trajectory_id for trajectory_id in source_ids if trajectory_id not in completed]
    if args.trajectory_id_file:
        wanted = {line.strip() for line in args.trajectory_id_file.read_text(encoding="utf-8").splitlines()
                  if line.strip()}
        remaining = [trajectory_id for trajectory_id in remaining if trajectory_id in wanted]
    if args.retry_failed:
        # Here target-count is the retry-pool cap, not the corpus total: the
        # non-failed rows deliberately do not consume the requested budget.
        remaining = remaining[:args.target_count]
    else:
        remaining = remaining[:max(0, args.target_count - len(completed))]
    if not remaining:
        print(json.dumps({"remaining": 0, "message": "没有需要并行处理的片段"}, ensure_ascii=False))
        return 0

    parallel_root = args.output_dir / ".parallel_workers"
    parallel_root.mkdir(parents=True, exist_ok=True)
    def bucket(value: str) -> int:
        digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % score_shard_count

    shards: list[list[str]] = [[] for _ in range(args.workers)]
    if args.shard_by_trajectory_id:
        # Every worker receives the full source corpus for historical context,
        # so score affinity is unnecessary here.  Round-robin assignment keeps
        # small resume sets balanced too (a hash can put their last few IDs in
        # only one or two workers).
        for index, trajectory_id in enumerate(remaining):
            shards[index % args.workers].append(trajectory_id)
    else:
        source_by_id = {str(row["trajectory_id"]): row for row in source_rows}
        for trajectory_id in remaining:
            shard_index = bucket(source_by_id[trajectory_id].get("score_key", ""))
            shards[shard_index].append(trajectory_id)

    processes: list[tuple[int, subprocess.Popen, Path]] = []
    for index, ids in enumerate(shards):
        if not ids:
            continue
        worker_dir = parallel_root / f"{args.worker_prefix}_{index:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        id_file = worker_dir / "trajectory_ids.txt"
        id_file.write_text("\n".join(ids) + "\n", encoding="utf-8")
        log_path = worker_dir / "worker.log"
        log_handle = log_path.open("a", encoding="utf-8", newline="\n")
        generator = SINGLE_STAGE_GENERATOR if args.stage == "single_stage" else TWO_STAGE_GENERATOR
        command = [sys.executable, str(generator),
                   "--input", str(args.input),
                   "--output-dir", str(worker_dir / "output"),
                   "--trajectory-id-file", str(id_file),
                   "--model", args.model,
                   "--max-tool-rounds", str(args.max_tool_rounds),
                   "--max-attempts", str(args.max_attempts),
                   "--min-interval", str(args.min_interval),
                   "--resume"]
        if args.retry_failed:
            command.append("--retry-failed")
        if not args.shard_by_trajectory_id:
            if args.stage == "single_stage":
                command.extend(("--shard-count", str(score_shard_count),
                                "--shard-index", str(index)))
            else:
                command.extend(("--score-shard-count", str(score_shard_count),
                                "--score-shard-index", str(index)))
        if args.allow_private_reasoning_leakage:
            command.append("--allow-private-reasoning-leakage")
        if args.stage and args.stage != "single_stage":
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
