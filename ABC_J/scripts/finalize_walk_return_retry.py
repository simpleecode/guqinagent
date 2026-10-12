#!/usr/bin/env python3
"""GLM-redact selected retries, validate, and replace the canonical teacher set."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "ABC_J/agent_training/two_stage_audited/teacher_train_final"
RETRY = BASE.parent / "walk_return_retry"


def ids(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line)["sample_id"] for line in handle if line.strip()]


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], cwd=ROOT, check=True)


def main():
    global BASE, RETRY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['two_stage', 'single_stage'], default='two_stage')
    args = parser.parse_args()
    single = args.method == 'single_stage'
    if single:
        BASE = ROOT / 'ABC_J/agent_training/single_stage_audited/teacher_final_redacted'
        RETRY = BASE.parent / 'walk_return_retry'
    expected = 64 if single else 174
    raw = RETRY / "raw"
    report = json.loads((raw / "generation_report.json").read_text())
    if report.get("failures") or report["accepted"] != expected:
        raise RuntimeError(f"{args.method} retry incomplete")
    original_ids = ids(BASE / "messages_train.jsonl")
    replacement_ids = ids(raw / "messages_train.jsonl")
    if len(replacement_ids) != expected or len(set(replacement_ids)) != expected or not set(replacement_ids) <= set(original_ids):
        raise RuntimeError("replacement IDs duplicate or absent from canonical set")
    worker_root = RETRY / "redaction_workers"
    worker_root.mkdir(exist_ok=True)
    processes = []
    for shard in range(7):
        log = (worker_root / f"worker_{shard}.log").open("a", encoding="utf-8")
        command = [sys.executable, "-u", str(ROOT / "ABC_J/scripts/redact_teacher_reasoning.py"),
                   "--input-dir", str(raw), "--output-dir", str(worker_root / f"worker_{shard}"),
                   "--model", "glm-5.3", "--shard-count", "7", "--shard-index", str(shard)]
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        processes.append((shard, process, log))
        print(f"redaction worker {shard}: pid {process.pid}", flush=True)
    statuses = []
    for shard, process, log in processes:
        status = process.wait()
        log.close()
        statuses.append(status)
        print(f"redaction worker {shard}: exit {status}", flush=True)
    if any(statuses):
        raise RuntimeError(f"redaction failed: {statuses}")
    redacted = RETRY / "redacted"
    run("scripts/merge_redaction_workers.py", "--source", raw,
        "--worker-root", worker_root, "--output-dir", redacted)
    run("scripts/validate_teacher_agent_messages.py", "--input-dir", redacted)
    merged = RETRY / "merged_candidate"
    run("scripts/merge_repaired_guqinizer_dataset.py", "--base", BASE,
        "--replacement", redacted, "--output", merged)
    run("scripts/validate_teacher_agent_messages.py", "--input-dir", merged)
    if ids(merged / "messages_train.jsonl") != original_ids:
        raise RuntimeError("canonical sample count/order unexpectedly changed")
    # Check tool actions and all non-reasoning message content remain intact.
    raw_rows = {r["sample_id"]: r for r in map(json.loads, (raw / "messages_train.jsonl").open())}
    with (redacted / "messages_train.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            original = raw_rows[row["sample_id"]]
            if len(row["messages"]) != len(original["messages"]):
                raise RuntimeError("redaction changed message count")
            for before, after in zip(original["messages"], row["messages"]):
                if before.get("role") != "assistant":
                    if before != after:
                        raise RuntimeError("redaction changed non-assistant message")
                elif before.get("tool_calls") != after.get("tool_calls"):
                    raise RuntimeError("redaction changed tool arguments")
    backup = RETRY / "previous_canonical"
    if backup.exists():
        raise RuntimeError("refusing to replace existing canonical backup")
    BASE.rename(backup)
    try:
        merged.rename(BASE)
    except BaseException:
        backup.rename(BASE)
        raise
    completion = {"status": "complete", "replaced_stage_trajectories": expected,
                  "replaced_phrases": expected if single else expected // 2, "canonical_stage_trajectories": len(original_ids),
                  "canonical_path": str(BASE), "backup_path": str(backup),
                  "redaction_report": str(redacted / "reasoning_redaction_report.json")}
    (RETRY / "finalization_report.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(completion, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
