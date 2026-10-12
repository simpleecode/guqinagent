#!/usr/bin/env python3
"""Run selected two-stage and single-stage retries sequentially (7 workers)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    for method in ("two_stage_audited", "single_stage_audited"):
        base = ROOT / "ABC_J/agent_training" / method / "walk_return_retry"
        report = json.loads((base / "selection_report.json").read_text(encoding="utf-8"))
        count = report["affected_phrase_count"]
        if not count:
            continue
        command = [sys.executable, "-u", str(ROOT / "ABC_J/scripts/run_teacher_batch_parallel.py"),
                   "--input", str(base / "score_context_input.jsonl"),
                   "--trajectory-id-file", str(base / "affected_phrase_ids.txt"),
                   "--output-dir", str(base / "raw"), "--target-count", str(count),
                   "--workers", "7", "--shard-by-trajectory-id", "--worker-prefix", "walk7",
                   "--model", "glm-5.3", "--min-interval", "0.5",
                   "--max-tool-rounds", "24", "--max-attempts", "3"]
        if method == "single_stage_audited":
            command.extend(["--stage", "single_stage"])
        print(f"START {method}: {count} phrases", flush=True)
        with (base / "run.log").open("a", encoding="utf-8") as log:
            status = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"EXIT {method}: {status}", flush=True)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
