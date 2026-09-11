#!/usr/bin/env python3
"""Watch parallel teacher outputs and render each newly observed pitch warning."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUALIZER = ROOT / "scripts" / "visualize_agent_trajectories.py"
MARKER = ":warning:音高不匹配"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A worker may be in the middle of writing its last line.
                    continue
    return rows


def warning_messages(row: dict) -> list[dict]:
    return [message for message in row.get("messages") or []
            if MARKER in json.dumps(message, ensure_ascii=False)]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    log_path = args.output_dir / "pitch_warning_monitor.log"

    def emit(text: str) -> None:
        print(text, flush=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(text + "\n")

    while True:
        workers = sorted(args.parallel_root.glob("worker_*/output"))
        for worker_output in workers:
            messages_path = worker_output / "messages_train.jsonl"
            audit_path = worker_output / "teacher_trajectory_audit.jsonl"
            for row in read_rows(messages_path):
                sample_id = str(row.get("sample_id") or "")
                if not sample_id or not warning_messages(row):
                    continue
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                emit(f"[pitch-warning] {sample_id} worker={worker_output.parent.name}")
                for message in warning_messages(row):
                    emit(json.dumps(message, ensure_ascii=False, separators=(",", ":")))
                output = args.output_dir / f"{sample_id}.html"
                command = [sys.executable, str(VISUALIZER), str(messages_path),
                           "--audit", str(audit_path), "--sample-id", sample_id,
                           "--show-tools", "--output", str(output)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                        text=True, encoding="utf-8")
                if result.returncode == 0:
                    emit(f"[pitch-warning-visualization] {output}")
                else:
                    emit(f"[pitch-warning-visualization-error] {result.stderr.strip()}")
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
