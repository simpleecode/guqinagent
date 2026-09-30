#!/usr/bin/env python3
"""Summarize tool-use reliability from two-stage evaluation predictions.

The evaluator stores every successfully parsed model tool turn in
``base.trace`` and ``guqinizer.trace``.  This script deliberately separates
model-emitted calls from calls the local tool runtime actually executed: a
model can name a tool correctly yet provide invalid arguments.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def prediction_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    workers = path / "workers"
    if workers.is_dir():
        files = sorted(workers.glob("predictions_*.jsonl"))
        if files:
            return files
    files = sorted(path.glob("predictions*.jsonl"))
    if files:
        return files
    raise FileNotFoundError(f"No prediction JSONL found under {path}")


def load_records(files: Iterable[Path]) -> list[dict]:
    # Retries can leave duplicate sample IDs.  The last physical record is the
    # newest one and is the only one included in the denominator.
    rows: dict[str, dict] = {}
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
    return list(rows.values())


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summary_for_stage(records: list[dict], stage: str) -> dict:
    result: Counter = Counter()
    per_tool: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        payload = record.get(stage)
        if not isinstance(payload, dict):
            continue
        trace = payload.get("trace")
        if not isinstance(trace, list):
            continue
        result["phrases_with_trace"] += 1
        for turn in trace:
            if not isinstance(turn, dict):
                continue
            result["turns"] += 1
            result["truncated_turns"] += bool(turn.get("truncated"))
            result["generation_errors"] += bool(turn.get("generation_error"))
            if turn.get("generation_error"):
                result["tool_format_errors"] += (
                    "invalid tool arguments" in str(turn["generation_error"])
                )
            calls = turn.get("tool_calls") or []
            outcomes = turn.get("tool_results") or []
            result["model_emitted_calls"] += len(calls)
            result["runtime_executed_calls"] += len(outcomes)
            for outcome in outcomes:
                if not isinstance(outcome, dict):
                    continue
                name = str(outcome.get("name") or "<unknown>")
                tool = per_tool[name]
                tool["executed"] += 1
                response = outcome.get("result")
                ok = isinstance(response, dict) and response.get("ok") is True
                tool["ok"] += ok
                tool["error"] += not ok
                result["runtime_ok"] += ok
                result["runtime_error"] += not ok
                if name == "edit_plan":
                    result["edit_plan_calls"] += 1
                    valid = (
                        ok
                        and isinstance(response.get("result"), dict)
                        and response["result"].get("valid") is True
                    )
                    result["edit_plan_valid"] += valid

    body = dict(result)
    body.update({
        "tool_execution_success_rate": safe_rate(
            result["runtime_ok"], result["runtime_executed_calls"]
        ),
        "truncated_turn_rate": safe_rate(
            result["truncated_turns"], result["turns"]
        ),
        "edit_plan_valid_rate": safe_rate(
            result["edit_plan_valid"], result["edit_plan_calls"]
        ),
        "tool_format_failure_rate": safe_rate(
            result["tool_format_errors"], result["turns"]
        ),
        "by_tool": {
            name: {
                **dict(stats),
                "execution_success_rate": safe_rate(stats["ok"], stats["executed"]),
            }
            for name, stats in sorted(per_tool.items())
        },
    })
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="predictions.jsonl or an evaluation output directory")
    parser.add_argument("--output", type=Path,
                        help="defaults to <input>/tool_call_stats.json")
    args = parser.parse_args()
    files = prediction_files(args.input)
    records = load_records(files)
    report = {
        "input_files": [str(path) for path in files],
        "records": len(records),
        "protocol_valid_records": sum(bool(row.get("protocol_valid")) for row in records),
        "stages": {
            stage: summary_for_stage(records, stage)
            for stage in ("base", "guqinizer")
        },
        "notes": [
            "Execution success counts only calls that reached tool_results.",
            "New evaluator traces record malformed server-side tool JSON as generation_errors; older runs may require worker-log inspection.",
        ],
    }
    output = args.output or (args.input if args.input.is_dir() else args.input.parent) / "tool_call_stats.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
