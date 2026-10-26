#!/usr/bin/env python3
"""Assemble verified worker predictions and replay final warnings in score order."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import validate_jianzi_only


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = list(read_jsonl(args.input))
    source_by_id = {str(row["sample_id"]): row for row in inputs}
    if len(source_by_id) != len(inputs):
        raise SystemExit("duplicate input sample IDs")
    predictions = {}
    for path in sorted((args.output_dir / "workers").glob("predictions_*.jsonl")):
        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            source = source_by_id.get(sample_id)
            if source is None or row.get("input_sha256") != source["input_sha256"]:
                raise SystemExit(f"unexpected input hash: {sample_id}")
            if sample_id in predictions or not row.get("protocol_valid"):
                raise SystemExit(f"duplicate or invalid prediction: {sample_id}")
            predictions[sample_id] = row
    if not predictions:
        raise SystemExit("no completed predictions")
    write_jsonl(args.output_dir / "predictions_snapshot.jsonl",
                (predictions[row["sample_id"]] for row in inputs
                 if row["sample_id"] in predictions))

    expected_ids = set(source_by_id)
    references = [row for row in read_jsonl(
        ROOT / "train/eval_inputs_v2_text_protocol/test50/evaluation_pairs_test.jsonl"
    ) if row["sample_id"] in expected_ids]
    if {row["sample_id"] for row in references} != expected_ids:
        raise SystemExit("sealed references do not cover all selected inputs")
    write_jsonl(args.output_dir / "references_subset.jsonl", references)

    history = {}
    by_score = defaultdict(lambda: {"expected": 0, "completed": 0,
                                    "warning_phrases": 0, "warning_count": 0,
                                    "codes": Counter()})
    cases = []
    for source in sorted(inputs, key=lambda row: (row["score_key"], row["phrase_id"])):
        score = source["score_key"]
        counts = by_score[score]
        counts["expected"] += 1
        prediction = predictions.get(source["sample_id"])
        if prediction is None:
            continue
        if not prediction.get("base", {}).get("ok") or not prediction.get("guqinizer", {}).get("ok"):
            raise SystemExit(f"stage not accepted: {source['sample_id']}")
        item = deepcopy(source["runtime_item"])
        item["baseline_plan"] = deepcopy(prediction["guqinizer"]["plan"])
        _, report = validate_jianzi_only(item, [], require_complete=False,
                                         historical=history)
        warnings = report["warnings"]
        counts["completed"] += 1
        counts["warning_count"] += len(warnings)
        counts["warning_phrases"] += bool(warnings)
        counts["codes"].update(str(w["code"]) for w in warnings)
        if warnings:
            cases.append({"sample_id": source["sample_id"], "warnings": warnings})
        item["reference_plan"] = deepcopy(prediction["guqinizer"]["plan"])
        history[(score, source["phrase_id"])] = item
    summary = {
        "completed": len(predictions), "expected": len(inputs),
        "by_score": {score: {**counts, "codes": dict(counts["codes"])}
                     for score, counts in sorted(by_score.items())},
        "cases": cases,
        "warning_count": sum(row["warning_count"] for row in by_score.values()),
        "warning_phrases": sum(row["warning_phrases"] for row in by_score.values()),
    }
    (args.output_dir / "final_warning_snapshot.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
