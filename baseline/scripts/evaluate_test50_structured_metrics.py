#!/usr/bin/env python3
"""Run the repository's structured final evaluator on a saved test50 arm.

This does not call an LLM and never exposes sealed labels to the actor.  It
only converts already-saved ``rows_<arm>.json`` predictions to the public
prediction JSONL schema, then invokes ``evaluation.run_eval``.  Partial runs are scored on valid contiguous prefixes with stateful replay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO)]

from evaluation.run_eval import run  # noqa: E402


DEFAULT_OUT = REPO / "baseline/runs/N_gepa16_from_original/test50_top3"
DEFAULT_REFERENCE = REPO / "train/eval_inputs_v2_text_protocol/test50/evaluation_pairs_test.jsonl"


def snapshot_predictions(rows_path: Path, output: Path) -> int:
    """Write a stable evaluator input from one atomically saved arm snapshot."""
    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    by_id: dict[str, dict] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            continue
        prediction = row.get("prediction") or {}
        by_id[sample_id] = {
            "sample_id": sample_id,
            "protocol_valid": bool(row.get("protocol_ok")),
            "jianzi_rows": prediction.get("jianzi_rows") or [],
        }
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_id in sorted(by_id):
            handle.write(json.dumps(by_id[sample_id], ensure_ascii=False) + "\n")
    return len(by_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="N55")
    parser.add_argument("--test50-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--name", default=None, help="output subdirectory name; defaults to <arm>_structured_partial")
    args = parser.parse_args()

    arm = str(args.arm)
    rows_path = args.test50_out / f"rows_{arm}.json"
    if not rows_path.exists():
        raise SystemExit(f"missing saved arm rows: {rows_path}")
    report_root = args.test50_out / "structured_metrics"
    report_root.mkdir(parents=True, exist_ok=True)
    pred_path = report_root / f"{arm}_predictions_snapshot.jsonl"
    count = snapshot_predictions(rows_path, pred_path)
    name = args.name or f"{arm}_structured_partial"
    result = run(argparse.Namespace(
        pred=pred_path,
        reference=args.reference,
        experiment=name,
        output_root=report_root,
        model=f"{arm} prompt / saved test50 snapshot",
        pitch_tolerance=50.0,
        strict_prefix=True,
    ))
    print(json.dumps({"saved_predictions": count, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
