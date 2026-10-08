"""Export the sealed-test protocol files from the new train/test split.

The 2026-10-07 re-split (``renderfix_train_test_20261007/pitch_eligible_two_or_half``)
ships one inferred-trajectories JSONL per split, with the private
``reference_plan`` in every row.  This script derives the two files the
evaluation pipeline consumes, mirroring the previous ``agent-eval-input-2.0``
and ``evaluation_pairs`` formats:

- ``test_runtime.jsonl`` — public inputs: runtime rows *without* reference
  material, for ``run_final_eval --input`` (and the production evaluator).
- ``evaluation_pairs_test.jsonl`` — reference pairs (runtime_item + sealed
  reference actions), for ``evaluation.run_eval --reference``.

    python -m baseline.gepa_guqin.export_eval_inputs \
        --out-dir train/eval_inputs_v2_text_protocol/renderfix_20261007
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = (ROOT / "ABC_J/agent_training/renderfix_train_test_20261007"
                  / "pitch_eligible_two_or_half/inferred_trajectories_test.jsonl")
INPUT_SCHEMA = "agent-eval-input-2.0"


def input_sha256(row: dict[str, Any]) -> str:
    payload = json.dumps(row["input"], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export(source: Path, out_dir: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in
            source.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if not ((row.get("reference_plan") or {}).get("actions")):
            raise SystemExit(f"test row without reference actions: {row.get('trajectory_id')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = out_dir / "test_runtime.jsonl"
    pairs_path = out_dir / "evaluation_pairs_test.jsonl"
    with runtime_path.open("w", encoding="utf-8", newline="\n") as runtime_out, \
            pairs_path.open("w", encoding="utf-8", newline="\n") as pairs_out:
        for row in rows:
            sample_id = str(row["trajectory_id"])
            runtime_item = {
                "trajectory_id": row["trajectory_id"],
                "score_key": row["score_key"],
                "phrase_id": row["phrase_id"],
                "split": row.get("split") or "test",
                "input": row["input"],
                "baseline_plan": row.get("baseline_plan") or {},
            }
            runtime_out.write(json.dumps({
                "schema_version": INPUT_SCHEMA,
                "sample_id": sample_id,
                "split": runtime_item["split"],
                "score_key": row["score_key"],
                "phrase_id": row["phrase_id"],
                "input_sha256": input_sha256(row),
                "runtime_item": runtime_item,
            }, ensure_ascii=False) + "\n")
            pairs_out.write(json.dumps({
                "sample_id": sample_id,
                "split": runtime_item["split"],
                "score_key": row["score_key"],
                "phrase_id": row["phrase_id"],
                "input_sha256": input_sha256(row),
                "runtime_item": {
                    "trajectory_id": row["trajectory_id"],
                    "score_key": row["score_key"],
                    "phrase_id": row["phrase_id"],
                    "split": runtime_item["split"],
                    "input": row["input"],
                },
                "reference": {"actions": row["reference_plan"]["actions"]},
            }, ensure_ascii=False) + "\n")
    scores = {str(row["score_key"]) for row in rows}
    summary = {
        "source": str(source), "phrases": len(rows), "scores": len(scores),
        "test_runtime": str(runtime_path), "evaluation_pairs": str(pairs_path),
    }
    (out_dir / "export_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.out_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
