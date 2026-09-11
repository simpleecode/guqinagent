#!/usr/bin/env python3
"""Score server-generated predictions against sealed local eval pairs."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


def canonical(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    value = re.sub(r"[\[\]（）()]", "", value)
    digits = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7"}
    for source, target in digits.items():
        value = value.replace(source + "弦", target + "弦")
    return re.sub(r"\s+", "", value)


def reference_map(pair: dict[str, Any]) -> dict[int, str]:
    return {
        int(item["source_index"]): str(item.get("text") or item.get("jianzi_text") or "")
        for item in pair.get("reference", {}).get("actions", [])
    }


def score(args: argparse.Namespace) -> int:
    pairs = {}
    for line in args.reference.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            pairs[row["sample_id"]] = row
    totals = Counter()
    per_sample = []
    predictions = {}
    for line in args.predictions.open(encoding="utf-8"):
        if not line.strip():
            continue
        pred = json.loads(line)
        # Resume/retry outputs are append-only; the latest attempt for a
        # phrase is authoritative and must not be counted twice.
        predictions[pred["sample_id"]] = pred
    for pred in predictions.values():
        pair = pairs.get(pred["sample_id"])
        if pair is None:
            continue
        refs = reference_map(pair)
        generated = {int(row[0]): str(row[1]) for row in pred.get("jianzi_rows", [])}
        indices = set(refs) | set(generated)
        exact = sum(canonical(refs.get(i)) == canonical(generated.get(i)) for i in indices)
        nonempty_ref = {i for i, text in refs.items() if canonical(text)}
        nonempty_pred = {i for i, text in generated.items() if canonical(text)}
        totals["samples"] += 1
        totals["protocol_valid"] += int(bool(pred.get("protocol_valid")))
        totals["rows_compared"] += len(indices)
        totals["exact_cells"] += exact
        totals["reference_nonempty"] += len(nonempty_ref)
        totals["predicted_nonempty"] += len(nonempty_pred)
        totals["nonempty_hits"] += len(nonempty_ref & nonempty_pred)
        per_sample.append({
            "sample_id": pred["sample_id"],
            "protocol_valid": bool(pred.get("protocol_valid")),
            "exact_cell_rate": exact / len(indices) if indices else None,
            "nonempty_recall": len(nonempty_ref & nonempty_pred) / len(nonempty_ref) if nonempty_ref else None,
            "predicted_rows": len(generated),
            "reference_rows": len(refs),
        })
    report = {
        "schema_version": "agent-eval-score-1.0",
        "reference": str(args.reference),
        "predictions": str(args.predictions),
        "aggregate": {
            **totals,
            "protocol_valid_rate": totals["protocol_valid"] / totals["samples"] if totals["samples"] else None,
            "exact_cell_rate": totals["exact_cells"] / totals["rows_compared"] if totals["rows_compared"] else None,
            "nonempty_recall": totals["nonempty_hits"] / totals["reference_nonempty"] if totals["reference_nonempty"] else None,
        },
        "per_sample": per_sample,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return score(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
