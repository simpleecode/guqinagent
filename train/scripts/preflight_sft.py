#!/usr/bin/env python3
"""Check Qwen3.5 chat-template compatibility and length before spending GPUs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * ratio))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cutoff-len", type=int, default=18432)
    parser.add_argument("--max-truncation-rate", type=float, default=0.05)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    lengths: list[int] = []
    errors: list[str] = []
    turns = Counter()
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if args.limit is not None and len(lengths) >= args.limit:
                break
            row = json.loads(line)
            try:
                tools = json.loads(row["tools"])
                rendered = processor.apply_chat_template(
                    row["messages"], tools=tools, tokenize=False, add_generation_prompt=False
                )
                count = len(processor.tokenizer(rendered, add_special_tokens=False)["input_ids"])
            except Exception as exc:
                errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
                continue
            lengths.append(count)
            turns[len(row["messages"])] += 1
    if not lengths:
        raise SystemExit("no renderable rows")
    over = sum(length > args.cutoff_len for length in lengths)
    report = {
        "schema_version": "guqin-sft-preflight-1.0",
        "rows": len(lengths),
        "template_errors": errors[:50],
        "cutoff_len": args.cutoff_len,
        "max_truncation_rate": args.max_truncation_rate,
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "over_cutoff": over,
        "truncation_rate": over / len(lengths),
        "message_count_distribution": dict(sorted(turns.items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("Qwen3.5 chat template rejected one or more rows")
    if report["truncation_rate"] > args.max_truncation_rate:
        raise SystemExit("truncation rate exceeds threshold; compact context before full SFT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
