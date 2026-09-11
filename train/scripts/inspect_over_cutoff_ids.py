#!/usr/bin/env python3
"""List SFT rows whose rendered Qwen3.5 prompt exceeds a cutoff."""
from __future__ import annotations

import argparse
import json

from transformers import AutoProcessor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--cutoff", type=int, default=8192)
    args = ap.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    rows = []
    with open(args.input, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rendered = processor.apply_chat_template(
                row["messages"], tools=json.loads(row["tools"]),
                tokenize=False, add_generation_prompt=False,
            )
            length = len(processor.tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if length > args.cutoff:
                rows.append({
                    "line": line_number,
                    "sample_id": row.get("sample_id"),
                    "agent_stage": row.get("agent_stage"),
                    "tokens": length,
                })
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
