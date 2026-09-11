"""Print the longest Qwen3.5-tokenized SFT rows with aligned metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--ids-output", type=Path)
    ap.add_argument("--enriched-output", type=Path)
    ap.add_argument("--selected-output", type=Path)
    args = ap.parse_args()

    metadata = [json.loads(line) for line in args.metadata.open(encoding="utf-8")]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rows = []
    enriched = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            sample = json.loads(line)
            tools = json.loads(sample["tools"]) if isinstance(sample.get("tools"), str) else sample.get("tools")
            tokens = len(tokenizer.apply_chat_template(
                sample["messages"], tools=tools, tokenize=True, add_generation_prompt=False
            ))
            item = metadata[line_number - 1]
            enriched_item = dict(sample)
            enriched_item["sample_id"] = item.get("source_sample_id")
            enriched_item["agent_stage"] = item.get("agent_stage")
            enriched_item["provenance"] = {
                "source_trajectory_id": item.get("source_trajectory_id")
            }
            enriched_item["tools"] = tools or []
            enriched_item["token_count"] = tokens
            enriched.append(enriched_item)
            rows.append({
                "tokens": tokens,
                "line": line_number,
                "sample_id": item.get("source_sample_id"),
                "trajectory_id": item.get("source_trajectory_id"),
                "stage": item.get("agent_stage"),
                "message_count": len(sample.get("messages") or []),
            })
    rows.sort(key=lambda row: (-row["tokens"], row["line"]))
    selected = rows[: max(1, args.top)]
    for row in selected:
        print(json.dumps(row, ensure_ascii=False))
    if args.ids_output:
        args.ids_output.write_text(
            "".join(f"{row['sample_id']}\n" for row in selected), encoding="utf-8"
        )
    if args.enriched_output:
        args.enriched_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enriched),
            encoding="utf-8",
        )
    if args.selected_output:
        selected_rows = [enriched[row["line"] - 1] for row in selected]
        args.selected_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_rows),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
