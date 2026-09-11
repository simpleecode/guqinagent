#!/usr/bin/env python3
"""Render the server-specific LLaMA-Factory configuration from a safe template."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "train" / "configs" / "guqin_agent_lora_4x3090.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-len", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-samples", type=int, help="Set only for a smoke run.")
    args = parser.parse_args()
    values = {
        "__MODEL_PATH__": args.model_path,
        "__DATASET_DIR__": args.dataset_dir,
        "__OUTPUT_DIR__": args.output_dir,
        "__MAX_SAMPLES__": "null" if args.max_samples is None else str(args.max_samples),
        "__CUTOFF_LEN__": str(args.cutoff_len),
        "__GRADIENT_ACCUMULATION_STEPS__": str(args.gradient_accumulation_steps),
    }
    rendered = TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    if "__" in rendered:
        raise SystemExit("unresolved configuration placeholder")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
