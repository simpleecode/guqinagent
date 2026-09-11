from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backends import AnthropicCompatibleBackend
from .orchestrator import AbcToJianzipuOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ABC notation to a baseline jianzipu plan.")
    parser.add_argument("input", type=Path, help="UTF-8 ABC file")
    parser.add_argument("--out", type=Path, default=Path("runs/abc_to_jianzipu"))
    parser.add_argument("--job-id")
    parser.add_argument("--tuning-strategy", choices=("auto", "fixed"), default="auto")
    parser.add_argument("--tuning-name", default="正调")
    parser.add_argument("--performance-context", choices=("solo", "ensemble"), default="solo")
    parser.add_argument("--open-tone-demand", choices=("auto", "low", "high"), default="auto")
    parser.add_argument(
        "--open-midi", default="48,50,53,55,57,60,62",
        help="Seven comma-separated open-string MIDI pitches",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--phrase-bars", type=int, default=8)
    parser.add_argument("--tolerance-cents", type=float, default=35.0)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument(
        "--model-backend", choices=("none", "minimax"), default="none",
        help="Optional model-backed analyst and style critic",
    )
    parser.add_argument("--model", help="Override MINIMAX_MODEL from .env")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    open_midi = [float(value) for value in args.open_midi.split(",")]
    if len(open_midi) != 7:
        raise SystemExit("--open-midi requires exactly seven values")
    backend = (
        AnthropicCompatibleBackend(model=args.model)
        if args.model_backend == "minimax" else None
    )
    orchestrator = AbcToJianzipuOrchestrator(args.out, backend=backend)
    result = orchestrator.run(
        args.input.read_text(encoding="utf-8"),
        job_id=args.job_id,
        config={
            "tuning_strategy": args.tuning_strategy,
            "performance_context": args.performance_context,
            "open_tone_demand": args.open_tone_demand,
            "tuning_name": args.tuning_name,
            "open_midi": open_midi,
            "top_k": args.top_k,
            "phrase_bars": args.phrase_bars,
            "candidate_tolerance_cents": args.tolerance_cents,
            "max_repairs": args.max_repairs,
        },
    )
    print(json.dumps({
        "job_id": result["job_id"],
        "status": result["status"],
        "audit": result.get("audit", {}).get("summary", {}),
        "output": str((args.out / result["job_id"] / "output.md").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0 if result.get("audit", {}).get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
