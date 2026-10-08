"""Run GEPA reflective prompt evolution on the Guqin optimization split.

    python -m baseline.gepa_guqin.run_optimize \
        --data-dir baseline/gepa_guqin/data/gepa_split_v1 \
        --run-dir baseline/gepa_guqin/runs/opt_v1 \
        --max-metric-calls 2000

The evolved component is the single_stage system prompt; the seed is the
production ``public_system_for("single_stage")`` text.  Results land in
``--run-dir`` (GEPA checkpoints + ``best_candidate.json`` written here).
Stop gracefully at any time by creating ``<run-dir>/gepa.stop``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gepa import optimize  # noqa: E402

from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    public_system_for,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


class JsonlLogger:
    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="output directory of baseline.gepa_guqin.data")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-metric-calls", type=int, default=2000)
    parser.add_argument("--reflection-minibatch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--sample-temperature", type=float, default=0.25)
    parser.add_argument("--min-interval", type=float, default=0.5,
                        help="minimum seconds between GLM API calls")
    parser.add_argument("--tolerance-cents", type=float, default=50.0)
    parser.add_argument("--weights", type=str, default=None,
                        help='JSON dict over pitch/tone_type/fingering/ornament/playability')
    parser.add_argument("--model", default=None, help="override GLM_MODEL for task and reflection")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    from baseline.gepa_guqin.adapter import GuqinAgentAdapter
    from baseline.gepa_guqin.glm_client import (
        GlmAnthropicClient, GlmBackend, GlmReflectionLM,
    )

    train_examples = load_jsonl(args.data_dir / "examples_train.jsonl")
    val_examples = load_jsonl(args.data_dir / "examples_val.jsonl")
    if not train_examples or not val_examples:
        raise SystemExit("empty optimization split; run baseline.gepa_guqin.data first")
    historical = {
        (str(row["score_key"]), str(row["phrase_id"])): row
        for row in load_jsonl(args.data_dir / "historical.jsonl")
    }
    weights = json.loads(args.weights) if args.weights else None

    args.run_dir.mkdir(parents=True, exist_ok=True)
    logger = None if args.quiet else JsonlLogger(args.run_dir / "adapter_log.jsonl")
    client = GlmAnthropicClient(model=args.model, min_interval=args.min_interval)
    adapter = GuqinAgentAdapter(
        train_examples + val_examples, historical, GlmBackend(client),
        max_rounds=args.max_rounds, attempts=args.attempts,
        sample_temperature=args.sample_temperature,
        tolerance_cents=args.tolerance_cents, weights=weights, log=logger,
    )
    seed_candidate = {"single_stage": public_system_for("single_stage", basic=False)}

    result = optimize(
        adapter=adapter,
        seed_candidate=seed_candidate,
        trainset=train_examples,
        valset=val_examples,
        reflection_lm=GlmReflectionLM(client),
        reflection_minibatch_size=args.reflection_minibatch_size,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.run_dir),
        cache_evaluation=True,
        seed=args.seed,
        perfect_score=1.0,
    )

    best = dict(result.candidates[result.best_idx])
    payload = {
        "best_candidate": best,
        "best_val_score": result.best_score,
        "num_candidates": result.num_candidates,
        "best_idx": result.best_idx,
        "val_aggregate_scores": result.val_aggregate_scores,
    }
    (args.run_dir / "best_candidate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats = {
        "api_usage": client.snapshot(),
        "leakage_audit": adapter.leakage_counts,
        "total_evals": result.total_evals,
        "model": client.model,
    }
    # Counters are per-process; a resumed run must accumulate across its
    # predecessor so api_usage stays a run-lifetime total.
    stats_path = args.run_dir / "api_stats.json"
    if stats_path.exists():
        try:
            previous = json.loads(stats_path.read_text(encoding="utf-8"))
            for key, value in previous.get("api_usage", {}).items():
                stats["api_usage"][key] = stats["api_usage"].get(key, 0) + value
            for key, value in previous.get("leakage_audit", {}).items():
                stats["leakage_audit"][key] = stats["leakage_audit"].get(key, 0) + value
            stats["total_evals"] = max(stats["total_evals"],
                                       previous.get("total_evals") or 0)
        except (json.JSONDecodeError, OSError):
            pass
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "best_candidate"}, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"best candidate written to {args.run_dir / 'best_candidate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
