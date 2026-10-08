"""Sealed-test evaluation with a chosen single_stage system prompt.

Mirrors the serial ``--workflow single_stage`` path of
``train/scripts/eval_two_stage_score.py`` (model-own handoff, phrase order by
event_range, skip-to-next-score on protocol failure) but drives the GLM API
backend.  Output is an ``agent-eval-prediction-1.3`` JSONL ready for the
standard reporting pipeline:

    python -m baseline.gepa_guqin.run_final_eval \
        --prompt-json baseline/gepa_guqin/runs/opt_v1/best_candidate.json \
        --output baseline/gepa_guqin/runs/opt_v1/predictions_test.jsonl

    python -m evaluation.run_eval \
        --pred baseline/gepa_guqin/runs/opt_v1/predictions_test.jsonl \
        --reference train/eval_inputs_v2_text_protocol/renderfix_20261007/evaluation_pairs_test.jsonl \
        --experiment gepa_single_stage_v1 --model glm-5.3+gepa
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    public_system_for,
)

from baseline.gepa_guqin.compat import action_text  # noqa: E402
from baseline.gepa_guqin.rollout import ScoreRunState, run_single_stage_phrase  # noqa: E402

EVAL_SCHEMA_VERSION = "agent-eval-prediction-1.3"
DEFAULT_INPUT = (ROOT / "train/eval_inputs_v2_text_protocol/renderfix_20261007"
                 / "test_runtime.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-json", type=Path, default=None,
                        help='candidate JSON containing {"single_stage": text}; '
                             "omit to run the production seed prompt")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-key", action="append", default=[],
                        help="restrict to one score; repeatable")
    parser.add_argument("--stop-after-scores", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--sample-temperature", type=float, default=0.25)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-handbook", action="store_true",
                        help="ablation: omit the fixed fingering handbook")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
    from baseline.gepa_guqin.handbook import compose_system
    from baseline.gepa_guqin.rollout import run_single_stage_phrase

    if args.prompt_json is not None:
        payload = json.loads(args.prompt_json.read_text(encoding="utf-8"))
        system_text = (payload.get("best_candidate") or payload)["single_stage"]
        prompt_source = str(args.prompt_json)
    else:
        system_text = public_system_for("single_stage", basic=False)
        prompt_source = "production_seed"
    # Identical fixed handbook for the plain-seed ("Strong ReAct + Handbook")
    # baseline and every GEPA candidate — same knowledge, different strategy.
    if not args.no_handbook:
        system_text = compose_system(system_text)

    rows = [json.loads(line) for line in
            args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        runtime_item = row.get("runtime_item") or {}
        if row.get("reference") is not None or runtime_item.get("reference_plan") is not None:
            raise SystemExit(f"private reference found in public eval input: {row.get('sample_id')}")
    if args.score_key:
        rows = [row for row in rows if row.get("score_key") in set(args.score_key)]
    by_score: dict[str, list[dict]] = {}
    source_meta: dict[str, dict] = {}
    for row in rows:
        item = deepcopy(row["runtime_item"])
        by_score.setdefault(str(row["score_key"]), []).append(item)
        source_meta[str(row["sample_id"])] = row
    for phrases in by_score.values():
        phrases.sort(key=lambda item: int(item["input"]["event_range"]["start"]))

    completed: dict[str, dict] = {}
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            if prior.get("schema_version") != EVAL_SCHEMA_VERSION:
                raise SystemExit(
                    f"resume output uses incompatible schema {prior.get('schema_version')!r}; "
                    "use a new --output path"
                )
            if (prior.get("protocol_valid")
                    and isinstance((prior.get("single_stage") or {}).get("plan"), dict)):
                completed[str(prior["sample_id"])] = prior

    client = GlmAnthropicClient(model=args.model, min_interval=args.min_interval)
    backend = GlmBackend(client)

    def log(event: dict) -> None:
        if not args.quiet:
            print(json.dumps(event, ensure_ascii=False, default=str), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    processed_scores = 0
    with args.output.open(mode, encoding="utf-8", newline="\n") as output:
        for score_key in sorted(by_score):
            state = ScoreRunState()
            phrases = by_score[score_key]
            for item in phrases:
                sample_id = str(item["trajectory_id"])
                state.prepare(item)
                prior = completed.get(sample_id)
                if prior is not None:
                    final_plan = deepcopy(prior["single_stage"]["plan"])
                    item["baseline_plan"] = final_plan
                    if not state.commit(item, {"ok": True, "plan": final_plan}):
                        log({"event": "score_aborted", "score_key": score_key})
                        break
                    continue
                example = {
                    "sample_id": sample_id,
                    "score_key": score_key,
                    "phrase_id": item["phrase_id"],
                    "runtime_item": item,
                }
                session = run_single_stage_phrase(
                    example, state.historical, system_text, backend,
                    max_rounds=args.max_rounds, attempts=args.attempts,
                    sample_temperature=args.sample_temperature, log=log,
                )
                record = {
                    "schema_version": EVAL_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "split": item.get("split") or source_meta.get(sample_id, {}).get("split"),
                    "score_key": score_key,
                    "phrase_id": item["phrase_id"],
                    "input_sha256": source_meta.get(sample_id, {}).get("input_sha256"),
                    "system_prompt_source": prompt_source,
                    "single_stage": session,
                    "protocol_valid": bool(session.get("ok")),
                }
                if record["protocol_valid"]:
                    final_plan = session["plan"]
                    record["jianzi_rows"] = [
                        [int(action["source_index"]), action_text(action)]
                        for action in final_plan.get("actions", [])
                    ]
                    record["parse_error"] = None
                else:
                    final_plan = None
                    record["jianzi_rows"] = []
                    record["parse_error"] = "single_stage_failed"
                output.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                output.flush()
                log({"event": "final_phrase", "sample_id": sample_id,
                     "score_key": score_key, "protocol_valid": record["protocol_valid"]})
                if final_plan is None:
                    # Later phrases must not see an incomplete predecessor.
                    log({"event": "score_aborted", "score_key": score_key})
                    break
                item["reference_plan"] = final_plan
                state.historical[(score_key, item["phrase_id"])] = item
                state.previous = item
            processed_scores += 1
            if (args.stop_after_scores is not None
                    and processed_scores >= args.stop_after_scores):
                break
    print(json.dumps({"event": "done", "scores": processed_scores,
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
