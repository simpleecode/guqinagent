#!/usr/bin/env python3
"""Run a configurable, auditable GEPA loop with the senior exact-cell scorer.

The default is a deliberately small one-score smoke test.  It exercises the
real GEPA implementation (feedback minibatches, reflective mutation, Pareto
candidate sampling, minibatch acceptance and full Pareto-set evaluation).
Because train and Pareto phrases come from one score, this mode validates the
pipeline only; it is not evidence for deployment or generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test/inferred_trajectories_train.jsonl"
PROMPTS = REPO / "baseline" / "prompts"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gepa import optimize  # noqa: E402
from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for  # noqa: E402
from baseline.gepa_guqin.compat import materialize_actions  # noqa: E402
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend, GlmReflectionLM  # noqa: E402
from baseline.gepa_guqin.adapter import USER_PROMPT_CONTEXT_VERSION  # noqa: E402
from baseline.gepa_guqin.teacher_exact_adapter import TeacherExactGuqinAdapter  # noqa: E402
from baseline.scripts.prompt_data_sources import training_identity, require_current_split  # noqa: E402
from baseline.gepa_guqin.handbook import compose_system  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def export_prompt_texts(run_dir: Path) -> None:
    """Expose accepted candidates and rejected proposals as usable UTF-8 text."""
    path = run_dir / "candidates.json"
    if not path.exists():
        return
    manifest_path = run_dir / "experiment_manifest.json"
    series = json.loads(manifest_path.read_text(encoding="utf-8")).get("series") if manifest_path.exists() else None
    if series:
        from baseline.scripts.prompt_series import export_series
        rounds_path = run_dir / "run_log.json"
        export_series(run_dir, series, json.loads(path.read_text(encoding="utf-8")),
                      json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.exists() else [],
                      read_jsonl(run_dir / "teacher_gepa_events.jsonl") if (run_dir / "teacher_gepa_events.jsonl").exists() else [])
        return
    output = run_dir / "prompts"
    output.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        (output / f"candidate_{index:03d}.txt").write_text(candidate["single_stage"] + "\n", encoding="utf-8")
    events = run_dir / "teacher_gepa_events.jsonl"
    if events.exists():
        for event in read_jsonl(events):
            if event.get("event") == "gepa_proposal":
                text = event.get("new_instructions", {}).get("single_stage")
                if text:
                    (output / f"proposal_round_{int(event['iteration']):03d}.txt").write_text(text + "\n", encoding="utf-8")


def order_key(row: dict[str, Any]) -> tuple[int, str]:
    indices = [int(note["index"]) for note in (row.get("input") or {}).get("notes_without_jianzi") or []
               if note.get("index") is not None]
    return (min(indices) if indices else -1, str(row.get("phrase_id") or ""))


def seed_prompt(source: str) -> tuple[str, str]:
    production = public_system_for("single_stage", basic=False)
    if source == "production":
        return production, "production"
    if source == "v9":
        path = PROMPTS / "single_stage_minimal_tone_compound_v9_addendum.txt"
        return production.rstrip() + "\n\n" + path.read_text(encoding="utf-8").strip(), str(path)
    if source in {"v15", "v27"}:
        path = PROMPTS / "single_stage_integrated_v15_prompt.txt"
        return path.read_text(encoding="utf-8").strip(), str(path)
    path = Path(source)
    if not path.is_absolute():
        path = (WORKSPACE / path).resolve()
    if not path.exists():
        raise SystemExit(f"seed prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip(), str(path)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict[str, Any]) -> None:
        enriched = {"logged_at": now(), **event}
        serialized = json.dumps(enriched, ensure_ascii=False, default=str)
        key = os.environ.get("GLM_API_KEY", "")
        if key:
            serialized = serialized.replace(key, "[API_KEY_REDACTED]")
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized + "\n")


def prepare_one_score(score_key: str, run_dir: Path, *, feedback_phrases: int,
                      pareto_phrases: int, seed: int) -> tuple[list[dict], list[dict], dict, dict]:
    all_rows = read_jsonl(DATASET)
    rows = sorted([row for row in all_rows if str(row.get("score_key")) == score_key], key=order_key)
    if len(rows) < feedback_phrases + pareto_phrases:
        raise SystemExit(
            f"score {score_key} has {len(rows)} phrases, fewer than requested "
            f"{feedback_phrases}+{pareto_phrases}"
        )
    allowed = set((DATASET.parent / "iteration_optimize_score_keys.txt").read_text(encoding="utf-8").split())
    if score_key not in allowed:
        raise SystemExit(f"score {score_key} is not in iteration_optimize; refusing to use it")

    selected = rows[:feedback_phrases + pareto_phrases]
    if [row["phrase_id"] for row in selected] != [f"p{i:04d}" for i in range(1, len(selected) + 1)]:
        raise SystemExit("Single-score smoke test requires a contiguous prefix starting at p0001")
    feedback_rows = selected[:feedback_phrases]
    pareto_rows = selected[feedback_phrases:feedback_phrases + pareto_phrases]
    historical: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        normalized = deepcopy(row)
        normalized["reference_plan"] = {
            "actions": materialize_actions(row["reference_plan"]["actions"]),
        }
        historical[(str(row["score_key"]), str(row["phrase_id"]))] = normalized

    def example(row: dict[str, Any], split: str) -> dict[str, Any]:
        item_input = deepcopy(row["input"])
        item_input.setdefault("phrase_handoff", {}).pop("previous_phrase", None)
        return {
            "sample_id": str(row["trajectory_id"]), "split": split,
            "score_key": str(row["score_key"]), "phrase_id": str(row["phrase_id"]),
            "runtime_item": {"input": item_input, "baseline_plan": deepcopy(row.get("baseline_plan") or {})},
            "reference": {"actions": historical[(str(row["score_key"]), str(row["phrase_id"]))]["reference_plan"]["actions"]},
        }

    feedback = [example(row, "D_feedback") for row in feedback_rows]
    pareto = [example(row, "D_pareto") for row in pareto_rows]
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("examples_feedback.jsonl", feedback), ("examples_pareto.jsonl", pareto)):
        with (data_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (data_dir / "historical.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in historical.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    split = {
        "mode": "one_score_pipeline_smoke_only",
        "warning": "D_feedback and D_pareto are disjoint phrases but share one score family; never use this run as formal validation evidence.",
        "score_key": score_key, "source": str(DATASET), "seed": seed,
        "all_score_phrase_ids": [str(row["trajectory_id"]) for row in rows],
        "feedback_sample_ids": [item["sample_id"] for item in feedback],
        "pareto_sample_ids": [item["sample_id"] for item in pareto],
    }
    (data_dir / "split_manifest.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return feedback, pareto, historical, split


def aggregate_official(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {key: 0 for key in ("rows_compared", "exact_cells", "reference_nonempty",
                                "predicted_nonempty", "nonempty_hits")}
    valid = 0
    for output in outputs:
        valid += int(bool(output.get("protocol_ok")))
        for key in totals:
            totals[key] += int((output.get("teacher_score") or {}).get(key) or 0)
    n = len(outputs)
    return {
        "samples": n, "protocol_valid_rate": valid / n if n else None,
        "exact_cell_rate": totals["exact_cells"] / totals["rows_compared"] if totals["rows_compared"] else 0.0,
        "nonempty_recall": totals["nonempty_hits"] / totals["reference_nonempty"] if totals["reference_nonempty"] else 0.0,
        "counts": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-key", default="SWkES6xH")
    parser.add_argument("--run-dir", type=Path, default=REPO / "baseline" / "runs" / "teacher_gepa_one_score")
    parser.add_argument("--seed-prompt", default="v9", help="production, v9, v15/v27, or UTF-8 file path")
    parser.add_argument("--feedback-phrases", type=int, default=4)
    parser.add_argument("--pareto-phrases", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--max-metric-calls", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-interval", type=float, default=0.5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pilot-manifest", type=Path, help="fixed family-isolated multi-score pilot plan")
    parser.add_argument("--max-proposals", type=int, default=6, help="maximum reflective proposals (separate from metric budget)")
    parser.add_argument("--feedback-detail", choices=("exact", "features"), default="exact",
                        help="exact: train-only reference/output text differences; features: answer-free diagnostic ablation")
    parser.add_argument("--series", default="", help="fresh proposal series; requires production original seed")
    args = parser.parse_args()
    if args.series and (not re.fullmatch(r"[A-Z]", args.series) or args.seed_prompt != "production"):
        parser.error("新字母系列必须使用production原版种子，系列名只能是一个大写字母。")
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env", override=True)
    except ImportError:
        pass
    if args.minibatch_size < 1 or (not args.pilot_manifest and args.minibatch_size > args.feedback_phrases):
        parser.error("--minibatch-size must be between 1 and --feedback-phrases")
    if not args.pilot_manifest and args.max_metric_calls < args.pareto_phrases:
        parser.error("--max-metric-calls must cover at least the initial Pareto evaluation")
    if args.feedback_phrases < 1 or args.pareto_phrases < 1:
        parser.error("Both feedback and Pareto phrase counts must be positive")
    if args.max_proposals < 1:
        parser.error("--max-proposals must be positive")

    identity = training_identity()
    require_current_split(identity)
    protocol_identity = {
        "user_prompt_context_version": USER_PROMPT_CONTEXT_VERSION,
        "runtime_sha256": hashlib.sha256((REPO / "agents/ToolRuntime/runtime.py").read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256((REPO / "train/scripts/score_agent_predictions.py").read_bytes()).hexdigest(),
        "effective_seed_prompt_sha256": "",
        "pilot_manifest_sha256": hashlib.sha256(args.pilot_manifest.read_bytes()).hexdigest() if args.pilot_manifest else None,
        "model": args.model or os.environ.get("GLM_MODEL", "glm-5.3"),
        "max_tokens": os.environ.get("GLM_MAX_TOKENS", "8000"),
        "thinking": os.environ.get("GLM_THINKING", "disabled"),
        "feedback_detail": args.feedback_detail,
        "series": args.series,
    }
    args.run_dir = args.run_dir.resolve()
    prompt, prompt_source = seed_prompt(args.seed_prompt)
    protocol_identity["effective_seed_prompt_sha256"] = hashlib.sha256(compose_system(prompt).encode("utf-8")).hexdigest()
    old_path = args.run_dir / "experiment_manifest.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if old.get("data_identity") != identity:
            raise SystemExit("This checkpoint uses a different dataset. Use a new --run-dir for new data.")
        if old.get("protocol_identity") != protocol_identity and (
            (args.run_dir / "gepa_state.bin").exists() or
            (args.run_dir / "teacher_gepa_events.jsonl").exists()
        ):
            raise SystemExit("Pilot/scorer/handbook changed; use a new run directory")
        previous = old.get("parameters") or {}
        frozen = ("score_key", "seed_prompt", "feedback_phrases", "pareto_phrases",
                  "minibatch_size", "seed", "max_rounds", "attempts", "temperature")
        if any(previous.get(key) != getattr(args, key) for key in frozen) or (
            old.get("seed_prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        ):
            raise SystemExit("Run configuration changed. Choose a new --run-dir instead of mixing checkpoints.")
        # A completed run is a report operation unless its budget is explicitly raised.
        if (args.run_dir / "selection.json").exists() and (
            args.max_metric_calls <= int(previous.get("max_metric_calls") or 0)
            and args.max_proposals <= int(previous.get("max_proposals") or 0)
        ):
            from baseline.scripts.render_teacher_gepa_run import render
            export_prompt_texts(args.run_dir)
            render(args.run_dir, args.run_dir / "optimization_report.html")
            print("Run already complete. HTML refreshed; no API calls made.")
            return 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.pilot_manifest:
        from baseline.scripts.prepare_gepa_pilot import prepare_pilot
        feedback, pareto, historical, split = prepare_pilot(args.pilot_manifest, args.run_dir, identity)
    else:
        feedback, pareto, historical, split = prepare_one_score(
            args.score_key, args.run_dir, feedback_phrases=args.feedback_phrases,
            pareto_phrases=args.pareto_phrases, seed=args.seed,
        )
    if args.minibatch_size > len(feedback) or args.max_metric_calls < len(pareto):
        raise SystemExit("Minibatch/metric budget is incompatible with the fixed pilot split")
    manifest = {
        "schema_version": "teacher-gepa-run-1.0", "created_at": now(),
        "algorithm": "GEPA reflective mutation + instance-wise Pareto candidate sampling",
        "official_scorer": "guqinagent/train/scripts/score_agent_predictions.py semantics",
        "gepa_scalar_metric": "protocol-valid exact_cell_rate; protocol failure=0",
        "final_safety_gate": "protocol_valid_rate, exact_cell_rate and nonempty_recall must all be >= seed; at least one > seed",
        "parameters": vars(args) | {"run_dir": str(args.run_dir)},
        "seed_prompt_source": prompt_source,
        "data_identity": identity,
        "model": protocol_identity["model"],
        "protocol_identity": protocol_identity,
        "seed_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "split": split,
        "series": args.series,
        "history_policy": "this_run_only; no earlier version prompts, predictions or lessons loaded",
        "original_prompt_function": "ABC_J.scripts.generate_teacher_tool_trajectories.public_system_for(single_stage,basic=False)",
    }
    (args.run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if args.series:
        prompt_dir = args.run_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / f"{args.series}0.txt").write_text(prompt + "\n", encoding="utf-8")
        source_file = REPO / "ABC_J/scripts/generate_teacher_tool_trajectories.py"
        from baseline.gepa_guqin.handbook import HANDBOOK_PATH
        provenance = {
            "version": f"{args.series}0", "seed": "teacher_repository_original",
            "function": "public_system_for('single_stage', basic=False)",
            "source_file": str(source_file),
            "source_code_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "strategy_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "matches_current_teacher_original": prompt == public_system_for("single_stage", basic=False),
            "effective_system_sha256": protocol_identity["effective_seed_prompt_sha256"],
            "shared_handbook": str(HANDBOOK_PATH),
            "shared_handbook_sha256": hashlib.sha256(HANDBOOK_PATH.read_bytes()).hexdigest(),
            "historical_version_addenda": [], "historical_run_imports": [],
        }
        (args.run_dir / "seed_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from baseline.scripts.render_teacher_gepa_run import render
    report_path = args.run_dir / "optimization_report.html"
    render(args.run_dir, report_path)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "run_dir": str(args.run_dir),
                          "html": str(report_path), "split": split}, ensure_ascii=False, indent=2))
        return 0

    logger = JsonlLogger(args.run_dir / "teacher_gepa_events.jsonl")
    client = GlmAnthropicClient(model=args.model, min_interval=args.min_interval)
    adapter = TeacherExactGuqinAdapter(
        feedback + pareto, historical, GlmBackend(client),
        max_rounds=args.max_rounds, attempts=args.attempts,
        sample_temperature=args.temperature, log=logger,
        feedback_detail=args.feedback_detail,
    )
    result = None
    from baseline.scripts.teacher_gepa_audit import TeacherGepaAudit
    audit = TeacherGepaAudit(logger, adapter, args.run_dir, render)
    from gepa.utils.stop_condition import MaxCandidateProposalsStopper
    prior_lessons = []
    if logger.path.exists():
        for line in logger.path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event") == "evolution_lesson":
                prior_lessons.append(event["lesson"])
    adapter.evolution_memory = prior_lessons
    try:
        result = optimize(
            adapter=adapter,
            seed_candidate={"single_stage": prompt},
            trainset=feedback,
            valset=pareto,
            reflection_lm=GlmReflectionLM(client),
            reflection_minibatch_size=args.minibatch_size,
            max_metric_calls=args.max_metric_calls,
            run_dir=str(args.run_dir),
            cache_evaluation=True,
            seed=args.seed,
            perfect_score=1.0,
            candidate_selection_strategy="pareto",
            batch_sampler="epoch_shuffled",
            callbacks=[audit],
            stop_callbacks=[MaxCandidateProposalsStopper(args.max_proposals)],
            reflection_prompt_template=(
                "当前提示词：\n<curr_param>\n训练rollout、标注差异及历史成败证据：\n<side_info>\n"
                "请围绕最有证据的一个失败模式，提出最小必要修改。保留父代已有正确规则、工具协议和音乐能力。"
                "不要根据一个小批就规定一律优先按音/散音，也不要用广泛置空提高空格匹配。"
                "省略字段可能继承前态，不能把文字差异自动认定为音色或音高错误。"
                "反馈中的训练标注是监督信号：归纳其中的记谱格式、字段次序、局部继承和技法选择模式，"
                "让独立生成的文字在师兄的规范化规则下更接近标注。不要只追求音高大致相同而忽略字符串差异。"
                "吸收历史有效片段，避免重复已失败的改动；不要直接拼接完整历史提示词。"
                "禁止写入曲谱编号、样本ID、特定音序答案或完整标注示例。"
                "只返回修改后的完整工作提示词，用三个反引号包围；不要输出隐藏推理过程。"
            ),
        )
        candidates = [dict(item) for item in result.candidates]
        from baseline.scripts.prompt_series import candidate_versions
        rounds_path = args.run_dir / "run_log.json"
        round_records = json.loads(rounds_path.read_text(encoding="utf-8")) if rounds_path.exists() else []
        versions = candidate_versions(args.series, round_records) if args.series else {}
        official = []
        # Use the exact predictions that GEPA evaluated.  Do not run a fresh,
        # potentially different stochastic validation just to build totals.
        scored = {}
        with logger.path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("event") == "teacher_phrase_scored":
                    scored[(event.get("candidate_sha256"), event.get("sample_id"))] = event
        for index, candidate in enumerate(candidates):
            sha = hashlib.sha256(candidate["single_stage"].encode("utf-8")).hexdigest()
            saved = [scored.get((sha, example["sample_id"])) for example in pareto]
            if not all(saved):
                raise RuntimeError("Missing saved Pareto evaluations; refusing an unbudgeted fresh evaluation")
            metrics = aggregate_official([
                {"protocol_ok": item["ok"], "teacher_score": item["teacher_score"]}
                for item in saved
            ])
            official.append({
                "candidate_idx": index,
                "version": versions.get(index),
                "prompt_sha256": hashlib.sha256(candidate["single_stage"].encode("utf-8")).hexdigest(),
                "gepa_val_score": result.val_aggregate_scores[index],
                **metrics,
            })
        seed_metrics = official[0]
        safe_indices = [item["candidate_idx"] for item in official if
                        item["protocol_valid_rate"] >= seed_metrics["protocol_valid_rate"] and
                        item["exact_cell_rate"] >= seed_metrics["exact_cell_rate"] and
                        item["nonempty_recall"] >= seed_metrics["nonempty_recall"] and
                        (item["protocol_valid_rate"] > seed_metrics["protocol_valid_rate"] or
                         item["exact_cell_rate"] > seed_metrics["exact_cell_rate"] or
                         item["nonempty_recall"] > seed_metrics["nonempty_recall"])]
        safe_best = max(safe_indices, key=lambda idx: (
            official[idx]["exact_cell_rate"], official[idx]["nonempty_recall"],
            official[idx]["protocol_valid_rate"], -idx,
        )) if safe_indices else 0
        selection = {
            "status": "complete", "completed_at": now(),
            "gepa_best_idx": result.best_idx,
            "gepa_best_val_score": result.best_score,
            "safe_best_idx": safe_best,
            "safe_best_is_seed": safe_best == 0,
            "gepa_best_version": versions.get(result.best_idx),
            "safe_best_version": versions.get(safe_best),
            "safe_candidate_indices": safe_indices,
            "official_candidate_scores": official,
            "note": "pilot screening only; requires independent confirmation and sealed final test",
            "api_usage": client.snapshot(),
            "leakage_audit": adapter.leakage_counts,
            "total_evals": result.total_evals,
            "instance_frontier": {str(key): sorted(value) for key, value in result.per_val_instance_best_candidates.items()},
            "objective_specialists": {
                name: [item["candidate_idx"] for item in official if item[name] == max(r[name] for r in official)]
                for name in ("exact_cell_rate", "nonempty_recall", "protocol_valid_rate")
            },
        }
        (args.run_dir / "selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        best_payload = {
            "best_candidate": candidates[result.best_idx],
            "best_idx": result.best_idx, "best_val_score": result.best_score,
            "safe_best_candidate": candidates[safe_best], "safe_best_idx": safe_best,
            "val_aggregate_scores": result.val_aggregate_scores,
            "num_candidates": result.num_candidates,
            "best_version": versions.get(result.best_idx),
            "safe_best_version": versions.get(safe_best),
        }
        (args.run_dir / "best_candidate.json").write_text(
            json.dumps(best_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        export_prompt_texts(args.run_dir)
        print(json.dumps(selection, ensure_ascii=False, indent=2))
    finally:
        render(args.run_dir, report_path)
        print(f"HTML: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
