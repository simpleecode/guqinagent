#!/usr/bin/env python3
"""Explicit paired confirmation/final-test stage, separate from prompt evolution."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from baseline.gepa_guqin.adapter import GuqinAgentAdapter
from baseline.gepa_guqin.teacher_exact_adapter import TeacherExactGuqinAdapter
from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
from baseline.gepa_guqin.handbook import compose_system
from baseline.scripts.prompt_data_sources import TRAIN, TEST50
from baseline.scripts.run_teacher_gepa_iteration import JsonlLogger, seed_prompt, aggregate_official
from baseline.scripts.render_v9_v27_prompt_iteration_report import render
from train.scripts.score_agent_predictions import score as official_score


def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="completed GEPA training run")
    parser.add_argument("--mode", choices=("confirmation", "test50"), default="confirmation")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    root = args.run_dir.resolve()
    training_manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    seed_text, _ = seed_prompt(training_manifest["parameters"]["seed_prompt"])
    best_path = root / "best_candidate.json"
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
    if not args.prepare_only and best is None:
        raise SystemExit("Training has not finished; no selected candidate available")
    if not args.prepare_only and best.get("safe_best_idx", 0) == 0:
        raise SystemExit("No candidate passed the three-metric gate; keep the seed and save confirmation/test budget")
    candidate = (best or {}).get("safe_best_candidate", {}).get("single_stage")
    texts = {"baseline": seed_text}
    if candidate:
        texts["candidate"] = candidate
    if args.mode == "test50" and not args.prepare_only:
        confirmation = root / "confirmation/paired_result.json"
        check = json.loads(confirmation.read_text(encoding="utf-8")) if confirmation.exists() else {}
        if check.get("status") != "complete" or not check.get("provisional_safety_pass"):
            raise SystemExit("Complete and pass independent confirmation before opening final test labels")
    if args.mode == "confirmation":
        source = root / "data/examples_confirmation.jsonl"
        examples = read(source)
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        plan = training_manifest["split"]
        if [e["sample_id"] for e in examples] != plan["confirmation_sample_ids"]:
            raise SystemExit("Confirmation IDs do not match the frozen experiment manifest")
    else:
        source = TEST50 / "test_runtime.jsonl"
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        expected_hashes = training_manifest["split"].get("test_hashes", {})
        if source_sha != expected_hashes.get("test50_public"):
            raise SystemExit("test50 public file changed since the pilot was registered")
        # Final test is an explicit separate command and never supplies feedback
        # to the training engine. prepare-only does not open the sealed labels.
        public = read(source)
        refs_path = TEST50 / "evaluation_pairs_test.jsonl"
        if hashlib.sha256(refs_path.read_bytes()).hexdigest() != expected_hashes.get("test50_reference"):
            raise SystemExit("test50 sealed reference file changed since pilot registration")
        references = {} if args.prepare_only else {x["sample_id"]: x for x in read(refs_path)}
        if not args.prepare_only and set(references) != {x["sample_id"] for x in public}:
            raise SystemExit("test50 public/reference IDs differ")
        examples = [{"sample_id": row["sample_id"], "score_key": row["score_key"],
                     "phrase_id": row["phrase_id"], "runtime_item": row["runtime_item"],
                     "reference": references.get(row["sample_id"], {}).get("reference", {"actions": []})}
                    for row in public]
    output_dir = root / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"mode": args.mode, "source_sha256": source_sha, "repeats": args.repeats,
              "prompt_hashes": {key: hashlib.sha256(compose_system(value).encode()).hexdigest()
                                for key, value in texts.items()},
              "sample_ids": [e["sample_id"] for e in examples],
              "parameters": training_manifest["parameters"]}
    if args.prepare_only:
        save(output_dir / "evaluation_plan.json", {**config, "status": "prepared",
                                                   "candidate_available": bool(candidate)})
        print(json.dumps({"status": "prepared", "mode": args.mode, "samples": len(examples),
                          "repeats": args.repeats, "candidate_available": bool(candidate)}, ensure_ascii=False))
        return 0
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=True)
    result_path = output_dir / "paired_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
        "configuration": config, "rows": [], "sample_ids": config["sample_ids"],
        "arms": [f"{arm}_r{rep}" for rep in range(1, args.repeats + 1) for arm in texts],
        "prompts": {f"{arm}_r{rep}": {"text": text} for rep in range(1, args.repeats + 1)
                    for arm, text in texts.items()},
        "phase": args.mode, "status": "running",
    }
    if result["configuration"] != config:
        raise SystemExit("Evaluation checkpoint configuration changed; use a separate run directory")
    series = training_manifest.get("series") or training_manifest.get("parameters", {}).get("series")
    origin = f"{series}0" if series else "本轮原始提示词"
    selected = (best or {}).get("safe_best_version") or "本轮选出的候选"
    result["generation"] = {
        "rationale": f"仅比较本轮的 {origin} 与 {selected}；不导入历史 V 系列提示词或结果。",
        "hypothesis": "训练集发现的修改能否在独立样本中减少生成结果与标注的差异",
        "risk": "小样本结果仍有波动，不能直接证明整体优于原版。",
        "promotion_gate": "以本页逐次官方评分和三指标复核为准：精确率提高，其他两项不下降。",
        "next_action": "本阶段结果不返回训练器修改提示词；通过独立确认后再决定是否进行最终测试。",
    }
    if result.get("status") == "complete":
        print("Evaluation already complete; no new API calls")
        return 0
    by_score = defaultdict(list)
    historical = {}
    comparison_rows = []
    for example in examples:
        item = deepcopy(example["runtime_item"])
        item["input"].setdefault("phrase_handoff", {}).pop("previous_phrase", None)
        row = {"trajectory_id": example["sample_id"], "score_key": example["score_key"],
               "phrase_id": example["phrase_id"], "input": item["input"],
               "baseline_plan": item.get("baseline_plan") or {},
               "reference_plan": {"actions": materialize_actions(example["reference"]["actions"])}}
        historical[(example["score_key"], example["phrase_id"])] = row
        comparison_rows.append(row)
        by_score[example["score_key"]].append(example)
    comparison_path = output_dir / "comparison_dataset.jsonl"
    with comparison_path.open("w", encoding="utf-8") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    references_path = output_dir / "references_for_scorer.jsonl"
    with references_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"sample_id": example["sample_id"], "reference": example["reference"]}, ensure_ascii=False) + "\n")
    completed = {(r["arm"], r["sample_id"]): r for r in result["rows"]}
    for repetition in range(1, args.repeats + 1):
        for role in (list(texts) if repetition % 2 else list(reversed(texts))):
            arm = f"{role}_r{repetition}"
            client = GlmAnthropicClient(min_interval=0.5)
            if client.model != training_manifest.get("model"):
                raise SystemExit("Current GLM model differs from the frozen training model")
            frozen_protocol = training_manifest.get("protocol_identity") or {}
            if (client.max_tokens != int(frozen_protocol.get("max_tokens", client.max_tokens)) or
                    client.thinking_mode != str(frozen_protocol.get("thinking", client.thinking_mode)).lower()):
                raise SystemExit("GLM token limit/thinking mode differs from the frozen training protocol")
            adapter = TeacherExactGuqinAdapter(examples, historical, GlmBackend(client),
                max_rounds=int(config["parameters"]["max_rounds"]),
                attempts=int(config["parameters"]["attempts"]),
                sample_temperature=float(config["parameters"]["temperature"]),
                log=JsonlLogger(output_dir / "rollout_events.jsonl"))
            # Resume reconstructs this candidate's own prior outputs, never gold.
            cache_key = hashlib.sha256(compose_system(texts[role]).encode()).hexdigest()[:16]
            sessions = adapter._session_cache.setdefault(cache_key, {})
            for key, row in completed.items():
                if key[0] == arm:
                    sessions[key[1]] = {"ok": row["protocol_ok"], "trace": [],
                        "plan": {"actions": [{"source_index": int(i), "jianzi_text": t}
                                               for i, t in (row.get("prediction") or {}).get("jianzi_rows", [])]},
                        "error": row.get("error")}
            for score_key in sorted(by_score):
                for example in sorted(by_score[score_key], key=lambda e: e["phrase_id"]):
                    if (arm, example["sample_id"]) in completed:
                        continue
                    output = adapter.evaluate([example], {"single_stage": texts[role]}).outputs[0]
                    record = {"sample_id": example["sample_id"], "arm": arm,
                              "protocol_ok": output["protocol_ok"], "teacher_score": output["teacher_score"],
                              "prediction": {"jianzi_rows": output["jianzi_rows"]},
                              "error": None if output["protocol_ok"] else "rollout_or_predecessor_failure"}
                    result["rows"].append(record)
                    completed[(arm, example["sample_id"])] = record
                    save(result_path, result)
                    render(result_path, comparison_path, output_dir / "comparison.html")
                    print(json.dumps({"arm": arm, "sample_id": example["sample_id"],
                                      "protocol_ok": output["protocol_ok"]}, ensure_ascii=False), flush=True)
    official = {}
    for arm in result["arms"]:
        predictions_path = output_dir / f"{arm}_predictions.jsonl"
        with predictions_path.open("w", encoding="utf-8") as handle:
            for row in result["rows"]:
                if row["arm"] == arm:
                    handle.write(json.dumps({"sample_id": row["sample_id"], "protocol_valid": row["protocol_ok"],
                                             "jianzi_rows": row["prediction"]["jianzi_rows"]}, ensure_ascii=False) + "\n")
        scores_path = output_dir / f"{arm}_official_score.json"
        official_score(SimpleNamespace(reference=references_path, predictions=predictions_path, output=scores_path))
        official[arm] = json.loads(scores_path.read_text(encoding="utf-8"))["aggregate"]
    result["status"] = "complete"
    result["official_scores"] = official
    metric_names = ("exact_cell_rate", "nonempty_recall", "protocol_valid_rate")
    # A small pilot confirmation is a provisional safety check, not statistical
    # evidence of general superiority. Report the repeat-level results as well.
    means = {role: {name: sum(float(official[f"{role}_r{rep}"][name] or 0)
                                     for rep in range(1, args.repeats + 1)) / args.repeats
                    for name in metric_names} for role in texts}
    result["repeat_mean_scores"] = means
    result["provisional_safety_pass"] = (
        all(means["candidate"][key] >= means["baseline"][key] for key in metric_names)
        and means["candidate"]["exact_cell_rate"] > means["baseline"]["exact_cell_rate"]
    )
    save(result_path, result)
    render(result_path, comparison_path, output_dir / "comparison.html")
    print(json.dumps({"status": "complete", "official_scores": official}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
