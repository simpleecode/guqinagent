"""Evaluate a conservative prompt addendum on fixed train examples.

This is a paired follow-up pilot, not a sealed-test evaluation. It keeps the
production prompt intact, appends a small policy, and records actions plus
metric feedback so failures can be diagnosed rather than inferred.
"""
from __future__ import annotations

import json
import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "guqinagent"
DATA = ROOT / "_tmp_zip_inspect/pitch_eligible_two_or_half/inferred_trajectories_train.jsonl"
SAMPLE_IDS = {"SMEq1nGR-p0001", "SBFpPuMq-p0001", "SqMgYYGH-p0001"}
sys.path.insert(0, str(REPO))

from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for
from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
from baseline.gepa_guqin.handbook import compose_system
from baseline.gepa_guqin.metric import phrase_report, reference_self_audit
from baseline.gepa_guqin.rollout import ScoreRunState, run_single_stage_phrase


def select_rows() -> list[dict]:
    found = {}
    with DATA.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("trajectory_id") in SAMPLE_IDS:
                found[row["trajectory_id"]] = row
    missing = SAMPLE_IDS - set(found)
    if missing:
        raise SystemExit(f"Missing samples: {sorted(missing)}")
    return [found[sample_id] for sample_id in sorted(SAMPLE_IDS)]


def prepare(row: dict) -> dict:
    input_data = deepcopy(row["input"])
    input_data.setdefault("phrase_handoff", {}).pop("previous_phrase", None)
    working = {
        "trajectory_id": row["trajectory_id"], "score_key": row["score_key"],
        "phrase_id": row["phrase_id"], "input": input_data,
        "baseline_plan": deepcopy(row.get("baseline_plan") or {}),
    }
    ScoreRunState().prepare(working)
    example = {
        "sample_id": row["trajectory_id"], "score_key": row["score_key"],
        "phrase_id": row["phrase_id"], "runtime_item": working,
        "reference": {"actions": materialize_actions(row["reference_plan"]["actions"])},
    }
    example["audit"] = reference_self_audit(example, {})
    return example


def compact_trace(trace: list[dict]) -> list[dict]:
    compact = []
    for entry in trace:
        calls = entry.get("tool_calls") or []
        compact.append({
            "round": entry.get("round"),
            "tool_calls": [call.get("name") if isinstance(call, dict) else str(call) for call in calls],
            "truncated": bool(entry.get("truncated", False)),
        })
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", type=Path, default=ROOT / "single_stage_conservative_v3_addendum.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "glm_prompt_v3_train_result.json")
    parser.add_argument("--label", default="production_seed_plus_conservative_v3_plus_handbook")
    args = parser.parse_args()
    examples = [prepare(row) for row in select_rows()]
    client = GlmAnthropicClient(min_interval=0.5, max_retries=1)
    backend = GlmBackend(client)
    production = public_system_for("single_stage", basic=False)
    addendum = args.addendum.read_text(encoding="utf-8").strip()
    prompt = production.rstrip() + "\n\n" + addendum
    result = {
        "kind": "train_followup_pilot", "model": client.model,
        "prompt": args.label,
        "prompt_policy": "production prompt unchanged; conservative addendum appended",
        "parameters": {"max_rounds": 8, "attempts": 1, "temperature": 0.0},
        "sample_ids": [example["sample_id"] for example in examples], "rows": [],
    }
    for example in examples:
        start_calls = client.actor_calls
        try:
            session = run_single_stage_phrase(
                example, {}, compose_system(prompt), backend,
                max_rounds=8, attempts=1, sample_temperature=0.0,
            )
            if session.get("ok"):
                actions = session["plan"]["actions"]
                prediction = {
                    "score_key": example["score_key"], "phrase_id": example["phrase_id"],
                    "jianzi_rows": [[int(action["source_index"]), action.get("jianzi_text") or action.get("text") or ""]
                                    for action in actions],
                }
                report = phrase_report(example, prediction, {})
                output = {
                    "sample_id": example["sample_id"], "protocol_ok": True,
                    "score": float(report["score"]) if report["score"] is not None else 0.5,
                    "components": report["metrics"]["components"],
                    "rule_violations": report["metrics"]["rule_violations"],
                    "pitch_evaluable_events": report["metrics"]["pitch_evaluable_events"],
                    "feedback": report["feedback"], "prediction": prediction,
                    "actor_calls": client.actor_calls - start_calls,
                    "rounds": len(session.get("trace") or []),
                    "trace_summary": compact_trace(session.get("trace") or []), "error": None,
                }
            else:
                output = {
                    "sample_id": example["sample_id"], "protocol_ok": False, "score": 0.0,
                    "components": {}, "rule_violations": None, "pitch_evaluable_events": None,
                    "feedback": "", "prediction": None,
                    "actor_calls": client.actor_calls - start_calls,
                    "rounds": len(session.get("trace") or []),
                    "trace_summary": compact_trace(session.get("trace") or []),
                    "error": session.get("error") or "no_valid_edit_plan",
                }
        except Exception as exc:
            output = {
                "sample_id": example["sample_id"], "protocol_ok": False, "score": 0.0,
                "components": {}, "rule_violations": None, "pitch_evaluable_events": None,
                "feedback": "", "prediction": None,
                "actor_calls": client.actor_calls - start_calls, "rounds": 0,
                "trace_summary": [], "error": f"{type(exc).__name__}: {exc}",
            }
        result["rows"].append(output)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: output[key] for key in ("sample_id", "protocol_ok", "score", "components", "rule_violations", "actor_calls", "rounds", "error")}, ensure_ascii=False), flush=True)
    result["mean_score"] = sum(row["score"] for row in result["rows"]) / len(result["rows"])
    result["usage"] = client.snapshot()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
