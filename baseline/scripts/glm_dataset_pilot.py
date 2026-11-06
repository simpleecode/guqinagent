"""Compare two frozen prompts on early train phrases with the real Guqin tools.

This is a small pilot, not a GEPA optimization run or a sealed-test result.
The source is read only; private references enter only the metric functions.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "guqinagent"
DATA = ROOT / "_tmp_zip_inspect/pitch_eligible_two_or_half/inferred_trajectories_train.jsonl"
OUT = ROOT / "glm_dataset_pilot_result.json"
sys.path.insert(0, str(REPO))

from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for
from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
from baseline.gepa_guqin.handbook import compose_system
from baseline.gepa_guqin.metric import phrase_report, reference_self_audit
from baseline.gepa_guqin.rollout import ScoreRunState, run_single_stage_phrase


def select_examples(limit: int = 3):
    candidates = []
    with DATA.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            notes = row["input"].get("notes_without_jianzi") or []
            if (str(row["trajectory_id"]).endswith("-p0001")
                    and 4 <= len(notes) <= 40
                    and (row.get("reference_plan") or {}).get("actions")):
                candidates.append(row)
    candidates.sort(key=lambda row: (len(row["input"]["notes_without_jianzi"]), str(row["score_key"])))
    return candidates[:limit]


def prepare(row):
    input_data = deepcopy(row["input"])
    input_data.setdefault("phrase_handoff", {}).pop("previous_phrase", None)
    working = {
        "trajectory_id": row["trajectory_id"],
        "score_key": row["score_key"],
        "phrase_id": row["phrase_id"],
        "input": input_data,
        "baseline_plan": deepcopy(row.get("baseline_plan") or {}),
    }
    ScoreRunState().prepare(working)
    example = {
        "sample_id": row["trajectory_id"],
        "score_key": row["score_key"],
        "phrase_id": row["phrase_id"],
        "runtime_item": working,
        "reference": {"actions": materialize_actions(row["reference_plan"]["actions"])},
    }
    example["audit"] = reference_self_audit(example, {})
    return example


def main():
    examples = [prepare(row) for row in select_examples()]
    if not examples:
        raise SystemExit("No suitable first phrases found")
    client = GlmAnthropicClient(min_interval=0.5, max_retries=1)
    backend = GlmBackend(client)
    prompts = {
        "production_seed_plus_handbook": public_system_for("single_stage", basic=False),
        "gpt_curated_v2_plus_handbook": (REPO / "baseline/gepa_guqin/prompts/single_stage_curated_v2.txt").read_text(encoding="utf-8").strip(),
    }
    result = {"kind": "train_pilot", "model": client.model,
              "parameters": {"max_rounds": 8, "attempts": 1, "temperature": 0.0},
              "sample_ids": [e["sample_id"] for e in examples], "rows": []}
    for example in examples:
        for prompt_name, prompt in prompts.items():
            start_calls = client.actor_calls
            try:
                session = run_single_stage_phrase(
                    example, {}, compose_system(prompt), backend,
                    max_rounds=8, attempts=1, sample_temperature=0.0,
                )
                if session.get("ok"):
                    plan = session["plan"]["actions"]
                    prediction = {"score_key": example["score_key"],
                                  "phrase_id": example["phrase_id"],
                                  "jianzi_rows": [[int(a["source_index"]), a.get("jianzi_text") or a.get("text") or ""] for a in plan]}
                    report = phrase_report(example, prediction, {})
                    score = float(report["score"]) if report["score"] is not None else 0.5
                    components = report["metrics"]["components"]
                    error = None
                else:
                    score, components = 0.0, {}
                    error = session.get("error") or "no_valid_edit_plan"
                output = {"sample_id": example["sample_id"], "prompt": prompt_name,
                          "protocol_ok": bool(session.get("ok")), "score": score,
                          "components": components, "actor_calls": client.actor_calls - start_calls,
                          "rounds": len(session.get("trace") or []), "error": error}
            except Exception as exc:
                output = {"sample_id": example["sample_id"], "prompt": prompt_name,
                          "protocol_ok": False, "score": 0.0, "components": {},
                          "actor_calls": client.actor_calls - start_calls,
                          "error": f"{type(exc).__name__}: {exc}"}
            result["rows"].append(output)
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(output, ensure_ascii=False), flush=True)
    result["usage"] = client.snapshot()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
