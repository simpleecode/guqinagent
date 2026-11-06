#!/usr/bin/env python3
"""Run the resumable V9/V27 prompt-iteration baseline on fixed train samples."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test"
PROMPT_DIR = REPO / "baseline" / "prompts"
DEFAULT_OUTPUT_ROOT = REPO / "baseline"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for  # noqa: E402
from baseline.gepa_guqin.compat import materialize_actions  # noqa: E402
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend  # noqa: E402
from baseline.gepa_guqin.handbook import compose_system  # noqa: E402
from baseline.gepa_guqin.rollout import ScoreRunState, run_single_stage_phrase  # noqa: E402
from train.scripts.score_agent_predictions import canonical  # noqa: E402
from baseline.scripts.prompt_data_sources import training_identity, require_current_split  # noqa: E402

from render_v9_v27_prompt_iteration_report import render  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_rows(sample_ids: list[str]) -> list[dict]:
    wanted = set(sample_ids)
    found = {}
    source = DATA_ROOT / "inferred_trajectories_train.jsonl"
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("trajectory_id") or "")
            if sample_id in wanted:
                found[sample_id] = row
    missing = [sample_id for sample_id in sample_ids if sample_id not in found]
    if missing:
        raise SystemExit(f"Dataset is missing requested samples: {missing}")
    return [found[sample_id] for sample_id in sample_ids]


def prepare(row: dict) -> dict:
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
    return {
        "sample_id": row["trajectory_id"],
        "score_key": row["score_key"],
        "phrase_id": row["phrase_id"],
        "runtime_item": working,
        # This reference is consumed only after rollout by the local scorer.
        "reference_actions": materialize_actions(row["reference_plan"]["actions"]),
    }


def prompts(candidate_id: str | None = None, candidate_prompt: Path | None = None) -> dict[str, dict[str, str]]:
    production = public_system_for("single_stage", basic=False)
    v9_path = PROMPT_DIR / "single_stage_minimal_tone_compound_v9_addendum.txt"
    v27_path = PROMPT_DIR / "single_stage_integrated_v15_prompt.txt"
    if not v9_path.exists() or not v27_path.exists():
        raise SystemExit("V9/V27 anchor prompt files are missing from guqinagent/baseline/prompts")
    v9_addendum = v9_path.read_text(encoding="utf-8").strip()
    v27_text = v27_path.read_text(encoding="utf-8").strip()
    catalog = {
        "v9": {
            "text": production.rstrip() + "\n\n" + v9_addendum,
            "source": f"production + {v9_path.relative_to(ROOT)}",
            "role": "正式基线候选",
        },
        "v27": {
            "text": v27_text,
            "source": f"{v27_path.relative_to(ROOT)}（V27 沿用 V15 整合提示词）",
            "role": "整合路线候选",
        },
    }
    if candidate_id:
        if candidate_prompt is None or not candidate_prompt.exists():
            raise SystemExit("--candidate-prompt must point to an existing UTF-8 prompt file")
        candidate_text = candidate_prompt.read_text(encoding="utf-8").strip()
        catalog[candidate_id] = {
            "text": production.rstrip() + "\n\n" + candidate_text,
            "source": f"production + {candidate_prompt.resolve().relative_to(ROOT)}",
            "role": "V9 最小修改候选",
        }
    return catalog


def score_prediction(reference_actions: list[dict], prediction: dict) -> dict:
    refs = {
        int(item["source_index"]): str(item.get("text") or item.get("jianzi_text") or "")
        for item in reference_actions if item.get("source_index") is not None
    }
    generated = {int(item[0]): str(item[1]) for item in prediction.get("jianzi_rows", [])}
    indices = set(refs) | set(generated)
    exact = sum(canonical(refs.get(index)) == canonical(generated.get(index)) for index in indices)
    nonempty_ref = {index for index, text in refs.items() if canonical(text)}
    nonempty_pred = {index for index, text in generated.items() if canonical(text)}
    return {
        "rows_compared": len(indices),
        "exact_cells": exact,
        "reference_nonempty": len(nonempty_ref),
        "predicted_nonempty": len(nonempty_pred),
        "nonempty_hits": len(nonempty_ref & nonempty_pred),
        "exact_cell_rate": exact / len(indices) if indices else None,
        "nonempty_recall": len(nonempty_ref & nonempty_pred) / len(nonempty_ref) if nonempty_ref else None,
    }


def summarize(rows: list[dict], arms: list[str]) -> dict:
    summary = {}
    for arm in arms:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        valid = [row for row in arm_rows if row.get("protocol_ok") and row.get("teacher_score")]
        totals = Counter()
        for row in valid:
            totals.update({key: int((row.get("teacher_score") or {}).get(key) or 0)
                           for key in ("rows_compared", "exact_cells", "reference_nonempty", "predicted_nonempty", "nonempty_hits")})
        summary[arm] = {
            "n": len(arm_rows),
            "protocol_successes": len(valid),
            "protocol_valid_rate": len(valid) / len(arm_rows) if arm_rows else None,
            "exact_cell_rate": totals["exact_cells"] / totals["rows_compared"] if totals["rows_compared"] else None,
            "nonempty_recall": totals["nonempty_hits"] / totals["reference_nonempty"] if totals["reference_nonempty"] else None,
            "counts": dict(totals),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("diagnostic", "check"), default="diagnostic")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="create the manifest and HTML without API calls")
    parser.add_argument("--retry-errors", action="store_true", help="retry rows that ended in transport/runtime exceptions")
    parser.add_argument("--parent", choices=("v9", "v27"), help="candidate comparison parent")
    parser.add_argument("--candidate-id", help="candidate arm name, for example v33")
    parser.add_argument("--candidate-prompt", type=Path, help="candidate addendum appended to production")
    parser.add_argument("--generation-manifest", type=Path, help="JSON with hypothesis, evidence and promotion gate")
    args = parser.parse_args()

    candidate_mode = bool(args.candidate_id or args.candidate_prompt or args.parent)
    if candidate_mode and not (args.candidate_id and args.candidate_prompt and args.parent):
        raise SystemExit("Candidate mode requires --parent, --candidate-id and --candidate-prompt together")

    identity = training_identity()
    require_current_split(identity)
    phase_file = DATA_ROOT / ("iteration_diagnostic_sample_ids.txt" if args.phase == "diagnostic" else "iteration_check_sample_ids.txt")
    sample_ids = load_ids(phase_file)
    run_name = (f"prompt_{args.candidate_id}_{args.phase}" if candidate_mode
                else f"v9_v27_{args.phase}_baseline")
    output = args.output or DEFAULT_OUTPUT_ROOT / "results" / f"{run_name}.json"
    default_report = f"{run_name}_comparison.html"
    html_output = args.html or DEFAULT_OUTPUT_ROOT / "reports" / default_report
    prompt_data = prompts(args.candidate_id, args.candidate_prompt)
    arms = [args.parent, args.candidate_id] if candidate_mode else ["v9", "v27"]
    prompt_manifest = {}
    for arm in arms:
        effective_text = compose_system(prompt_data[arm]["text"])
        prompt_manifest[arm] = {
            **prompt_data[arm],
            "instruction_sha256": digest_text(prompt_data[arm]["text"]),
            "instruction_length": len(prompt_data[arm]["text"]),
            # Resume identity includes the fixed handbook that the model
            # actually receives, so a handbook change cannot reuse old rows.
            "sha256": digest_text(effective_text),
            "effective_length": len(effective_text),
        }
    configuration = {
        "data_identity": identity,
        "schema_version": "prompt-iteration-run-1.0",
        "phase": args.phase,
        "sample_ids": sample_ids,
        "arms": arms,
        "prompt_hashes": {arm: prompt_manifest[arm]["sha256"] for arm in arms},
        "scorer": "guqinagent/train/scripts/score_agent_predictions.py canonical exact-cell v1",
        "parameters": {"max_rounds": 8, "attempts": 1, "temperature": 0.0},
    }

    result = None
    if output.exists():
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("data_identity") != identity:
            raise SystemExit("Existing checkpoint uses a different dataset; choose a new --output path")
        existing_configuration = {key: result.get(key) for key in ("schema_version", "phase", "sample_ids", "arms")}
        expected_configuration = {key: configuration[key] for key in existing_configuration}
        existing_hashes = {arm: (result.get("prompts", {}).get(arm) or {}).get("sha256") for arm in arms}
        if existing_configuration != expected_configuration or existing_hashes != configuration["prompt_hashes"]:
            if result.get("rows"):
                raise SystemExit("Existing checkpoint belongs to a different run; choose a new --output path")
            result = None
    if result is None:
        generation = {
            "parent": None,
            "hypothesis": "双基线测量，不修改提示词",
            "rationale": "在迭代集同一批样本上重新建立 V9 与 V27 的师兄三指标基线，并逐格定位两条路线的共同错误与互补优势。",
            "next_action": "基线完成后只选择一个高频、可归因错误族，分别生成最小修改候选。",
        }
        if candidate_mode:
            manifest = {}
            if args.generation_manifest:
                manifest = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
            generation = {
                "parent": args.parent,
                "candidate": args.candidate_id,
                "hypothesis": manifest.get("hypothesis") or "最小修改候选",
                "rationale": manifest.get("expected_improvement") or "在同题上验证单一修改假设。",
                "risk": manifest.get("regression_risk"),
                "promotion_gate": manifest.get("promotion_gate"),
                "evidence": manifest.get("evidence"),
                "next_action": "先在迭代诊断小批与父代同题比较；通过后才进入 iteration_check。",
            }
        result = {
            **configuration,
            "kind": "prompt_iteration_candidate" if candidate_mode else "v9_v27_prompt_iteration_baseline",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "model": os.environ.get("GLM_MODEL", "glm-5.3"),
            "prompts": prompt_manifest,
            "generation": generation,
            "rows": [],
            "summary": {},
            "status": "prepared",
        }
    else:
        # Paths may change as artifacts are organized; the exact effective
        # prompt hashes above, rather than the display path, define identity.
        result["prompts"] = prompt_manifest

    result["summary"] = summarize(result.get("rows", []), arms)
    result["updated_at"] = utc_now()
    atomic_json(output, result)
    render(output, DATA_ROOT / "inferred_trajectories_train.jsonl", html_output)
    if args.prepare_only:
        print(json.dumps({"status": result.get("status"), "result": str(output), "html": str(html_output),
                          "completed": len(result.get("rows", [])), "expected": len(sample_ids) * len(arms)}, ensure_ascii=False))
        return 0

    examples = [prepare(row) for row in select_rows(sample_ids)]
    client = GlmAnthropicClient(min_interval=0.5, max_retries=1)
    backend = GlmBackend(client)
    completed = {}
    for row in result.get("rows", []):
        completed[(str(row.get("sample_id")), str(row.get("arm")), str(row.get("prompt_sha256")))] = row

    for sample_index, example in enumerate(examples):
        order = arms if sample_index % 2 == 0 else list(reversed(arms))
        for arm in order:
            key = (example["sample_id"], arm, prompt_manifest[arm]["sha256"])
            prior = completed.get(key)
            if prior and not (args.retry_errors and prior.get("error_kind") == "exception"):
                print(json.dumps({"status": "resume_skip", "sample_id": key[0], "arm": arm}, ensure_ascii=False), flush=True)
                continue
            if prior:
                result["rows"].remove(prior)
            start_calls = client.actor_calls
            try:
                session = run_single_stage_phrase(
                    example, {}, compose_system(prompt_data[arm]["text"]), backend,
                    max_rounds=8, attempts=1, sample_temperature=0.0,
                )
                if session.get("ok"):
                    actions = session["plan"]["actions"]
                    prediction = {
                        "score_key": example["score_key"],
                        "phrase_id": example["phrase_id"],
                        "jianzi_rows": [[int(action["source_index"]), action.get("jianzi_text") or action.get("text") or ""] for action in actions],
                    }
                    teacher_score = score_prediction(example["reference_actions"], prediction)
                    row = {
                        "sample_id": example["sample_id"], "arm": arm,
                        "prompt_sha256": prompt_manifest[arm]["sha256"], "protocol_ok": True,
                        "score": teacher_score["exact_cell_rate"], "teacher_score": teacher_score,
                        "prediction": prediction, "actor_calls": client.actor_calls - start_calls,
                        "rounds": len(session.get("trace") or []), "error": None, "error_kind": None,
                        "completed_at": utc_now(),
                    }
                else:
                    row = {
                        "sample_id": example["sample_id"], "arm": arm,
                        "prompt_sha256": prompt_manifest[arm]["sha256"], "protocol_ok": False,
                        "score": None, "teacher_score": None, "prediction": None,
                        "actor_calls": client.actor_calls - start_calls, "rounds": len(session.get("trace") or []),
                        "error": session.get("error") or "no_valid_edit_plan", "error_kind": "protocol_failure",
                        "completed_at": utc_now(),
                    }
            except (KeyboardInterrupt, SystemExit):
                result["status"] = "interrupted"
                result["updated_at"] = utc_now()
                result["summary"] = summarize(result["rows"], arms)
                atomic_json(output, result)
                render(output, DATA_ROOT / "inferred_trajectories_train.jsonl", html_output)
                raise
            except Exception as exc:
                row = {
                    "sample_id": example["sample_id"], "arm": arm,
                    "prompt_sha256": prompt_manifest[arm]["sha256"], "protocol_ok": False,
                    "score": None, "teacher_score": None, "prediction": None,
                    "actor_calls": client.actor_calls - start_calls, "rounds": 0,
                    "error": f"{type(exc).__name__}: {exc}", "error_kind": "exception",
                    "completed_at": utc_now(),
                }
            result["rows"].append(row)
            completed[key] = row
            result["summary"] = summarize(result["rows"], arms)
            result["updated_at"] = utc_now()
            result["status"] = "running"
            result["usage"] = client.snapshot()
            atomic_json(output, result)
            render(output, DATA_ROOT / "inferred_trajectories_train.jsonl", html_output)
            print(json.dumps({key: row.get(key) for key in ("sample_id", "arm", "protocol_ok", "score", "actor_calls", "error")}, ensure_ascii=False), flush=True)

    result["status"] = "complete"
    result["summary"] = summarize(result["rows"], arms)
    result["updated_at"] = utc_now()
    result["usage"] = client.snapshot()
    atomic_json(output, result)
    render(output, DATA_ROOT / "inferred_trajectories_train.jsonl", html_output)
    print(json.dumps({"status": "complete", "summary": result["summary"], "result": str(output), "html": str(html_output)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
