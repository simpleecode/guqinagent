"""Paired prompt pilot on deterministic, previously unused p0001 examples.

Candidate selection uses the repository's sealed exact-cell scoring protocol:
exact_cell_rate, nonempty_recall and protocol_valid_rate.  Train-mode is for
candidate selection. Test-mode is a frozen one-shot pilot, not the formal full
sequential test benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from copy import deepcopy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PROMPT_DIR = ROOT / "prompts"
RESULT_DIR = ROOT / "results"
REPO = ROOT / "guqinagent"
DATA_ROOT = ROOT / "_tmp_zip_inspect/pitch_eligible_two_or_half"
PRIOR_IDS = {
    "SMEq1nGR-p0001", "SBFpPuMq-p0001", "SqMgYYGH-p0001",
    "SWkES6xH-p0001", "S45fs5uq-p0001", "ShnpF8Qg-p0001",
    "SNt85j0u-p0001", "Sn1Yvu4u-p0001", "SqY0ZGAm-p0001",
    "SK7bMnBG-p0001", "SwCGgMsJ-p0001", "SkgRafvU-p0001",
    "SZZkpAAL-p0001", "SB4hDksv-p0001", "SQYzHeTq-p0001",
    "Sh8BUAQh-p0001", "S8Tq46CE-p0001", "SKIU1Qgw-p0001",
    "SrHw2CVs-p0001", "S3rVwfH0-p0001", "S4nhaTGN-p0001",
    "S3JNAahW-p0001", "SRa07eQk-p0001",
    "SkAD4fmu-p0001", "SQAf6j3a-p0001", "SApzZmDj-p0001",
    "S6Heech0-p0001", "S1zt293d-p0001", "SjLqLX8p-p0001",
    "SEhKToLa-p0001", "S8Y5kmNg-p0001", "SBAwX43Q-p0001",
    "SmgJYQmw-p0001", "SywB9Jym-p0001", "SaDRc7nJ-p0001",
    "SDcE3Geq-p0001", "STcGty0f-p0001",
    "SFd9d5qs-p0001", "STYDv7xh-p0001", "SaN3X9UQ-p0001",
    "SzV4KbB5-p0001", "Sy0h59DY-p0001", "SzQNSg3J-p0001",
    "Sw4S16SU-p0001", "STTPh4nr-p0001",
    "SNVqKaVy-p0001",
    "SSGNQthN-p0001", "SXv5nCde-p0001", "SknHwPhW-p0001",
    "SfXYSY5p-p0001", "SBEG4da1-p0001", "STe3AARv-p0001",
    "SZh5gUkP-p0001",
    "SZkHNtAN-p0001", "SxC3qbtd-p0001", "STyg7J1U-p0001",
    "SpiUELEp-p0001", "SxhX3fgD-p0001", "S2T7rDyJ-p0001",
    "SqJmJZhj-p0001", "SN7RK3c4-p0001",
}
sys.path.insert(0, str(REPO))

from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for
from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
from baseline.gepa_guqin.handbook import compose_system
from baseline.gepa_guqin.rollout import ScoreRunState, run_single_stage_phrase
from teacher_standard_score import score_prediction


def write_result_html(result: dict, output: Path) -> None:
    """Human-readable, continuously updated score page for a paired run."""
    def rate(value):
        return "—" if value is None else f"{float(value):.2%}"
    summary_rows = []
    for arm in result.get("arms") or []:
        item = (result.get("summary") or {}).get(arm) or {}
        summary_rows.append(
            f"<tr><td>{html.escape(arm)}</td><td>{item.get('n', 0)}</td>"
            f"<td>{rate(item.get('exact_cell_rate'))}</td><td>{rate(item.get('nonempty_recall'))}</td>"
            f"<td>{rate(item.get('protocol_valid_rate'))}</td><td>{rate((item.get('selection') or {}).get('retain_as_evidence'))}</td></tr>"
        )
    detail_rows = []
    for row in result.get("rows") or []:
        score = row.get("teacher_score") or {}
        detail_rows.append(
            f"<tr><td>{html.escape(str(row.get('sample_id', '')))}</td><td>{html.escape(str(row.get('arm', '')))}</td>"
            f"<td>{'成功' if row.get('protocol_ok') else '失败'}</td><td>{rate(score.get('exact_cell_rate'))}</td>"
            f"<td>{rate(score.get('nonempty_recall'))}</td><td>{html.escape(str(row.get('error') or '—'))}</td></tr>"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><meta charset='utf-8'><title>提示词快速筛选成绩</title>
<style>body{{max-width:1200px;margin:28px auto;font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif;background:#f6f8fb;color:#17212b}}table{{border-collapse:collapse;width:100%;background:#fff;margin:12px 0}}th,td{{border:1px solid #dce3eb;padding:7px;text-align:left}}th{{background:#edf3f8}}details{{margin:18px 0}}summary{{cursor:pointer;font-weight:600}}</style>
<h1>提示词快速筛选成绩</h1><p>师兄标准：逐格精确率、非空召回、协议合规率。此页面在每条样本完成后更新。</p>
<table><thead><tr><th>提示词版本</th><th>已完成样本</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th><th>Pareto 保留</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
<details><summary>逐样本明细</summary><table><thead><tr><th>样本</th><th>版本</th><th>协议</th><th>逐格精确率</th><th>非空召回</th><th>错误</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table></details>""", encoding="utf-8")


def stable_rank(sample_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def select_rows(dataset: str, limit: int, seed: str,
                extra_excluded: set[str] | None = None) -> list[dict]:
    path = DATA_ROOT / f"inferred_trajectories_{dataset}.jsonl"
    candidates = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            sample_id = str(row.get("trajectory_id"))
            notes = row["input"].get("notes_without_jianzi") or []
            attacks = sum(bool(note.get("event_index") is not None and
                               not str(note.get("abc", "")).startswith(("-", "z", "|")))
                          for note in notes)
            has_chord = any("[" in str(note.get("abc", "")) for note in notes)
            if (sample_id.endswith("-p0001") and sample_id not in PRIOR_IDS
                    and sample_id not in (extra_excluded or set())
                    and 6 <= attacks <= 20 and (row.get("reference_plan") or {}).get("actions")):
                bucket = ("chord" if has_chord else "single", attacks // 5)
                candidates.append((bucket, stable_rank(sample_id, seed), row))
    # Round-robin buckets prevents the pilot from containing only one phrase shape.
    by_bucket: dict[tuple, list[tuple]] = {}
    for bucket, rank, row in candidates:
        by_bucket.setdefault(bucket, []).append((rank, row))
    for rows in by_bucket.values():
        rows.sort(key=lambda item: item[0])
    selected = []
    buckets = sorted(by_bucket, key=str)
    while len(selected) < limit and any(by_bucket.values()):
        for bucket in buckets:
            if by_bucket[bucket] and len(selected) < limit:
                selected.append(by_bucket[bucket].pop(0)[1])
    return selected


def select_explicit_rows(dataset: str, sample_ids: list[str]) -> list[dict]:
    wanted = set(sample_ids)
    found = {}
    path = DATA_ROOT / f"inferred_trajectories_{dataset}.jsonl"
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            sample_id = str(row.get("trajectory_id"))
            if sample_id in wanted:
                found[sample_id] = row
    missing = [sample_id for sample_id in sample_ids if sample_id not in found]
    if missing:
        raise SystemExit(f"Unknown sample ids: {missing}")
    return [found[sample_id] for sample_id in sample_ids]


def prepare(row: dict) -> dict:
    input_data = deepcopy(row["input"])
    input_data.setdefault("phrase_handoff", {}).pop("previous_phrase", None)
    working = {"trajectory_id": row["trajectory_id"], "score_key": row["score_key"],
               "phrase_id": row["phrase_id"], "input": input_data,
               "baseline_plan": deepcopy(row.get("baseline_plan") or {})}
    ScoreRunState().prepare(working)
    example = {"sample_id": row["trajectory_id"], "score_key": row["score_key"],
               "phrase_id": row["phrase_id"], "runtime_item": working,
               "reference": {"actions": materialize_actions(row["reference_plan"]["actions"])}}
    return example


def prompt_arms(names: list[str]) -> dict[str, str]:
    production = public_system_for("single_stage", basic=False)
    files = {
        "v3": PROMPT_DIR / "single_stage_conservative_v3_addendum.txt",
        "v4": PROMPT_DIR / "single_stage_guardrails_v4_addendum.txt",
        "v5": PROMPT_DIR / "single_stage_parseable_v5_addendum.txt",
        "v6": PROMPT_DIR / "single_stage_parseable_tone_v6_addendum.txt",
        "v7": PROMPT_DIR / "single_stage_parseable_compound_v7_addendum.txt",
        "v8": PROMPT_DIR / "single_stage_tone_state_compound_v8_addendum.txt",
        "v9": PROMPT_DIR / "single_stage_minimal_tone_compound_v9_addendum.txt",
        "v10": PROMPT_DIR / "single_stage_verified_multistring_v10_addendum.txt",
        "v11": PROMPT_DIR / "single_stage_phrase_route_v11_addendum.txt",
        "v12": PROMPT_DIR / "single_stage_motif_consistency_v12_addendum.txt",
        "v13": PROMPT_DIR / "single_stage_motif_tiebreak_v13_addendum.txt",
        "v14": PROMPT_DIR / "single_stage_canonical_order_v14_addendum.txt",
        "v16": PROMPT_DIR / "single_stage_minimal_format_v16_addendum.txt",
        "v17": PROMPT_DIR / "single_stage_structured_v17_addendum.txt",
        "v18": PROMPT_DIR / "single_stage_structured_v18_addendum.txt",
        "v19": PROMPT_DIR / "single_stage_structured_v19_addendum.txt",
        "v20": PROMPT_DIR / "single_stage_candidate_v20_addendum.txt",
        "v21": PROMPT_DIR / "single_stage_candidate_v21_addendum.txt",
        "v22": PROMPT_DIR / "single_stage_candidate_v22_addendum.txt",
        "v23": PROMPT_DIR / "single_stage_local_multistring_repair_v23_addendum.txt",
        "v25": PROMPT_DIR / "single_stage_pareto_handoff_v25_addendum.txt",
        "v26": PROMPT_DIR / "single_stage_canonical_guard_v26_addendum.txt",
    }
    full_files = {
        "v15": PROMPT_DIR / "single_stage_integrated_v15_prompt.txt",
        "v27": PROMPT_DIR / "single_stage_integrated_v15_prompt.txt",
    }
    derived_files = {
        "v28": (PROMPT_DIR / "single_stage_integrated_v15_prompt.txt",
                PROMPT_DIR / "single_stage_repeated_open_v28_addendum.txt"),
        "v29": (PROMPT_DIR / "single_stage_integrated_v15_prompt.txt",
                PROMPT_DIR / "single_stage_repeated_tone_v29_addendum.txt"),
        "v30": (PROMPT_DIR / "single_stage_integrated_v15_prompt.txt",
                PROMPT_DIR / "single_stage_tone_first_v30_addendum.txt"),
    }
    transformed_files = {
        "v31": (PROMPT_DIR / "single_stage_integrated_v15_prompt.txt",
                "散音后紧接同弦按音时优先给后一个按音换弦；", ""),
        "v32": (PROMPT_DIR / "single_stage_integrated_v15_prompt.txt",
                "每次工具调用前简要说明理由。", ""),
    }
    arms = {}
    for name in names:
        if name == "production":
            arms[name] = production
        elif name in derived_files:
            parent, addendum = derived_files[name]
            arms[name] = (parent.read_text(encoding="utf-8").strip() + "\n\n" +
                          addendum.read_text(encoding="utf-8").strip())
        elif name in transformed_files:
            parent, old, new = transformed_files[name]
            parent_text = parent.read_text(encoding="utf-8").strip()
            if old not in parent_text:
                raise ValueError(f"Expected V31 source clause not found: {old}")
            arms[name] = parent_text.replace(old, new, 1)
        elif name in full_files:
            arms[name] = full_files[name].read_text(encoding="utf-8").strip()
        else:
            arms[name] = production.rstrip() + "\n\n" + files[name].read_text(encoding="utf-8").strip()
    return arms


def summarize(rows: list[dict], arm_names: list[str], selection_baseline: str = "production") -> dict:
    """Aggregate only the three metrics from score_agent_predictions.py.

    An arm is safe only when its protocol rate, exact-cell rate and nonempty
    recall are all no worse than its declared baseline.  A safe specialist is
    retained when it improves any one of those official metrics, which keeps
    the requested Pareto memory without inventing a weighted total score.
    """
    summary: dict[str, dict] = {}
    for arm in arm_names:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        valid_rows = [row for row in arm_rows if row.get("protocol_ok") and row.get("teacher_score")]
        totals = {key: sum(int((row.get("teacher_score") or {}).get(key) or 0) for row in valid_rows)
                  for key in ("rows_compared", "exact_cells", "reference_nonempty", "nonempty_hits")}
        summary[arm] = {
            "n": len(arm_rows),
            "protocol_successes": len(valid_rows),
            "protocol_valid_rate": len(valid_rows) / len(arm_rows) if arm_rows else None,
            "exact_cell_rate": (totals["exact_cells"] / totals["rows_compared"]
                                if totals["rows_compared"] else None),
            "nonempty_recall": (totals["nonempty_hits"] / totals["reference_nonempty"]
                                if totals["reference_nonempty"] else None),
            "counts": totals,
            "mean_actor_calls": (sum(float(row.get("actor_calls") or 0) for row in arm_rows) / len(arm_rows)
                                 if arm_rows else None),
        }
    parent = summary.get(selection_baseline)
    for arm in arm_names:
        if arm == selection_baseline:
            continue
        candidate = summary[arm]
        if not parent:
            candidate["selection"] = {"retain_as_evidence": False, "reason": "baseline_missing"}
            continue
        metrics = ("protocol_valid_rate", "exact_cell_rate", "nonempty_recall")
        comparable = [metric for metric in metrics if parent.get(metric) is not None and candidate.get(metric) is not None]
        safe = bool(comparable) and all(candidate[metric] + 1e-9 >= parent[metric] for metric in comparable)
        gains = [metric for metric in comparable if candidate[metric] > parent[metric] + 1e-9]
        candidate["selection"] = {
            "safe_gate": safe,
            "official_metric_gains": gains if safe else [],
            "retain_as_evidence": bool(safe and gains),
            "rule": "师兄标准三指标 Pareto：不加权合成；只保留安全且至少一项提高的候选。",
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("train", "test"), required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", default="prompt-round-v6-20261024")
    parser.add_argument("--arms", default="production,v5,v6")
    parser.add_argument("--selection-baseline", default="production",
                        help="Arm used for Pareto safety gates; does not change official scores.")
    parser.add_argument("--exclude-known-results", action="store_true",
                        help="Exclude sample ids found in earlier prompt_*result.json files.")
    parser.add_argument("--sample-ids", default="",
                        help="Comma-separated deterministic replay ids; bypasses hash selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path, help="Continuously updated human-readable score page")
    args = parser.parse_args()
    arm_names = [name.strip() for name in args.arms.split(",") if name.strip()]
    if args.selection_baseline not in arm_names:
        raise SystemExit("--selection-baseline must be one of --arms")
    allowed_test_arms = {
        ("production", name) for name in
        ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16")
    }
    allowed_test_arms.add(("v9", "v27"))
    if args.dataset == "test" and tuple(arm_names) not in allowed_test_arms:
        raise SystemExit("Test pilot must compare one frozen candidate with its declared baseline")
    explicit_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    known_ids: set[str] = set()
    if args.exclude_known_results:
        for result_path in RESULT_DIR.glob("prompt_*result.json"):
            if result_path.resolve() == args.output.resolve():
                continue
            try:
                prior = json.loads(result_path.read_text(encoding="utf-8"))
                known_ids.update(str(item) for item in prior.get("sample_ids", []))
                known_ids.update(str(row.get("sample_id")) for row in prior.get("rows", [])
                                 if row.get("sample_id"))
            except (OSError, ValueError, TypeError):
                continue
    selected_rows = (select_explicit_rows(args.dataset, explicit_ids) if explicit_ids
                     else select_rows(args.dataset, args.limit, args.seed, known_ids))
    examples = [prepare(row) for row in selected_rows]
    if len(examples) != args.limit:
        raise SystemExit(f"Only selected {len(examples)} examples, expected {args.limit}")
    prompts = prompt_arms(arm_names)
    client = GlmAnthropicClient(min_interval=0.5, max_retries=1)
    backend = GlmBackend(client)
    result = {"kind": f"{args.dataset}_paired_prompt_pilot", "model": client.model,
              "seed": args.seed, "sample_ids": [item["sample_id"] for item in examples],
              "arms": arm_names, "scorer": "teacher_standard_exact_cell_v1",
              "parameters": {"max_rounds": 8, "attempts": 1, "temperature": 0.0}, "rows": []}
    for sample_index, example in enumerate(examples):
        order = arm_names if sample_index % 2 == 0 else list(reversed(arm_names))
        for arm in order:
            start_calls = client.actor_calls
            try:
                session = run_single_stage_phrase(example, {}, compose_system(prompts[arm]), backend,
                                                  max_rounds=8, attempts=1, sample_temperature=0.0,
                                                  structured=(arm in {"v17", "v18", "v19"}),
                                                  candidate_selection=(arm in {"v20", "v21", "v22"}))
                if session.get("ok"):
                    actions = session["plan"]["actions"]
                    prediction = {"score_key": example["score_key"], "phrase_id": example["phrase_id"],
                                  "jianzi_rows": [[int(action["source_index"]), action.get("jianzi_text") or action.get("text") or ""] for action in actions]}
                    teacher_score = score_prediction(example["reference"]["actions"], prediction)
                    row = {"sample_id": example["sample_id"], "arm": arm, "protocol_ok": True,
                           "score": teacher_score["exact_cell_rate"], "teacher_score": teacher_score,
                           "prediction": prediction,
                           "actor_calls": client.actor_calls - start_calls,
                           "rounds": len(session.get("trace") or []), "error": None}
                else:
                    row = {"sample_id": example["sample_id"], "arm": arm, "protocol_ok": False,
                           "score": None, "teacher_score": None, "prediction": None,
                           "actor_calls": client.actor_calls - start_calls,
                           "rounds": len(session.get("trace") or []),
                           "error": session.get("error") or "no_valid_edit_plan"}
            except Exception as exc:
                row = {"sample_id": example["sample_id"], "arm": arm, "protocol_ok": False,
                       "score": None, "teacher_score": None, "prediction": None,
                       "actor_calls": client.actor_calls - start_calls, "rounds": 0,
                       "error": f"{type(exc).__name__}: {exc}"}
            result["rows"].append(row)
            result["summary"] = summarize(result["rows"], arm_names, args.selection_baseline)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.html:
                write_result_html(result, args.html)
            print(json.dumps({key: row[key] for key in ("sample_id", "arm", "protocol_ok", "score", "actor_calls", "error")}, ensure_ascii=False), flush=True)
    result["summary"] = summarize(result["rows"], arm_names, args.selection_baseline)
    result["usage"] = client.snapshot()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.html:
        write_result_html(result, args.html)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
