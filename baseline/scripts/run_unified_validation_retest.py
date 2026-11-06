#!/usr/bin/env python3
"""Full-score, resumable common-validation retest for historical prompts.

Predictions are generated anew on the 25% fixed validation split.  The model
never receives reference_plan; references are used only after generation by
the exact-cell scorer matching score_agent_predictions.py.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "guqinagent"
DATASET = ROOT / "_tmp_zip_inspect" / "pitch_eligible_two_or_half"
OUT = ROOT / "results" / "unified_validation_retest"
REPORT = ROOT / "reports" / "unified_validation_retest_progress.html"
SCORE_REPORT = ROOT / "reports" / "unified_validation_retest_scores.html"

import sys
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ROOT / "scripts"))

from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for
from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
from baseline.gepa_guqin.handbook import compose_system
from baseline.gepa_guqin.rollout import ScoreRunState, run_single_stage_phrase
from prompt_candidate_paired_experiment import prompt_arms
from teacher_standard_score import score_prediction


# V15 and V27 are byte-identical full prompts.  V24 is a one-off local repair
# experiment, not a standalone system prompt, so it has no comparable rollout.
VERSIONS = ("production", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32")
ALIASES = {"v27": "v15"}
DISPLAY_VERSIONS = VERSIONS + ("v24",)


def load_rows() -> dict[str, list[dict[str, Any]]]:
    keys = set((DATASET / "fixed_validation_score_keys_25pct.txt").read_text(encoding="utf-8").split())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (DATASET / "inferred_trajectories_train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row["score_key"]) in keys:
                grouped[str(row["score_key"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int((row.get("input", {}).get("event_range") or {}).get("start", 0)))
    return dict(sorted(grouped.items()))


def prepare(row: dict[str, Any], state: ScoreRunState) -> tuple[dict[str, Any], dict[str, Any]]:
    input_data = deepcopy(row["input"])
    input_data.setdefault("phrase_handoff", {}).pop("previous_phrase", None)
    working = {"trajectory_id": row["trajectory_id"], "score_key": row["score_key"],
               "phrase_id": row["phrase_id"], "input": input_data,
               "baseline_plan": deepcopy(row.get("baseline_plan") or {})}
    state.prepare(working)
    example = {"sample_id": row["trajectory_id"], "score_key": row["score_key"],
               "phrase_id": row["phrase_id"], "runtime_item": working,
               "reference": {"actions": materialize_actions(row["reference_plan"]["actions"])}}
    return example, working


def saved_records(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    if not path.exists():
        return result
    for line in path.open(encoding="utf-8"):
        if line.strip():
            item = json.loads(line)
            result[str(item["sample_id"])] = item
    return result


def summary(records: dict[str, dict[str, Any]], score_keys: set[str]) -> dict[str, Any]:
    valid = [row for row in records.values() if row.get("protocol_ok") and row.get("teacher_score")]
    totals = {key: sum(int(row["teacher_score"].get(key) or 0) for row in valid)
              for key in ("rows_compared", "exact_cells", "reference_nonempty", "nonempty_hits")}
    per_score: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        per_score[str(row["score_key"])].append(row)
    per_score_summary = {}
    for key, rows in per_score.items():
        count = {field: sum(int(row["teacher_score"].get(field) or 0) for row in rows)
                 for field in totals}
        per_score_summary[key] = {
            "phrases": len(rows),
            "exact_cell_rate": count["exact_cells"] / count["rows_compared"] if count["rows_compared"] else None,
            "nonempty_recall": count["nonempty_hits"] / count["reference_nonempty"] if count["reference_nonempty"] else None,
        }
    complete_scores = sum(all(row["trajectory_id"] in records for row in rows) for rows in DATA_ROWS.values())
    return {"completed_phrases": len(records), "expected_phrases": sum(len(DATA_ROWS[key]) for key in score_keys),
            "completed_scores": complete_scores, "expected_scores": len(score_keys),
            "protocol_valid_rate": len(valid) / len(records) if records else None,
            "exact_cell_rate": totals["exact_cells"] / totals["rows_compared"] if totals["rows_compared"] else None,
            "nonempty_recall": totals["nonempty_hits"] / totals["reference_nonempty"] if totals["reference_nonempty"] else None,
            "counts": totals, "per_score": per_score_summary}


def write_dashboard(all_summaries: dict[str, dict[str, Any]], started: str,
                    output_path: Path = REPORT) -> None:
    def rate(value: Any) -> str:
        return "—" if value is None else f"{float(value):.2%}"
    rows = []
    details = []
    for version in DISPLAY_VERSIONS:
        item = all_summaries.get(version)
        if item is None:
            status = "与 V15 共用结果" if version in ALIASES else ("局部修复，未纳入" if version == "v24" else "未开始")
            rows.append(f"<tr><td>{html.escape(version)}</td><td>{status}</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>")
            continue
        status = "完成" if item["completed_phrases"] == item["expected_phrases"] else "进行中"
        rows.append(f"<tr><td>{html.escape(version)}</td><td>{status}</td><td>{item['completed_phrases']}/{item['expected_phrases']}</td>"
                    f"<td>{item['completed_scores']}/{item['expected_scores']}</td><td>{rate(item['exact_cell_rate'])}</td>"
                    f"<td>{rate(item['nonempty_recall'])}</td><td>{rate(item['protocol_valid_rate'])}</td></tr>")
        pieces = []
        for score_key, score in sorted((item.get("per_score") or {}).items()):
            pieces.append(f"<tr><td>{html.escape(score_key)}</td><td>{score['phrases']}</td><td>{rate(score['exact_cell_rate'])}</td><td>{rate(score['nonempty_recall'])}</td></tr>")
        details.append(f"<details><summary>{html.escape(version)}｜逐曲成绩（{len(pieces)} 首）</summary><table><thead><tr><th>曲目</th><th>已完成短语</th><th>逐格精确率</th><th>非空召回</th></tr></thead><tbody>{''.join(pieces)}</tbody></table></details>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"""<!doctype html><meta charset='utf-8'><title>统一验证集复测成绩</title>
<style>body{{max-width:1050px;margin:32px auto;font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif;background:#f6f8fb;color:#17212b}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #d9e1ea;padding:8px;text-align:left}}th{{background:#edf3f8}}.note{{color:#536271}}</style>
<h1>历史提示词统一验证集复测</h1><p class='note'>开始：{html.escape(started)}；固定验证集：1,079 条短语／61 首曲。结果会在每个短语完成后更新。V15 与 V27 文本相同，V27 使用 V15 的同一份新生成结果；V24 不是完整提示词，未纳入。</p>
<table><thead><tr><th>版本</th><th>状态</th><th>短语进度</th><th>曲目进度</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th></tr></thead><tbody>{''.join(rows)}</tbody></table><section>{''.join(details)}</section>""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", default=",".join(VERSIONS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="regenerate the all-version score page without API calls")
    parser.add_argument("--max-scores", type=int, default=None, help="development smoke limit; omit for all 61")
    args = parser.parse_args()
    versions = [value.strip() for value in args.versions.split(",") if value.strip()]
    unknown = set(versions) - set(VERSIONS) - set(ALIASES)
    if unknown:
        raise SystemExit(f"unknown versions: {sorted(unknown)}")
    score_keys = list(DATA_ROWS)
    if args.max_scores is not None:
        score_keys = score_keys[:args.max_scores]
    expected = sum(len(DATA_ROWS[key]) for key in score_keys)
    if args.dry_run:
        print(json.dumps({"versions": versions, "scores": len(score_keys), "phrases_per_version": expected,
                          "total_phrase_rollouts": expected * len([v for v in versions if v not in ALIASES]),
                          "aliases": ALIASES, "output": str(OUT)}, ensure_ascii=False, indent=2))
        return 0
    if args.render_only:
        rendered = {}
        for version in versions:
            source = ALIASES.get(version, version)
            path = OUT / f"{source}.jsonl"
            if path.exists():
                rendered[version] = summary(saved_records(path), set(score_keys))
        write_dashboard(rendered, "本地结果快照", SCORE_REPORT)
        print(json.dumps({"score_page": str(SCORE_REPORT), "versions_with_results": sorted(rendered)}, ensure_ascii=False))
        return 0
    if not os.getenv("GLM_API_KEY") or not os.getenv("GLM_BASE_URL"):
        raise SystemExit("GLM_API_KEY and GLM_BASE_URL must be set for a live run")
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    prompts = prompt_arms([version for version in versions if version not in ALIASES])
    client = GlmAnthropicClient(min_interval=0.5, max_retries=2)
    backend = GlmBackend(client)
    all_summaries: dict[str, dict[str, Any]] = {}
    for version in versions:
        source_version = ALIASES.get(version, version)
        source_path = OUT / f"{source_version}.jsonl"
        target_path = OUT / f"{version}.jsonl"
        if version in ALIASES and source_path.exists():
            target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            all_summaries[version] = summary(saved_records(target_path), set(score_keys))
            write_dashboard(all_summaries, started)
            continue
        records = saved_records(target_path)
        system = compose_system(prompts[version])
        with target_path.open("a", encoding="utf-8", newline="\n") as output:
            for score_key in score_keys:
                state = ScoreRunState()
                poisoned = False
                for raw_row in DATA_ROWS[score_key]:
                    sample_id = str(raw_row["trajectory_id"])
                    prior = records.get(sample_id)
                    example, working = prepare(raw_row, state)
                    if prior is not None:
                        if prior.get("protocol_ok") and prior.get("prediction"):
                            state.commit(working, {"ok": True, "plan": {"actions": [
                                {"source_index": cell[0], "jianzi_text": cell[1]}
                                for cell in prior["prediction"].get("jianzi_rows") or []]}})
                        else:
                            poisoned = True
                        continue
                    if poisoned:
                        break
                    try:
                        session = run_single_stage_phrase(example, state.historical, system, backend,
                                                          max_rounds=8, attempts=1, sample_temperature=0.0,
                                                          structured=version in {"v17", "v18", "v19"},
                                                          candidate_selection=version in {"v20", "v21", "v22"})
                        if session.get("ok"):
                            prediction = {"score_key": raw_row["score_key"], "phrase_id": raw_row["phrase_id"],
                                          "jianzi_rows": [[int(action["source_index"]), action.get("jianzi_text") or action.get("text") or ""] for action in session["plan"]["actions"]]}
                            record = {"sample_id": sample_id, "score_key": raw_row["score_key"], "phrase_id": raw_row["phrase_id"],
                                      "arm": version, "protocol_ok": True, "prediction": prediction,
                                      "teacher_score": score_prediction(example["reference"]["actions"], prediction),
                                      "rounds": len(session.get("trace") or []), "error": None}
                            state.commit(working, session)
                        else:
                            record = {"sample_id": sample_id, "score_key": raw_row["score_key"], "phrase_id": raw_row["phrase_id"],
                                      "arm": version, "protocol_ok": False, "prediction": None, "teacher_score": None,
                                      "rounds": len(session.get("trace") or []), "error": session.get("error") or "no_valid_edit_plan",
                                      "failure_trace": session.get("trace") or []}
                            poisoned = True
                    except Exception as exc:
                        record = {"sample_id": sample_id, "score_key": raw_row["score_key"], "phrase_id": raw_row["phrase_id"],
                                  "arm": version, "protocol_ok": False, "prediction": None, "teacher_score": None,
                                  "rounds": 0, "error": f"{type(exc).__name__}: {exc}"}
                        poisoned = True
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    records[sample_id] = record
                    all_summaries[version] = summary(records, set(score_keys))
                    write_dashboard(all_summaries, started)
                    print(json.dumps({"version": version, "sample_id": sample_id, "ok": record["protocol_ok"],
                                      "exact": (record.get("teacher_score") or {}).get("exact_cell_rate"), "error": record["error"]}, ensure_ascii=False), flush=True)
        all_summaries[version] = summary(records, set(score_keys))
        (OUT / f"{version}_summary.json").write_text(json.dumps(all_summaries[version], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_dashboard(all_summaries, started)
    (OUT / "run_manifest.json").write_text(json.dumps({"started": started, "versions": versions, "score_keys": score_keys,
        "scorer": "guqinagent/train/scripts/score_agent_predictions.py", "parameters": {"temperature": 0.0, "max_rounds": 8, "attempts": 1},
        "usage": client.snapshot(), "summaries": all_summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


DATA_ROWS = load_rows()

if __name__ == "__main__":
    raise SystemExit(main())
