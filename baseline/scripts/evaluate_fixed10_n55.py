#!/usr/bin/env python3
"""Run historical N55 on the frozen random ten whole scores with live HTML."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "guqinagent"
sys.path[:0] = [str(WORKSPACE / ".gepa_packages"), str(REPO), str(REPO / "baseline/gepa/src")]

from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend
from baseline.gepa_guqin.handbook import compose_system
from baseline.gepa_guqin.metric import replay_events
from baseline.gepa_guqin.teacher_exact_adapter import TeacherExactGuqinAdapter
from baseline.scripts.prompt_data_sources import TRAIN, training_identity
from baseline.scripts.run_auto_prompt_training import RunLock, check_config
from baseline.scripts.run_teacher_gepa_iteration import JsonlLogger, seed_prompt
from train.scripts.score_agent_predictions import canonical

PLAN = TRAIN.parent / "gepa_pilot_manifest.json"
OLD_RUN = REPO / "baseline/runs/N_gepa16_from_original"
RUN = REPO / "baseline/runs/M_whole10_from_original"
OUT = RUN / "n55_fixed10_baseline"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_examples():
    plan = read_json(PLAN)
    if plan.get("score_count") != 10 or plan.get("prefix_phrases") != 0:
        raise RuntimeError("当前清单不是冻结的十首整曲清单。")
    wanted, role_by_id = [], {}
    for role in ("feedback", "pareto", "confirmation"):
        ids = plan["splits"][role]["sample_ids"]
        wanted.extend(ids)
        role_by_id.update({sample_id: role for sample_id in ids})
    if len(wanted) != 172 or len(set(wanted)) != len(wanted):
        raise RuntimeError("十首清单应包含172个不重复乐句。")
    sources = {}
    with TRAIN.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["trajectory_id"] in role_by_id:
                    sources[row["trajectory_id"]] = row
    if set(sources) != set(wanted):
        raise RuntimeError("训练源中缺少冻结清单乐句。")
    examples, historical = [], {}
    for sample_id in wanted:
        row = deepcopy(sources[sample_id])
        row["input"].setdefault("phrase_handoff", {}).pop("previous_phrase", None)
        row["reference_plan"] = {"actions": materialize_actions(row["reference_plan"]["actions"])}
        historical[(str(row["score_key"]), str(row["phrase_id"]))] = row
        examples.append({
            "sample_id": sample_id, "score_key": row["score_key"], "phrase_id": row["phrase_id"],
            "split": "fixed10_baseline", "data_role": role_by_id[sample_id],
            "runtime_item": {"input": deepcopy(row["input"]), "baseline_plan": deepcopy(row["baseline_plan"])},
            "reference": {"actions": deepcopy(row["reference_plan"]["actions"])},
        })
    return sorted(examples, key=lambda e: (e["score_key"], e["phrase_id"])), historical, plan


def prompt_and_protocol(arm: str):
    prompt = ((OLD_RUN / "prompts/N55.txt").read_text(encoding="utf-8").strip()
              if arm == "N55" else seed_prompt("production")[0])
    selection = read_json(OLD_RUN / "selection.json", {})
    expected = next(x["prompt_sha256"] for x in selection["official_candidate_scores"] if x["version"] == arm)
    if hashlib.sha256(prompt.encode()).hexdigest() != expected:
        raise RuntimeError(f"{arm}提示词与历史登记哈希不一致。")
    old_manifest = read_json(OLD_RUN / "experiment_manifest.json")
    return prompt, old_manifest["parameters"]


def score_totals(rows):
    totals = Counter()
    by_score = defaultdict(Counter)
    for row in rows:
        score = row["teacher_score"]
        for target in (totals, by_score[row["score_key"]]):
            target["samples"] += 1
            target["protocol_valid"] += int(row["protocol_ok"])
            for key in ("rows_compared", "exact_cells", "reference_nonempty", "predicted_nonempty", "nonempty_hits"):
                target[key] += int(score.get(key) or 0)
            target["pitch_matched"] += sum(x[1] == "matched" for x in row.get("pitch_cells", []))
            target["pitch_mismatched"] += sum(x[1] == "mismatched" for x in row.get("pitch_cells", []))
    def rates(value):
        return {
            **dict(value),
            "exact_cell_rate": value["exact_cells"] / value["rows_compared"] if value["rows_compared"] else None,
            "nonempty_recall": value["nonempty_hits"] / value["reference_nonempty"] if value["reference_nonempty"] else None,
            "protocol_valid_rate": value["protocol_valid"] / value["samples"] if value["samples"] else None,
        }
    return rates(totals), {key: rates(value) for key, value in by_score.items()}


def pitch_cells(example, prediction, historical):
    try:
        events, _, _ = replay_events(example, prediction, historical)
    except Exception as exc:
        return [], f"音高重放失败：{type(exc).__name__}: {exc}"
    cells = {}
    for event in events:
        audit = event.get("pitch_audit") or {}
        covered = audit.get("covered_indices") or ([audit["index"]] if audit.get("index") is not None else [])
        pairs = audit.get("pairs") or []
        details = "、".join(
            f"{p.get('jianpu_name', '?')} vs {p.get('jianzi_name', '?')}，偏差{float(p.get('delta_cents', 0)):.0f}音分"
            for p in pairs
        ) or str(audit.get("reason") or audit.get("jianpu") or "无可解析音高")
        for index in covered:
            cells[int(index)] = [int(index), str(audit.get("status") or "not_applicable"), details]
    return [cells[key] for key in sorted(cells)], None


def render(result, examples):
    OUT.mkdir(parents=True, exist_ok=True)
    refs = {e["sample_id"]: {int(a["source_index"]): str(a.get("jianzi_text") or a.get("text") or "")
                             for a in e["reference"]["actions"]} for e in examples}
    totals, by_score = score_totals(result["rows"])
    result["aggregate"] = totals
    score_rows = []
    expected_by_score = Counter(e["score_key"] for e in examples)
    for key in sorted(expected_by_score):
        item = by_score.get(key, {})
        score_rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{item.get('samples', 0)}/{expected_by_score[key]}</td>"
            f"<td>{pct(item.get('exact_cell_rate'))}</td><td>{pct(item.get('nonempty_recall'))}</td>"
            f"<td>{pct(item.get('protocol_valid_rate'))}</td><td>✓ {item.get('pitch_matched', 0)} / ✗ {item.get('pitch_mismatched', 0)}</td></tr>"
        )
    blocks = []
    for row in reversed(result["rows"]):
        reference = refs[row["sample_id"]]
        generated = {int(i): str(text) for i, text in row["prediction"]["jianzi_rows"]}
        pitches = {int(i): (status, detail) for i, status, detail in row.get("pitch_cells", [])}
        lines = []
        for index in sorted(set(reference) | set(generated) | set(pitches)):
            ref, pred = reference.get(index, ""), generated.get(index, "")
            if canonical(ref) == canonical(pred):
                kind, label = "exact", "一致"
            elif ref and not pred:
                kind, label = "missing", "漏填"
            elif pred and not ref:
                kind, label = "extra", "多填"
            else:
                kind, label = "different", "文字不同"
            status, detail = pitches.get(index, ("not_applicable", "该格无独立可核音高"))
            mark = "✓" if status == "matched" else "✗" if status == "mismatched" else "—"
            lines.append(f"<tr class='{kind}'><td>{index}</td><td>{html.escape(ref or '（空）')}</td>"
                         f"<td>{html.escape(pred or '（空）')}</td><td>{label}</td>"
                         f"<td title='{html.escape(detail, quote=True)}'>{mark} {html.escape(detail)}</td></tr>")
        score = row["teacher_score"]
        blocks.append(
            f"<details><summary>{html.escape(row['sample_id'])}｜{html.escape(row['data_role'])}｜"
            f"精确 {pct(score.get('exact_cell_rate'))}｜召回 {pct(score.get('nonempty_recall'))}</summary>"
            f"<table><tr><th>音序</th><th>训练标注</th><th>{html.escape(result['configuration']['arm'])}生成</th><th>差异</th><th>音高核对（±50音分）</th></tr>{''.join(lines)}</table></details>"
        )
    state = {"running": "运行中", "complete": "已完成", "failed": "已中断，保留断点"}.get(result["status"], result["status"])
    refresh = "<meta http-equiv='refresh' content='15'>" if result["status"] == "running" else ""
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>{refresh}
<title>{html.escape(result['configuration']['arm'])}固定十首整曲评测</title><style>
body{{font:15px/1.65 system-ui,'Microsoft YaHei';max-width:1300px;margin:28px auto;padding:0 18px;background:#f4f6fa;color:#263146}}
section,details{{background:#fff;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:14px 0}}summary{{cursor:pointer;font-weight:700}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #e1e6ee;text-align:left;vertical-align:top}}.exact{{background:#eaf8ef}}.missing{{background:#fff0f0}}.extra{{background:#fff7e5}}.different{{background:#f5f0ff}}code{{overflow-wrap:anywhere}}</style>
<h1>{html.escape(result['configuration']['arm'])} · 固定随机十首整曲</h1><section><h2>{state}：{len(result['rows'])}/{len(examples)} 个乐句</h2>
<p>提示词固定为{html.escape(result['configuration']['arm'])}；模型为GLM 5.3；十首名单由训练集按固定种子随机抽取，运行中不会重新抽样。</p>
<p>当前总分：逐格精确率 <b>{pct(totals.get('exact_cell_rate'))}</b>，非空召回 <b>{pct(totals.get('nonempty_recall'))}</b>，协议合规 <b>{pct(totals.get('protocol_valid_rate'))}</b>。</p>
<p>音高核对：✓ {totals.get('pitch_matched', 0)}；✗ {totals.get('pitch_mismatched', 0)}。音高按工具重放结果、±50音分容差核验。</p>
<p>最近更新（UTC）：{html.escape(result['updated_at'])}。本页15秒自动刷新。</p></section>
<section><h2>十首曲目实时评分</h2><table><tr><th>曲谱</th><th>已完成乐句</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th><th>音高</th></tr>{''.join(score_rows)}</table></section>
<section><h2>逐句：标注、{html.escape(result['configuration']['arm'])}生成与音高核对</h2><p>展开任一乐句查看全部格子。绿色一致，红色漏填，黄色多填，紫色文字不同。</p>{''.join(blocks) or '<p>等待首条结果……</p>'}</section></html>"""
    (OUT / f"{result['configuration']['arm'].lower()}_fixed10_live.html").write_text(page, encoding="utf-8", newline="\n")
    save(OUT / "result.json", result)


def pct(value):
    return "—" if value is None else f"{float(value):.2%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("N0", "N55"), default="N55", help="N0原版或历史N55")
    parser.add_argument("--check", action="store_true", help="只核验，不调用API")
    parser.add_argument("--recover-stale-lock", action="store_true",
                        help="归档超过2分钟未更新的锁后，从已保存断点恢复")
    args = parser.parse_args()
    global OUT
    OUT = RUN / f"{args.arm.lower()}_fixed10_baseline"
    configuration = check_config()
    examples, historical, plan = build_examples()
    prompt, parameters = prompt_and_protocol(args.arm)
    identity = {
        "mode": f"{args.arm}_on_frozen_random_10_whole_scores", "arm": args.arm, "source": training_identity(),
        "pilot_manifest_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "effective_prompt_sha256": hashlib.sha256(compose_system(prompt).encode()).hexdigest(),
        "model": os.environ.get("GLM_MODEL"), "sample_ids": [e["sample_id"] for e in examples],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "result.json"
    result = read_json(result_path, {"configuration": identity, "status": "running", "rows": [],
                                    "started_at": now(), "updated_at": now()})
    if result["configuration"] != identity:
        raise RuntimeError(f"{args.arm}十首评测配置发生变化，请使用新的输出目录。")
    render(result, examples)
    if args.check:
        print(json.dumps({"status": "verified", "scores": 10, "phrases": len(examples),
                          "roles": {k: v["rows"] for k, v in plan["splits"].items()},
                          "html": str(OUT / f"{args.arm.lower()}_fixed10_live.html")}, ensure_ascii=False, indent=2))
        return 0
    if result["status"] == "complete":
        print(f"{args.arm}固定十首整曲评测已完成；未产生新的API调用。")
        return 0
    lock = RunLock(OUT)
    lock.path = OUT / f"{args.arm.lower()}_fixed10.lock"
    if args.recover_stale_lock and lock.path.exists():
        previous = read_json(lock.path, {})
        created = str(previous.get("created_at") or "")
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(created)).total_seconds()
        except ValueError:
            age = 0
        if age < 120:
            raise RuntimeError("锁创建不足两分钟，拒绝将其当作失效锁。")
        archived = lock.path.with_name(f"{args.arm.lower()}_fixed10.stale_lock_{int(time.time())}.json")
        lock.path.replace(archived)
        print(f"已归档疑似失效锁：{archived.name}；从断点恢复。", flush=True)
    with lock:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env", override=True)
        client = GlmAnthropicClient(min_interval=0.5)
        adapter = TeacherExactGuqinAdapter(
            examples, historical, GlmBackend(client), max_rounds=int(parameters["max_rounds"]),
            attempts=int(parameters["attempts"]), sample_temperature=float(parameters["temperature"]),
            log=JsonlLogger(OUT / "rollout_events.jsonl"),
        )
        cache_key = hashlib.sha256(compose_system(prompt).encode()).hexdigest()[:16]
        sessions = adapter._session_cache.setdefault(cache_key, {})
        completed = {row["sample_id"]: row for row in result["rows"]}
        for sample_id, row in completed.items():
            sessions[sample_id] = {"ok": row["protocol_ok"], "trace": [],
                "plan": {"actions": [{"source_index": int(i), "jianzi_text": text}
                                       for i, text in row["prediction"]["jianzi_rows"]]},
                "error": row.get("error")}
        try:
            for example in examples:
                if example["sample_id"] in completed:
                    continue
                result.update(status="running", current_sample=example["sample_id"], updated_at=now())
                render(result, examples)
                for attempt in range(3):
                    try:
                        output = adapter.evaluate([example], {"single_stage": prompt}).outputs[0]
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(30)
                prediction = {"score_key": example["score_key"], "phrase_id": example["phrase_id"],
                              "jianzi_rows": output["jianzi_rows"]}
                pitch, pitch_error = pitch_cells(example, prediction, historical)
                row = {"sample_id": example["sample_id"], "score_key": example["score_key"],
                       "phrase_id": example["phrase_id"], "data_role": example["data_role"],
                       "protocol_ok": output["protocol_ok"], "teacher_score": output["teacher_score"],
                       "prediction": prediction, "difference_counts": output["difference_counts"],
                       "pitch_cells": pitch, "pitch_error": pitch_error,
                       "error": None if output["protocol_ok"] else "rollout_or_predecessor_failure"}
                result["rows"].append(row)
                completed[row["sample_id"]] = row
                result["updated_at"] = now()
                render(result, examples)
                print(f"[{args.arm} {len(result['rows'])}/{len(examples)}] {row['sample_id']} "
                      f"exact={pct(row['teacher_score'].get('exact_cell_rate'))}", flush=True)
            result.update(status="complete", current_sample=None, completed_at=now(), updated_at=now())
            render(result, examples)
        except BaseException as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}", updated_at=now())
            render(result, examples)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
