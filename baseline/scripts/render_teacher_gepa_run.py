#!/usr/bin/env python3
"""Render a self-contained audit page for the teacher-standard GEPA run."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from train.scripts.score_agent_predictions import canonical


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, ValueError, TypeError):
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                if line.strip():
                    records.append(json.loads(line))
            except ValueError:
                continue
    return records


def pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def f4(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def diff(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile="parent", tofile="candidate", lineterm=""
    )) or "文本相同"


def render(run_dir: Path, output: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "experiment_manifest.json", {})
    split = (manifest.get("split") or load_json(run_dir / "data" / "split_manifest.json", {}))
    role_table = ""
    if split.get("pilot"):
        role_rows = []
        for role, part in split["pilot"].items():
            role_rows.append(f"<tr><th>{esc(role)}</th><td>{part['score_count']}</td><td>{part['rows']}</td>"
                             f"<td>{esc('、'.join(part['score_keys']))}</td></tr>")
        role_table = "<table><tr><th>数据角色</th><th>曲谱数</th><th>段数</th><th>曲谱名单</th></tr>" + "".join(role_rows) + "</table>"
    iterations = load_json(run_dir / "run_log.json", [])
    series = manifest.get("series")
    from baseline.scripts.prompt_series import candidate_versions
    versions = candidate_versions(series, iterations) if series else {}
    def version(index):
        return versions.get(index, f"候选{index}")
    candidates = load_json(run_dir / "candidates.json", [])
    if not candidates and series and (run_dir / "prompts" / f"{series}0.txt").exists():
        candidates = [{"single_stage": (run_dir / "prompts" / f"{series}0.txt").read_text(encoding="utf-8").strip()}]
    selection = load_json(run_dir / "selection.json", {})
    events = load_jsonl(run_dir / "teacher_gepa_events.jsonl")
    feedback_events = [event for event in events if event.get("event") == "reflection_feedback"]
    feedback_by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in feedback_events:
        feedback_by_batch[int(event.get("feedback_batch") or 0)].append(event)
    # Old checkpoints predate structured proposal callbacks. Recover their
    # rejected prompt text from the saved UTF-8 GEPA log, never invent it.
    text_path = run_dir / "run_log.txt"
    text_log = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
    legacy_proposals = {}
    for match in re.finditer(
        r"^Iteration (\d+): Proposed new text for single_stage: (.*?)(?=^Iteration \d+:|\Z)",
        text_log, re.MULTILINE | re.DOTALL,
    ):
        legacy_proposals[int(match.group(1))] = match.group(2).strip()

    feedback_ids = list(split.get("feedback_sample_ids") or [])
    pareto_ids = list(split.get("pareto_sample_ids") or [])
    sources = {}
    for role in ("feedback", "pareto"):
        for example in load_jsonl(run_dir / "data" / f"examples_{role}.jsonl"):
            sources[example["sample_id"]] = example
    prediction_events = {}
    for event in events:
        if event.get("event") == "teacher_phrase_scored" and "prediction" in event:
            prediction_events[(event.get("candidate_sha256"), event.get("sample_id"))] = event

    def round_comparison(ids, parent_text, child_text):
        parent_sha = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
        child_sha = hashlib.sha256(child_text.encode("utf-8")).hexdigest() if child_text else None
        blocks = []
        for sid in ids:
            example = sources.get(sid)
            old = prediction_events.get((parent_sha, sid))
            new = prediction_events.get((child_sha, sid))
            if not example or not old:
                continue
            refs = {int(a["source_index"]): str(a.get("text") or a.get("jianzi_text") or "")
                    for a in example["reference"]["actions"]}
            old_rows = {int(i): str(text) for i, text in old["prediction"]["jianzi_rows"]}
            new_rows = {int(i): str(text) for i, text in new["prediction"]["jianzi_rows"]} if new else {}
            corrected = regressed = 0
            rows = []
            for index in sorted(set(refs) | set(old_rows) | set(new_rows)):
                ref, before, after = refs.get(index, ""), old_rows.get(index, ""), new_rows.get(index, "")
                old_ok = canonical(ref) == canonical(before)
                new_ok = canonical(ref) == canonical(after) if new else False
                corrected += int(new is not None and not old_ok and new_ok)
                regressed += int(new is not None and old_ok and not new_ok)
                change = ("改对" if not old_ok and new_ok else "改错" if old_ok and not new_ok else
                          "保持正确" if new_ok else "仍有差异") if new else "未完成新提案评测"
                rows.append(f"<tr><th>{index}</th><td>{esc(ref or '（空）')}</td>"
                            f"<td class='{'eq' if old_ok else 'neq'}'>{esc(before or '（空）')}</td>"
                            f"<td class='{'eq' if new_ok else 'neq'}'>{esc((after or '（空）') if new else '未评测')}</td>"
                            f"<td>{esc(change)}</td></tr>")
            blocks.append(f"<details><summary>{esc(sid)}：改对 {corrected} 格，改错 {regressed} 格</summary>"
                          "<table><tr><th>音序</th><th>训练标注</th><th>父代实际生成</th><th>本轮新提案实际生成</th><th>变化</th></tr>"
                          + "".join(rows) + "</table></details>")
        return "".join(blocks) or "<p class='muted'>等待实际rollout结果。</p>"

    def sampled(entry: dict[str, Any]) -> list[str]:
        result = []
        for value in entry.get("subsample_ids") or []:
            if isinstance(value, int) and 0 <= value < len(feedback_ids):
                result.append(feedback_ids[value])
            else:
                result.append(str(value))
        return result

    iteration_blocks = []
    for position, entry in enumerate(iterations if isinstance(iterations, list) else [], 1):
        parent = entry.get("selected_program_candidate")
        children = entry.get("new_program_indices") or []
        before = entry.get("subsample_scores") or []
        after = entry.get("new_subsample_scores") or []
        before_sum = sum(float(x) for x in before) if before else None
        after_sum = sum(float(x) for x in after) if after else None
        accepted = bool(children)
        iteration_number = int(entry.get("i", position - 1)) + 1
        exact_feedback = [event for event in events if
                          event.get("event") == "gepa_reflective_dataset" and
                          event.get("iteration_id") == entry.get("iteration_id")]
        if exact_feedback:
            batch_feedback = []
            for component_records in exact_feedback[-1].get("dataset", {}).values():
                for sid, record in zip(sampled(entry), component_records):
                    batch_feedback.append({"sample_id": record.get("sample_id") or sid, "score": record.get("score"),
                                           "feedback": record.get("Feedback"),
                                           "feedback_access": record.get("Feedback access", "features_only"),
                                           "difference_counts": record.get("Difference counts", {})})
        else:
            # Batch counters reset after process restarts. Match legacy records
            # by actual sample, parent prompt hash, and score rather than batch #.
            parent_text = str((candidates[parent] or {}).get("single_stage") or "") if (
                isinstance(parent, int) and 0 <= parent < len(candidates)
            ) else ""
            parent_sha = hashlib.sha256(parent_text.encode("utf-8")).hexdigest()
            batch_feedback = []
            for sid, expected_score in zip(sampled(entry), before):
                matches = [event for event in feedback_events if
                           event.get("sample_id") == sid and
                           event.get("candidate_sha256") == parent_sha and
                           event.get("score") == expected_score]
                if matches:
                    batch_feedback.append(matches[-1])
        proposal_events = [event for event in events if
                           event.get("event") == "gepa_proposal" and
                           event.get("iteration") == iteration_number]
        proposal_text = (proposal_events[-1].get("new_instructions", {}).get("single_stage", "")
                         if proposal_events else legacy_proposals.get(iteration_number, ""))
        parent_text = str((candidates[parent] or {}).get("single_stage") or "") if (
            isinstance(parent, int) and 0 <= parent < len(candidates)
        ) else ""
        proposal_html = (
            f"<details><summary>本轮提案全文（包括淘汰版本）</summary><pre>{esc(proposal_text)}</pre></details>"
            f"<details><summary>本轮相对父代的修改</summary><pre>{esc(diff(parent_text, proposal_text))}</pre></details>"
        ) if proposal_text else "<p class='muted'>本轮未保存提案正文。</p>"
        feedback_html = "".join(
            f"<details><summary>{esc(item.get('sample_id'))} · 得分 {f4(item.get('score'))}</summary>"
            f"<p><b>差异计数：</b>{esc(json.dumps(item.get('difference_counts') or {}, ensure_ascii=False))}</p>"
            f"<p>反馈来源：{esc(item.get('feedback_access') or '历史记录')}</p>"
            f"<pre>{esc(item.get('feedback') or '未记录反馈')}</pre></details>"
            for item in batch_feedback
        ) or "<p class='muted'>该轮尚未保存自然语言反馈。</p>"
        decision = (
            f"{', '.join(version(index) for index in children)} 在 minibatch 上超过父代，已进入 D_pareto 完整评估。"
            if accepted else "新候选未在 minibatch 上超过父代，本轮淘汰。"
        )
        iteration_blocks.append(
            f"<article><h3>第 {position} 轮{' · ' + esc(series + str(position)) if series else ''} · {esc(entry.get('iteration_id') or '')}</h3>"
            f"<p><b>父候选：</b>{esc(version(parent))} · <b>采样轨迹：</b>{esc(', '.join(sampled(entry)))}"
            f" · <b>minibatch 分数：</b>{f4(before_sum)} → {f4(after_sum)}</p>"
            f"<p class='{'pass' if accepted else 'fail'}'><b>本轮决定：</b>{esc(decision)}</p>"
            f"<details open><summary>本轮自然语言反馈</summary>{feedback_html}</details>"
            + "<details open><summary>本轮训练标注—父代—新提案一一对照</summary>"
            + round_comparison(sampled(entry), parent_text, proposal_text) + "</details>"
            + proposal_html +
            f"<details><summary>本轮原始 GEPA 日志</summary><pre>{esc(json.dumps(entry, ensure_ascii=False, indent=2))}</pre></details></article>"
        )

    scores = selection.get("official_candidate_scores") or []
    score_by_idx = {int(item["candidate_idx"]): item for item in scores if "candidate_idx" in item}
    candidate_blocks = []
    for index, candidate in enumerate(candidates if isinstance(candidates, list) else []):
        prompt = str((candidate or {}).get("single_stage") or "")
        parent_idx = None
        for entry in iterations if isinstance(iterations, list) else []:
            if index in (entry.get("new_program_indices") or []):
                parent_idx = int(entry.get("selected_program_candidate") or 0)
                break
        parent_prompt = ""
        if parent_idx is not None and parent_idx < len(candidates):
            parent_prompt = str((candidates[parent_idx] or {}).get("single_stage") or "")
        metrics = score_by_idx.get(index, {})
        tags = []
        if index == selection.get("gepa_best_idx"):
            tags.append("GEPA验证最优")
        if index == selection.get("safe_best_idx"):
            tags.append("三指标安全选择")
        candidate_blocks.append(
            f"<article><h3>{esc(version(index))} {' · '.join(tags)}</h3>"
            f"<p>SHA-256: <code>{esc(hashlib.sha256(prompt.encode('utf-8')).hexdigest())}</code> · "
            f"GEPA val={f4(metrics.get('gepa_val_score'))} · 逐格={pct(metrics.get('exact_cell_rate'))} · "
            f"非空召回={pct(metrics.get('nonempty_recall'))} · 协议={pct(metrics.get('protocol_valid_rate'))}</p>"
            + (f"<details><summary>相对父候选 {esc(version(parent_idx))} 的完整修改</summary><pre>{esc(diff(parent_prompt, prompt))}</pre></details>"
               if parent_idx is not None else "<p class='muted'>种子提示词</p>")
            + f"<details><summary>完整提示词</summary><pre>{esc(prompt)}</pre></details></article>"
        )

    best_idx = selection.get("gepa_best_idx")
    safe_idx = selection.get("safe_best_idx")
    def origin(index):
        if index is None:
            return "尚未选择"
        if index == 0:
            return f"师兄库原版提示词（{version(0)}）" if series else "种子提示词（候选0）"
        for position, entry in enumerate(iterations, 1):
            if index in (entry.get("new_program_indices") or []):
                return f"第{position}轮新提案（{version(index)}）"
        return f"候选{index}"
    state = "尚未完成" if not selection else "已完成"
    completed_at = selection.get("completed_at", "")
    if completed_at and any(e.get("event") == "gepa_iteration_start"
                            and e.get("logged_at", "") > completed_at for e in events):
        state = "续跑中（下方选择为上一批结果）"
    last_predictions = {}
    for event in events:
        if event.get("event") == "teacher_phrase_scored" and "prediction" in event:
            last_predictions[(event.get("candidate_sha256"), event.get("sample_id"))] = event
    comparisons = []
    for (prompt_sha, sample_id), event in last_predictions.items():
        example = sources.get(sample_id)
        if not example:
            continue
        refs = {int(a["source_index"]): str(a.get("text") or a.get("jianzi_text") or "")
                for a in example["reference"]["actions"]}
        pred = {int(i): str(text) for i, text in event["prediction"]["jianzi_rows"]}
        notes = {int(note["index"]): note for note in example["runtime_item"]["input"]["notes_without_jianzi"]}
        cells = []
        for index in sorted(set(refs) | set(pred)):
            ref, generated = refs.get(index, ""), pred.get(index, "")
            style = "blank" if not canonical(ref) and not canonical(generated) else (
                "eq" if canonical(ref) == canonical(generated) else "neq")
            cells.append(f"<tr class='{style}'><th>{index}</th><td>{esc(notes.get(index, {}).get('jianpu'))}</td>"
                         f"<td>{esc(ref or '（空）')}</td><td>{esc(generated or '（空）')}</td></tr>")
        comparisons.append(
            f"<details><summary>{esc(sample_id)} · 提示词 {esc(str(prompt_sha)[:10])} · "
            f"{'协议成功' if event.get('ok') else '协议失败'}</summary><table>"
            "<tr><th>音序</th><th>简谱</th><th>真实标注</th><th>实际生成</th></tr>"
            + "".join(cells) + "</table></details>"
        )
    score_note = (
        "GEPA按短语精确率的算术平均选候选；下方正式汇总按格数加权，因此两者可能不同。"
        "本次旧检查点在最终汇总时另行执行过评估，记录在 selection.json 中；后续运行已改为复用同次评估日志。"
        if selection and not any(event.get("event") == "gepa_iteration_start" for event in events)
        else "GEPA按短语精确率的算术平均选候选；正式汇总按格数加权，使用已保存的同次评估结果。"
    )
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>师兄标准 GEPA 迭代日志</title><style>
:root{{--bg:#17191c;--surface:#24282d;--line:#46505b;--fg:#ece8df;--muted:#b9b4aa;--accent:#293844;--pass:#173b2a;--fail:#482526}}
*{{box-sizing:border-box}}body{{max-width:1400px;margin:24px auto;padding:0 18px;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif}}
h1,h2,h3{{font-weight:550}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}}.card,article,section{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:13px;margin:10px 0}}.card strong{{display:block;font-size:22px}}details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;overflow:hidden}}summary{{cursor:pointer;padding:8px 11px;background:var(--accent);font-weight:550}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#191c20;color:var(--fg);font:12px/1.5 ui-monospace,'Cascadia Code',monospace}}.muted{{color:var(--muted)}}.pass{{background:var(--pass);padding:8px}}.fail{{background:var(--fail);padding:8px}}code{{overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid var(--line);padding:7px;text-align:left}}
</style></head><body><h1>{esc(series + '系列 · 从师兄原版重新迭代' if series else '师兄标准 GEPA 自动迭代')}</h1>
<style>.eq{{background:#173b2a}}.neq{{background:#482526}}.blank{{color:#aaa;background:#292d31}}</style>
<div class='cards'><div class='card'>状态<strong>{esc(state)}</strong></div><div class='card'>曲谱<strong>{esc(split.get('score_key') or '—')}</strong></div><div class='card'>GEPA候选<strong>{len(candidates)}</strong></div><div class='card'>安全最优<strong>{esc(version(safe_idx) if safe_idx is not None else '—')}</strong></div></div>
<section><h2>数据边界与参数</h2><p class='fail'><b>注意：</b>{esc(split.get('warning') or '一曲谱运行只用于管线测试。')}</p>
{role_table}
<p><b>D_feedback：</b>{esc(', '.join(feedback_ids))}</p><p><b>D_pareto：</b>{esc(', '.join(pareto_ids))}</p>
<details><summary>全部参数</summary><pre>{esc(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre></details></section>
<section><h2>每一轮怎样优化</h2><p>从实例级Pareto候选池抽父代 → 从feedback集抽小批轨迹 → 执行Agent → 本地对照标注并生成差异反馈 → GLM结合轨迹和历史成败证据提出新提示词 → 同一小批比较 → 通过后评估全部Pareto集并更新候选池。</p><p>完整候选的部署选择另需三项安全门和独立confirmation；test50只在最终单独运行。每首前两段是快速筛选，不能代替完整曲谱验证。</p></section>
<section><h2>最终选择</h2><p>GEPA 按 D_pareto 平均逐格分选择 <b>{esc(version(best_idx) if best_idx is not None else '—')}</b>；
项目三指标安全门选择 <b>{esc(version(safe_idx) if safe_idx is not None else '—')}</b>。</p>
<p>已尝试提案：{len(iterations)}次。GEPA最佳来源：{esc(origin(best_idx))}；最终安全保留：{esc(origin(safe_idx))}。</p>
<p class='muted'>{esc(score_note)}</p>
<pre>{esc(json.dumps(selection, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>每轮采样、反馈与决策</h2>{''.join(iteration_blocks) or '<p class="muted">尚未开始API迭代。</p>'}</section>
<section><h2>候选演化与完整提示词</h2>{''.join(candidate_blocks) or '<p class="muted">尚无候选记录。</p>'}</section>
<section><h2>各候选实际生成与标注逐格比较</h2><p>绿色为师兄规范化后相同；红色为不同；灰色为两边均空。训练标注可用于反思监督；验证标注只用于评分和本地报告，不会自动反馈给反思模型。</p>{''.join(comparisons) or '<p class="muted">尚无实际rollout；运行后自动填充。</p>'}</section>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8", newline="\n")
    return {"output": str(output), "iterations": len(iterations), "candidates": len(candidates), "status": state}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "optimization_report.html"
    print(json.dumps(render(args.run_dir, output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
