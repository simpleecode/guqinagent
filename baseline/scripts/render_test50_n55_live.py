#!/usr/bin/env python3
"""Read-only live HTML dashboard for an in-progress N55 test50 run."""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "guqinagent"
sys.path[:0] = [str(WORKSPACE / ".gepa_packages"), str(REPO), str(REPO / "baseline/gepa/src")]

from baseline.gepa_guqin.compat import materialize_actions
from baseline.gepa_guqin.metric import replay_events
from baseline.scripts.prompt_data_sources import TEST50
from train.scripts.score_agent_predictions import canonical

OUT = REPO / "baseline/runs/N_gepa16_from_original/test50_top3"
STRUCTURED_SUMMARY = OUT / "structured_metrics/N55_structured_partial/summary.json"
STRUCTURED_EVENTS = OUT / "structured_metrics/N55_structured_partial/per_event.jsonl"


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sources():
    public = read_jsonl(TEST50 / "test_runtime.jsonl")
    sealed = {row["sample_id"]: row for row in read_jsonl(TEST50 / "evaluation_pairs_test.jsonl")}
    examples, historical = {}, {}
    for public_row in public:
        sample_id = public_row["sample_id"]
        item = deepcopy(public_row["runtime_item"])
        item["input"].setdefault("phrase_handoff", {}).pop("previous_phrase", None)
        reference = sealed[sample_id].get("reference", {"actions": []})
        example = {"sample_id": sample_id, "score_key": public_row["score_key"],
                   "phrase_id": public_row["phrase_id"], "runtime_item": item, "reference": reference}
        examples[sample_id] = example
        historical[(example["score_key"], example["phrase_id"])] = {
            "trajectory_id": sample_id, "score_key": example["score_key"], "phrase_id": example["phrase_id"],
            "input": item["input"], "baseline_plan": item.get("baseline_plan") or {},
            "reference_plan": {"actions": materialize_actions(reference["actions"])},
        }
    return examples, historical


def pitch_map(example, prediction, historical):
    try:
        events, _, _ = replay_events(example, prediction, historical)
    except Exception as exc:
        return {}, f"重放失败：{type(exc).__name__}"
    output = {}
    for event in events:
        audit = event.get("pitch_audit") or {}
        covered = audit.get("covered_indices") or ([audit["index"]] if audit.get("index") is not None else [])
        pairs = audit.get("pairs") or []
        detail = "、".join(
            f"{item.get('jianpu_name', '?')} / {item.get('jianzi_name', '?')}（偏差{float(item.get('delta_cents', 0)):.0f}音分）"
            for item in pairs
        ) or str(audit.get("reason") or "无独立可核音高")
        for index in covered:
            output[int(index)] = (str(audit.get("status") or "not_applicable"), detail)
    return output, None


def structured_pitch_maps() -> dict[tuple[str, str], dict[int, tuple[str, str]]]:
    """Reuse the completed evaluator's event-level pitch audit for exports.

    It avoids replaying every phrase a second time while retaining exactly the
    same ±50-cent pitch decision recorded in the final structured evaluation.
    """
    result: dict[tuple[str, str], dict[int, tuple[str, str]]] = defaultdict(dict)
    if not STRUCTURED_EVENTS.exists():
        return result
    try:
        with STRUCTURED_EVENTS.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                audit = event.get("pitch_audit") or {}
                index = audit.get("index")
                if index is None:
                    continue
                pairs = audit.get("pairs") or []
                detail = "、".join(
                    f"{item.get('jianpu_name', '?')} / {item.get('jianzi_name', '?')}"
                    f"（偏差{float(item.get('delta_cents', 0)):.0f}音分）"
                    for item in pairs
                ) or str(audit.get("reason") or "无独立可核音高")
                key = (str(event.get("piece_id") or ""), str(event.get("phrase_id") or ""))
                result[key][int(index)] = (str(audit.get("status") or "not_applicable"), detail)
    except (OSError, json.JSONDecodeError):
        return defaultdict(dict)
    return result


def fmt_unit(value):
    return "—" if value is None else f"{float(value):.4f}"


def structured_metrics_section(current_rows: int) -> str:
    """Show paper-facing names without pretending unlike metrics are equal."""
    summary = read_json(STRUCTURED_SUMMARY, {})
    metrics = summary.get("metrics") or {}
    snapshot_count = int(summary.get("prediction_samples") or 0)
    incomplete = summary.get("tone_type_replay_incomplete_scores") or []
    if not metrics:
        return ("<section><h2>师兄指标（结构化音乐评测）</h2>"
                "<p>尚未生成结构化评测快照；这不会影响正在进行的 GLM 推理。</p></section>")
    technique = metrics.get("performance_technique_usage") or {}
    ornament = metrics.get("ornament") or {}
    tone = metrics.get("tone_type") or {}
    string = metrics.get("string") or {}
    string_hui = metrics.get("string_hui") or {}
    nonbasic = metrics.get("nonbasic_technique_rate") or {}
    cards = [
        ("Pitch Acc. ↑", pct(metrics.get("pitch_accuracy_at_50_cents")),
         "音高准确率：可核对音高在 ±50 音分内与简谱目标一致的比例。"),
        ("String-Hui F1 ↑", pct(string_hui.get("f1")),
         "弦-徽组合 F1：弦号与徽位必须同时正确才命中；"
         f"精确率 {pct(string_hui.get('precision'))}，召回率 {pct(string_hui.get('recall'))}。"),
        ("Tech. Sim. ↑", fmt_unit(technique.get("distribution_similarity")),
         "技法相似度：生成与标注的技法使用分布相似程度，满分为 1。"),
        ("Orn. F1 ↑", pct(ornament.get("f1")),
         "装饰音 F1：吟、猱、绰、注等装饰音与标注的一致性，越高越好。"),
        ("Nonbasic Rate ↑", pct(nonbasic.get("prediction_rate")),
         "非基础技法比例：目标音中含非基础技法且没有最终规则警告的比例；越高表示技法更丰富。"),
        ("Tone Sim. ↑", fmt_unit(tone.get("distribution_similarity")),
         "音色相似度：按音、散音、泛音三类的总体分布相似程度，满分为 1。"),
        ("Rule Violation Rate ↓", pct(metrics.get("rule_violation_rate")),
         "规则违规率：音高不符或无法可靠解析/演奏的事件比例，越低越好。"),
        ("String Acc.（补充）", pct(string.get("accuracy")),
         "弦准确率：只比较主弦号；它是补充诊断，不等同于 String-Hui F1。"),
    ]
    cards_html = "".join(
        f"<article class='metric'><h3>{html.escape(name)}</h3><strong>{html.escape(value)}</strong>"
        f"<p>{html.escape(note)}</p></article>" for name, value, note in cards
    )
    warning = ""
    warning_incomplete = summary.get("warning_replay_incomplete_scores") or []
    if incomplete:
        warning = (f"当前快照有 {len(incomplete)} 首曲谱尚未跑完；涉及跨句状态的 Tone Sim. "
                   "只基于已完整重放的曲谱，test50 完成后会重算。")
    if warning_incomplete:
        warning += (" " if warning else "") + (
            f"有 {len(warning_incomplete)} 首曲谱的最终规则重放不完整；"
            "Rule Violation Rate 仅按可完整审计的乐句计算。")
    return f"""<section><h2>师兄指标（结构化音乐评测）</h2>
<p>这是对已保存生成结果的只读复算，不会调用 GLM，也不会把封存标注传给模型。结构化快照覆盖 <b>{snapshot_count}/{current_rows}</b> 句。</p>
<div class='metrics'>{cards_html}</div><p>{html.escape(warning)}</p></section>"""


def generated_annotation_summary(rows, examples) -> str:
    """Human-readable summary of what N55 actually wrote, not just its score."""
    counts = Counter()
    generated_texts = Counter()
    predicted_nonempty = reference_nonempty = 0
    for row in rows:
        example = examples.get(str(row.get("sample_id")))
        if not example:
            continue
        reference = {int(item["source_index"]): str(item.get("jianzi_text") or item.get("text") or "")
                     for item in example["reference"].get("actions") or []}
        prediction = {int(index): str(text or "")
                      for index, text in (row.get("prediction") or {}).get("jianzi_rows") or []}
        for text in prediction.values():
            if canonical(text):
                predicted_nonempty += 1
                generated_texts[text] += 1
        for text in reference.values():
            if canonical(text):
                reference_nonempty += 1
        for index in set(reference) | set(prediction):
            ref, pred = reference.get(index, ""), prediction.get(index, "")
            if canonical(ref) == canonical(pred):
                counts["完全一致"] += 1
            elif ref and not pred:
                counts["漏写"] += 1
            elif pred and not ref:
                counts["多写"] += 1
            else:
                counts["文字不同"] += 1
    top_rows = "".join(
        f"<tr><td>{html.escape(text)}</td><td>{number}</td></tr>"
        for text, number in generated_texts.most_common(12)
    ) or "<tr><td colspan='2'>尚无生成结果</td></tr>"
    return f"""<section><h2>N55 生成标注总结</h2>
<p>这里总结的是 N55 实际写出的减字，不是模型的思考过程。文字不同不一定表示音高错误，音高是否正确请看下方的 ✓/✗ 核对。</p>
<div class='metrics'>
<article class='metric'><h3>生成非空减字格</h3><strong>{predicted_nonempty:,}</strong><p>N55 实际写入内容的位置数。</p></article>
<article class='metric'><h3>标注非空减字格</h3><strong>{reference_nonempty:,}</strong><p>真实标注中需要写入内容的位置数。</p></article>
<article class='metric'><h3>完全一致</h3><strong>{counts['完全一致']:,}</strong><p>生成文字与标注文字完全相同的格。</p></article>
<article class='metric'><h3>文字不同</h3><strong>{counts['文字不同']:,}</strong><p>两边都有内容，但减字写法不同。</p></article>
<article class='metric'><h3>多写</h3><strong>{counts['多写']:,}</strong><p>标注为空，但 N55 写了减字。</p></article>
<article class='metric'><h3>漏写</h3><strong>{counts['漏写']:,}</strong><p>标注有减字，但 N55 没有写。</p></article>
</div><h3>最常见的 12 种 N55 生成减字</h3>
<table><tr><th>生成减字</th><th>出现次数</th></tr>{top_rows}</table></section>"""


def _detail_order(row, examples):
    example = examples.get(str(row.get("sample_id"))) or {}
    phrase_id = str(example.get("phrase_id") or row.get("phrase_id") or "")
    try:
        phrase_number = int(phrase_id.removeprefix("p"))
    except ValueError:
        phrase_number = 10**9
    return (str(example.get("score_key") or row.get("score_key") or ""), phrase_number, phrase_id)


def render(*, output_path: Path | None = None, auto_refresh: bool = True,
           all_details: bool = False):
    result = read_json(OUT / "rows_N55.json", {"rows": [], "configuration": {"sample_count": 1031}, "status": "missing"})
    rows = result.get("rows") or []
    examples, historical = sources()
    saved_pitches = structured_pitch_maps() if all_details else {}
    totals, by_score, expected = Counter(), defaultdict(Counter), Counter(e["score_key"] for e in examples.values())
    for row in rows:
        score = row.get("teacher_score") or {}
        score_key = row.get("score_key") or examples.get(row.get("sample_id"), {}).get("score_key") or "未知曲谱"
        for target in (totals, by_score[score_key]):
            target["samples"] += 1
            target["protocol"] += int(bool(row.get("protocol_ok")))
            for key in ("rows_compared", "exact_cells", "reference_nonempty", "nonempty_hits"):
                target[key] += int(score.get(key) or 0)
    def rate(value, top, bottom):
        return value[top] / value[bottom] if value[bottom] else None
    score_rows = []
    for score_key in sorted(expected):
        value = by_score[score_key]
        score_rows.append(f"<tr><td>{html.escape(score_key)}</td><td>{value['samples']}/{expected[score_key]}</td>"
                          f"<td>{pct(rate(value, 'exact_cells', 'rows_compared'))}</td>"
                          f"<td>{pct(rate(value, 'nonempty_hits', 'reference_nonempty'))}</td>"
                          f"<td>{pct(rate(value, 'protocol', 'samples'))}</td></tr>")
    # Serial runs use rollout_events_N55.jsonl; score-level parallel runs use
    # one file per worker.  Read the last event of every file so the page shows
    # the current state of every concurrent score without concurrent appends.
    active = []
    for event_path in sorted(OUT.glob("rollout_events_N55*.jsonl")):
        try:
            lines = [line for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                active.append(json.loads(lines[-1]))
        except (OSError, json.JSONDecodeError):
            continue
    active.sort(key=lambda event: str(event.get("logged_at") or ""), reverse=True)
    if active:
        current = "；".join(
            f"{event.get('sample_id', '未知样本')} · 第{event.get('round', '?')}轮 · "
            f"{', '.join(event.get('tool_calls') or []) or '整理结果'}"
            for event in active[:3]
        )
    else:
        current = "等待首个调用"
    details = []
    detail_rows = (sorted(rows, key=lambda row: _detail_order(row, examples))
                   if all_details else reversed(rows[-8:]))
    for row in detail_rows:
        example = examples.get(row["sample_id"])
        if not example:
            continue
        reference = {int(a["source_index"]): str(a.get("jianzi_text") or a.get("text") or "")
                     for a in example["reference"]["actions"]}
        prediction = row.get("prediction") or {"jianzi_rows": []}
        generated = {int(i): str(text) for i, text in prediction.get("jianzi_rows") or []}
        pitches = saved_pitches.get((str(example["score_key"]), str(example["phrase_id"])))
        if pitches is None:
            pitches, problem = pitch_map(example, prediction, historical)
        else:
            problem = None
        cells = []
        for index in sorted(set(reference) | set(generated) | set(pitches)):
            ref, pred = reference.get(index, ""), generated.get(index, "")
            if canonical(ref) == canonical(pred): kind, difference = "exact", "一致"
            elif ref and not pred: kind, difference = "missing", "漏填"
            elif pred and not ref: kind, difference = "extra", "多填"
            else: kind, difference = "different", "文字不同"
            state, detail = pitches.get(index, ("not_applicable", "无独立可核音高"))
            mark = "✓" if state == "matched" else "✗" if state == "mismatched" else "—"
            cells.append(f"<tr class='{kind}'><td>{index}</td><td>{html.escape(ref or '（空）')}</td>"
                         f"<td>{html.escape(pred or '（空）')}</td><td>{difference}</td>"
                         f"<td title='{html.escape(detail, quote=True)}'>{mark} {html.escape(detail)}</td></tr>")
        metric = row.get("teacher_score") or {}
        details.append(f"<details><summary>{html.escape(row['sample_id'])}｜精确{pct(metric.get('exact_cell_rate'))}｜召回{pct(metric.get('nonempty_recall'))}</summary>"
                       f"<table><tr><th>音序</th><th>封存标注</th><th>N55生成</th><th>差异</th><th>音高校对（±50音分）</th></tr>{''.join(cells)}</table>"
                       f"<p>{html.escape(problem or '')}</p></details>")
    total = int((result.get("configuration") or {}).get("sample_count") or 1031)
    refresh_tag = "<meta http-equiv='refresh' content='15'>" if auto_refresh else ""
    title = "N55 test50 实时进度" if auto_refresh else "N55 test50 稳定快照"
    snapshot_note = ("" if auto_refresh else
                     "<p><b>稳定快照：</b>此文件不会自动刷新；分数和逐句差异固定为导出时已保存的结果。</p>")
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
{refresh_tag}<title>{title}</title><style>
body{{font:15px/1.65 system-ui,'Microsoft YaHei';max-width:1300px;margin:28px auto;padding:0 18px;background:#f4f6fa;color:#263146}}
section,details{{background:#fff;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:14px 0}}summary{{font-weight:700;cursor:pointer}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #e1e6ee;text-align:left;vertical-align:top}}.exact{{background:#eaf8ef}}.missing{{background:#fff0f0}}.extra{{background:#fff7e5}}.different{{background:#f5f0ff}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}.metric{{background:#f8fafc;border:1px solid #dce3ed;border-radius:9px;padding:12px}}.metric h3{{margin:0;color:#384862;font-size:15px}}.metric strong{{display:block;font-size:23px;margin:5px 0;color:#12624d}}.metric p{{margin:0;font-size:13px;color:#5b6575}}</style>
<h1>N55 · test50 封存集实时评测</h1><section><h2>已完成 {len(rows)}/{total}</h2>
<p>当前调用：{html.escape(current)}</p><p>已完成部分累计：逐格精确率 <b>{pct(rate(totals, 'exact_cells', 'rows_compared'))}</b>；非空召回 <b>{pct(rate(totals, 'nonempty_hits', 'reference_nonempty'))}</b>；协议合规 <b>{pct(rate(totals, 'protocol', 'samples'))}</b>。</p>
<p>{'本页每15秒刷新；' if auto_refresh else ''}仅展示已写入断点的结果。曲谱间可并行，同一曲谱的乐句始终串行，保证前句传递状态不被打乱。封存标注只用于最终评测，不会回流到提示词训练。</p>{snapshot_note}</section>
{structured_metrics_section(len(rows))}
{generated_annotation_summary(rows, examples)}
<section><h2>按曲谱累计评分</h2><table><tr><th>曲谱</th><th>完成</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th></tr>{''.join(score_rows)}</table></section>
<section><h2>{'全部已完成乐句' if all_details else '最近完成的8句'}：标注、生成差异与音高</h2><p>绿色一致；红色漏填；黄色多填；紫色文字不同。音高 ✓/✗ 根据工具重放、±50音分容差计算。{'最终稳定版按曲谱与乐句顺序收录全部明细。' if all_details else ''}</p>{''.join(details) or '<p>尚无已保存结果。</p>'}</section></html>"""
    target = output_path or (OUT / "n55_test50_live.html")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(page, encoding="utf-8", newline="\n")
    temporary.replace(target)


def pct(value):
    return "—" if value is None else f"{100 * float(value):.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None,
                        help="write to a custom HTML file instead of the live dashboard")
    parser.add_argument("--static", action="store_true",
                        help="disable automatic browser refresh; use for a stable exported snapshot")
    parser.add_argument("--all-details", action="store_true",
                        help="include every saved phrase's label/prediction/pitch detail (for final exports)")
    args = parser.parse_args()
    while True:
        render(output_path=args.output, auto_refresh=not args.static, all_details=args.all_details)
        if not args.watch:
            return 0
        time.sleep(max(5, args.seconds))


if __name__ == "__main__":
    main()
