#!/usr/bin/env python3
"""Render a self-contained V9/V27 prompt-iteration comparison report."""
from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def rate(value: object) -> str:
    return "—" if value is None else f"{float(value):.2%}"


def canonical(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = re.sub(r"[\[\]（）()]", "", text)
    for source, target in {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7"}.items():
        text = text.replace(source + "弦", target + "弦")
    return re.sub(r"\s+", "", text)


def mode(text: str) -> str:
    value = canonical(text)
    if "泛" in value:
        return "泛"
    if "散" in value:
        return "散"
    if "按" in value or "徽" in value:
        return "按"
    return "未明"


def strings(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"([1-7])弦", canonical(text)))


def difference_kind(reference: str, prediction: str) -> str:
    ref, pred = canonical(reference), canonical(prediction)
    if ref == pred:
        return "both_empty" if not ref else "exact"
    if ref and not pred:
        return "missing"
    if pred and not ref:
        return "extra"
    if mode(ref) != mode(pred):
        return "mode"
    if strings(ref) != strings(pred):
        return "string"
    if ("徽" in ref) != ("徽" in pred):
        return "hui"
    return "notation"


def aggregate(rows: list[dict], arms: list[str]) -> dict[str, dict]:
    result = {}
    for arm in arms:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        valid = [row for row in arm_rows if row.get("protocol_ok") and row.get("teacher_score")]
        totals = Counter()
        for row in valid:
            totals.update({key: int((row.get("teacher_score") or {}).get(key) or 0)
                           for key in ("rows_compared", "exact_cells", "reference_nonempty", "nonempty_hits")})
        result[arm] = {
            "n": len(arm_rows),
            "protocol_valid_rate": len(valid) / len(arm_rows) if arm_rows else None,
            "exact_cell_rate": totals["exact_cells"] / totals["rows_compared"] if totals["rows_compared"] else None,
            "nonempty_recall": totals["nonempty_hits"] / totals["reference_nonempty"] if totals["reference_nonempty"] else None,
        }
    return result


def render(result_path: Path, dataset_path: Path, output: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [row for row in result.get("rows", []) if isinstance(row, dict)]
    arms = list(result.get("arms") or ["v9", "v27"])
    sample_ids = list(result.get("sample_ids") or [])
    wanted = set(sample_ids)
    sources = {}
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("trajectory_id") or row.get("sample_id") or "")
            if sample_id in wanted:
                sources[sample_id] = row
    by_sample: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_sample.setdefault(str(row.get("sample_id")), {})[str(row.get("arm"))] = row

    summary = aggregate(rows, arms)
    expected = len(sample_ids) * len(arms)
    completed = len({(str(row.get("sample_id")), str(row.get("arm"))) for row in rows})
    top_rows = []
    for arm in arms:
        item = summary[arm]
        top_rows.append(
            f"<tr><th>{esc(arm)}</th><td>{item['n']}/{len(sample_ids)}</td>"
            f"<td>{rate(item['exact_cell_rate'])}</td><td>{rate(item['nonempty_recall'])}</td>"
            f"<td>{rate(item['protocol_valid_rate'])}</td></tr>"
        )

    error_totals = {arm: Counter() for arm in arms}
    blocks = []
    for position, sample_id in enumerate(sample_ids):
        records = by_sample.get(sample_id, {})
        source = sources.get(sample_id, {})
        notes = (source.get("input") or {}).get("notes_without_jianzi") or []
        note_map = {int(item["index"]): item for item in notes if item.get("index") is not None}
        ref_map = {
            int(item["source_index"]): str(item.get("text") or item.get("jianzi_text") or "")
            for item in ((source.get("reference_plan") or {}).get("actions") or [])
            if item.get("source_index") is not None
        }
        predictions = {
            arm: {
                int(item[0]): str(item[1])
                for item in (((records.get(arm) or {}).get("prediction") or {}).get("jianzi_rows") or [])
                if isinstance(item, list) and len(item) >= 2
            }
            for arm in arms
        }
        indices = sorted(set(note_map) | set(ref_map) | {i for mapping in predictions.values() for i in mapping})
        comparison_rows = []
        for index in indices:
            ref = ref_map.get(index, "")
            cells = []
            for arm in arms:
                if arm not in records:
                    cells.append("<td class='pending'>等待运行</td>")
                    continue
                pred = predictions[arm].get(index, "")
                kind = difference_kind(ref, pred)
                error_totals[arm][kind] += 1
                label = {
                    "exact": "一致", "both_empty": "均为空", "missing": "漏填", "extra": "多填",
                    "mode": "按散泛差异", "string": "弦序差异", "hui": "徽位差异", "notation": "文字差异",
                }[kind]
                cells.append(f"<td class='{esc(kind)}'><small>{esc(label)}</small>{esc(pred or '（空）')}</td>")
            note = note_map.get(index, {})
            comparison_rows.append(
                f"<tr><th>{esc(note.get('event_index', index))}</th><td>{esc(note.get('jianpu') or '')}</td>"
                f"<td class='annotation'>{esc(ref or '（空）')}</td>{''.join(cells)}</tr>"
            )
        cards = []
        for arm in arms:
            record = records.get(arm)
            if not record:
                cards.append(f"<tr><th>{esc(arm)}</th><td>等待运行</td><td>—</td><td>—</td><td>—</td></tr>")
                continue
            score = record.get("teacher_score") or {}
            cards.append(
                f"<tr><th>{esc(arm)}</th><td>{'成功' if record.get('protocol_ok') else '失败'}</td>"
                f"<td>{rate(score.get('exact_cell_rate'))}</td><td>{rate(score.get('nonempty_recall'))}</td>"
                f"<td>{esc(record.get('error') or '—')}</td></tr>"
            )
        blocks.append(
            f"<details class='phrase' data-search='{esc(sample_id)}'{' open' if position == 0 else ''}>"
            f"<summary><code>{esc(sample_id)}</code>｜完成 {len(records)}/{len(arms)}</summary>"
            f"<h3>本短语成绩</h3><div class='table-wrap'><table><thead><tr><th>版本</th><th>协议</th>"
            f"<th>逐格精确率</th><th>非空召回</th><th>错误</th></tr></thead><tbody>{''.join(cards)}</tbody></table></div>"
            f"<h3>模型输出与真实标注逐格对照</h3><div class='table-wrap'><table><thead><tr><th>音序</th><th>简谱</th>"
            f"<th>真实标注</th>{''.join('<th>' + esc(arm) + '</th>' for arm in arms)}</tr></thead>"
            f"<tbody>{''.join(comparison_rows)}</tbody></table></div></details>"
        )

    error_rows = []
    kinds = ("exact", "both_empty", "missing", "extra", "mode", "string", "hui", "notation")
    for arm in arms:
        error_rows.append(
            f"<tr><th>{esc(arm)}</th>" + "".join(f"<td>{error_totals[arm][kind]}</td>" for kind in kinds) + "</tr>"
        )

    prompts = result.get("prompts") or {}
    prompt_blocks = []
    for arm in arms:
        item = prompts.get(arm) or {}
        prompt_blocks.append(
            f"<details><summary>{esc(arm)}｜{esc(item.get('role') or '候选')}｜SHA-256 {esc(item.get('sha256') or '—')}</summary>"
            f"<p>来源：{esc(item.get('source') or '—')}</p><pre>{esc(item.get('text') or '（结果文件未保存提示词正文）')}</pre></details>"
        )
    generation = result.get("generation") or {}
    rationale = generation.get("rationale") or "本页为 V9 与 V27 双基线；尚未生成新候选。"
    next_action = generation.get("next_action") or "双基线完成后，依据训练标注归纳错误族，再提出单一、可检验的提示词修改。"
    parent = generation.get("parent")
    candidate = generation.get("candidate")
    hypothesis = generation.get("hypothesis") or "双基线测量"
    risk = generation.get("risk") or "—"
    promotion_gate = generation.get("promotion_gate") or "本页仅建立基线，不做晋级判定。"
    prompt_diff = ""
    decision = "双基线阶段：不选拔新版本。"
    if parent in prompts and candidate in prompts:
        before = str((prompts.get(parent) or {}).get("text") or "").splitlines()
        after = str((prompts.get(candidate) or {}).get("text") or "").splitlines()
        prompt_diff = "\n".join(difflib.unified_diff(
            before, after, fromfile=str(parent), tofile=str(candidate), lineterm=""
        ))
        if result.get("status") == "complete":
            parent_score, candidate_score = summary.get(parent, {}), summary.get(candidate, {})
            metric_names = ("protocol_valid_rate", "exact_cell_rate", "nonempty_recall")
            comparable = all(parent_score.get(key) is not None and candidate_score.get(key) is not None
                             for key in metric_names)
            safe = comparable and all(candidate_score[key] >= parent_score[key] for key in metric_names)
            improved = comparable and any(candidate_score[key] > parent_score[key] for key in metric_names)
            decision = (
                "训练小批三指标安全门通过：可进入 iteration_check，但尚不是正式胜者。"
                if safe and improved else
                "训练小批三指标安全门未通过：完整候选不晋级；若有局部错误类别改善，只保留规则片段。"
            )

    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{esc(' / '.join(arms))} 提示词迭代</title><style>
:root{{--bg:#181818;--fg:#e8e4dc;--line:#494949;--surface:#252525;--muted:#b9b3a8;--accent:#2f3a40;--same:#1e3a26;--diff:#40261e;--warn:#4a3d18;--empty:#303030}}
*{{box-sizing:border-box}}body{{max-width:1450px;margin:24px auto;padding:0 16px;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif}}h1,h2,h3{{font-weight:500}}p{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}}.card{{padding:12px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}}.card strong{{font-size:22px;display:block}}details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--surface);overflow:hidden}}summary{{cursor:pointer;padding:9px 12px;font-weight:500;background:var(--accent)}}.phrase>summary{{background:#2c2c2c}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#1d1d1d;color:var(--fg);font:12px/1.45 ui-monospace,'Cascadia Code',monospace}}.table-wrap{{overflow:auto;margin:8px 0 12px}}table{{border-collapse:collapse;width:100%;min-width:1050px;background:var(--surface)}}th,td{{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top;white-space:pre-wrap}}thead th{{background:#303030;position:sticky;top:0}}td small{{display:block;color:var(--muted)}}code{{font-family:ui-monospace,'Cascadia Code',monospace}}.exact{{background:var(--same)}}.both_empty{{background:var(--empty);color:var(--muted)}}.missing,.extra{{background:var(--warn)}}.mode,.string,.hui,.notation{{background:var(--diff)}}.annotation{{background:#343434}}.pending{{color:var(--muted)}}input{{width:100%;padding:10px;border:1px solid var(--line);border-radius:7px;background:#222;color:var(--fg);margin:8px 0}}
</style></head><body><h1>{esc(' / '.join(arms))} 提示词迭代对照</h1>
<div class='cards'><div class='card'>完成进度<strong>{completed}/{expected}</strong></div><div class='card'>阶段<strong>{esc(result.get('phase') or '—')}</strong></div><div class='card'>正式评分<strong>师兄三指标</strong></div><div class='card'>选择方式<strong>同题配对</strong></div></div>
<h2>本轮修改逻辑</h2><p><b>可检验假设：</b>{esc(hypothesis)}</p><p><b>证据与预期：</b>{esc(rationale)}</p><p><b>已知回归风险：</b>{esc(risk)}</p><p><b>晋级条件：</b>{esc(promotion_gate)}</p><p><b>当前判定：</b>{esc(decision)}</p><p><b>下一步：</b>{esc(next_action)}</p>
<h2>版本汇总</h2><div class='table-wrap'><table><thead><tr><th>版本</th><th>完成</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th></tr></thead><tbody>{''.join(top_rows)}</tbody></table></div>
<h2>错误类型汇总</h2><div class='table-wrap'><table><thead><tr><th>版本</th>{''.join('<th>'+esc(kind)+'</th>' for kind in kinds)}</tr></thead><tbody>{''.join(error_rows)}</tbody></table></div>
{("<h2>父代→候选的完整文本差异</h2><pre>" + esc(prompt_diff or "无文本差异") + "</pre>") if parent and candidate else ""}
<h2>提示词来源与正文</h2>{''.join(prompt_blocks)}
<h2>逐样本一一对应比较</h2><input id='search' placeholder='输入 sample_id 搜索'><div id='phrases'>{''.join(blocks)}</div>
<script>const q=document.getElementById('search');q.addEventListener('input',()=>{{const v=q.value.toLowerCase();document.querySelectorAll('.phrase').forEach(x=>x.style.display=x.dataset.search.toLowerCase().includes(v)?'':'none')}});</script>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8", newline="\n")
    return {"output": str(output), "completed": completed, "expected": expected, "samples": len(sample_ids)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.result, args.dataset, args.output), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
