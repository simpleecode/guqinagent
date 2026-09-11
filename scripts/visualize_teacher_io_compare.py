#!/usr/bin/env python3
"""Render a compact two-level teacher I/O viewer with final-vs-annotation tables."""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path


MARKER = "无（由于是再作部分，省略）"


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pretty(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def block(title: str, value, *, opened: bool = False, css: str = "") -> str:
    flag = " open" if opened else ""
    return (
        f'<details class="{css}"{flag}><summary>{esc(title)}</summary>'
        f'<pre>{esc(pretty(value))}</pre></details>'
    )


def parse_annotation(text: str) -> dict[int, tuple[str, str]]:
    rows: dict[int, tuple[str, str]] = {}
    for line in (text or "").splitlines():
        match = re.match(r"音｜(\[.*\])$", line.strip())
        if not match:
            continue
        try:
            item = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(item, list) and len(item) >= 3 and isinstance(item[0], int):
            rows[item[0]] = (str(item[1]), "" if item[2] is None else str(item[2]))
    return rows


def align_annotation(rows: dict[int, tuple[str, str]], source_item: dict | None,
                     *, event_indexed: bool = False) -> dict[int, tuple[str, str]]:
    """Align continuous event indexes with legacy source indexes."""
    if not source_item:
        return rows
    notes = (source_item.get("input") or {}).get("notes_without_jianzi") or []
    by_source = {int(n["index"]): n for n in notes if n.get("index") is not None}
    by_event = {int(n["event_index"]): n for n in notes if n.get("event_index") is not None}
    aligned = {}
    for raw, value in rows.items():
        candidate = by_event.get(int(raw))
        if event_indexed and candidate is not None:
            target = int(candidate["index"])
        elif candidate is not None and str(candidate.get("jianpu") or "") == str(value[0] or ""):
            target = int(candidate["index"])
        elif int(raw) in by_source:
            target = int(raw)
        elif candidate is not None:
            target = int(candidate["index"])
        else:
            target = int(raw)
        aligned[target] = value
    return aligned


def accepted_rows(sample: dict) -> dict[int, str]:
    plan = sample.get("teacher_private", {}).get("accepted_plan") or {}
    actions = plan.get("actions") if isinstance(plan, dict) else []
    result: dict[int, str] = {}
    for action in actions or []:
        if isinstance(action, dict) and isinstance(action.get("source_index"), int):
            result[action["source_index"]] = "" if action.get("jianzi_text") is None else str(action.get("jianzi_text"))
    return result


def stage_name(sample: dict) -> str:
    objective = sample.get("teacher_private", {}).get("objective")
    return {"basic_fingering": "Base/Fingering", "toward_reference": "Guqinizer"}.get(str(objective), str(objective))


def render_io(sample: dict) -> str:
    out = []
    for item in sample.get("teacher_private", {}).get("teacher_io_trace") or []:
        request = item.get("request") or {}
        request_without_tools = {k: v for k, v in request.items() if k not in {"tools", "tool_choice"}}
        out.append(
            f'<section><h4>第 {esc(item.get("round"))} 轮</h4>'
            + block("真实请求：system + messages", request_without_tools, opened=True)
            + block("真实请求：tools", request.get("tools") or [], css="tools")
            + block("真实请求：tool_choice", request.get("tool_choice"))
            + block("教师原始响应", item.get("response"), opened=True, css="response")
            + "</section>"
        )
    return "".join(out)


def compare_table(base: dict | None, guqinizer: dict | None,
                  annotation: dict[int, tuple[str, str]], source_item: dict | None = None) -> str:
    base_rows = accepted_rows(base) if base else {}
    gq_rows = accepted_rows(guqinizer) if guqinizer else {}
    indices = sorted(set(annotation) | set(base_rows) | set(gq_rows))
    body = []
    note_by_index = {
        int(n["index"]): n for n in ((source_item or {}).get("input", {}).get("notes_without_jianzi") or [])
        if n.get("index") is not None
    }
    ordered = []
    last_marker = None
    for index, note in note_by_index.items():
        section = note.get("section") or {}
        marker = str(section.get("marker") or "")
        if marker and (section.get("start") or marker != last_marker):
            ordered.append(("section", marker))
        if marker:
            last_marker = marker
        is_bar = (str(note.get("jianpu") or "").strip() == "|"
                  or str(note.get("abc") or "").strip() == "|"
                  or str(note.get("duration") or "").strip() == "小节线")
        ordered.append(("bar", "") if is_bar else ("note", index))
    ordered.extend(("note", index) for index in indices if index not in note_by_index)
    for kind, value in ordered:
        if kind == "section":
            body.append(f'<tr class="structure section"><td colspan="5">段落标记｜{esc(value)}</td></tr>')
            continue
        if kind == "bar":
            body.append('<tr class="structure bar"><td colspan="5">小节线</td></tr>')
            continue
        index = value
        note = note_by_index.get(index) or {}
        visible_index = (int(note["event_index"])
                         if note.get("event_index") is not None else index)
        jianpu, target = annotation.get(index, ("", ""))
        base_text = base_rows.get(index, "")
        gq_text = gq_rows.get(index, "")
        def cell(value: str, reference: str) -> str:
            cls = "same" if value == reference else "diff"
            shown = "（空）" if value == "" else value
            return f'<td class="{cls}">{esc(shown)}</td>'
        body.append(
            f'<tr><th>{visible_index}</th><td>{esc(jianpu)}</td><td>{esc(target) if target else "（空）"}</td>'
            + cell(base_text, target) + cell(gq_text, target) + "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>序号</th><th>简谱</th>'
        '<th>标注</th><th>Base/Fingering 最终</th><th>Guqinizer 最终</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        help="inferred GQS source for index alignment and structure rows")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    source = {}
    if args.source:
        source = {row["trajectory_id"]: row for row in
                  (json.loads(line) for line in args.source.open(encoding="utf-8") if line.strip())}
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    annotations: dict[str, dict[int, tuple[str, str]]] = {}
    for sample in records:
        sample_id = str(sample.get("sample_id", "unknown"))
        key = re.sub(r"-(?:fingering_agent|guqinization)-teacher-tools$", "", sample_id)
        stage = stage_name(sample)
        grouped[key][stage] = sample
        annotation_gqs = str(sample.get("teacher_private", {}).get("annotation_gqs", "") or "")
        annotations[key] = align_annotation(
            parse_annotation(annotation_gqs), source.get(key),
            event_indexed="GQS｜annotation-phrase-1.2" in annotation_gqs,
        )

    score_groups: dict[str, list[tuple[str, dict[str, dict]]]] = defaultdict(list)
    for phrase_key, stages in grouped.items():
        score, _, phrase = phrase_key.partition("-")
        score_groups[score].append((phrase or phrase_key, stages))
    score_articles = []
    for score in sorted(score_groups):
        phrases = []
        for phrase, stages in sorted(score_groups[score], key=lambda x: x[0]):
            base = stages.get("Base/Fingering")
            gq = stages.get("Guqinizer")
            key = f"{score}-{phrase}"
            target_count = sum(1 for _, text in annotations.get(key, {}).values() if text == MARKER)
            status = "；含再作标记" if target_count else ""
            phrases.append(
                f'<details class="phrase"><summary>{esc(phrase)}｜Base：{"有" if base else "无"}｜Guqinizer：{"有" if gq else "无"}{esc(status)}</summary>'
            f'<h3>最终结果 vs 标注｜{esc(key)}</h3>{compare_table(base, gq, annotations.get(key, {}), source.get(key))}'
                + (f'<details class="io"><summary>教师 I/O｜Base/Fingering</summary>{render_io(base)}</details>' if base else "")
                + (f'<details class="io"><summary>教师 I/O｜Guqinizer</summary>{render_io(gq)}</details>' if gq else "")
                + "</details>"
            )
        score_articles.append(f'<details class="score"><summary>{esc(score)}｜{len(phrases)} 个 phrase</summary>{"".join(phrases)}</details>')

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>教师 I/O 两级对比</title><style>
:root{{color-scheme:light;--bg:#f5f2eb;--fg:#28251f;--line:#d9d2c4;--surface:#fffdf8;--muted:#756d61;--same:#dff2df;--diff:#fff0e8;--accent:#e8f1f4}}
*{{box-sizing:border-box}}body{{max-width:1180px;margin:24px auto;padding:0 16px;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
 h1{{font-size:23px;margin:0 0 8px}}h2,h3,h4{{font-weight:500;margin:14px 0 8px}}p{{color:var(--muted);margin:4px 0 12px}}details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--surface);overflow:hidden}}summary{{cursor:pointer;padding:9px 12px;font-weight:500;background:var(--accent)}}.score>summary{{font-size:17px}}.phrase{{margin:8px 12px}}.phrase>summary{{background:#f2eee5}}.io{{margin:10px 0}}.io>summary{{background:#f1eee8}}.table-wrap{{overflow:auto;margin:8px 0 12px}}table{{border-collapse:collapse;width:100%;min-width:700px;background:var(--surface)}}th,td{{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}}thead th{{font-weight:500;background:#eee9df;position:sticky;top:0}}tbody th{{font-weight:500;white-space:nowrap}}td.same{{background:var(--same)}}td.diff{{background:var(--diff)}}tr.structure td{{font-weight:600;text-align:center;color:var(--muted);background:#eef1f5}}tr.structure.section td{{color:#315dbb;background:#eaf0ff}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#fff;font:12px/1.45 ui-monospace,"Cascadia Code",monospace}}.tools>summary{{background:var(--accent)}}.response>summary{{background:#e9f2e7}}
</style></head><body><h1>教师输入／输出与最终结果对比</h1><p>两级展开：先展开曲谱，再展开 phrase。表格中的绿色格表示与标注文字完全一致；（空）表示空字符串。当前页面包含 {len(records)} 条已选成功轨迹。</p>{"".join(score_articles)}</body></html>'''
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(records), "scores": len(score_groups)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
