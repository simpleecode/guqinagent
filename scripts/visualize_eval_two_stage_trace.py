#!/usr/bin/env python3
"""Render two-stage eval output with a final-vs-annotation table and expandable traces."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def pretty(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def details(title: str, value: object, *, opened: bool = False, css: str = "") -> str:
    flag = " open" if opened else ""
    return f'<details class="{esc(css)}"{flag}><summary>{esc(title)}</summary><pre>{esc(pretty(value))}</pre></details>'


def load_one(path: Path, sample_id: str) -> dict:
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("sample_id") == sample_id:
                return row
    raise SystemExit(f"sample_id not found: {sample_id}")


def parse_md_annotation(path: Path) -> dict[int, str]:
    """Read the final jianzi column from jianpu_jianzi_mapped.md."""
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or re.match(r"\|\s*-", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not re.fullmatch(r"\d+", cells[0]):
            continue
        text = re.sub(r"<br\s*/?>", "；", cells[5], flags=re.I).strip()
        result[int(cells[0])] = text
    return result


def action_map(stage: dict | None) -> dict[int, str]:
    if not stage:
        return {}
    actions = ((stage.get("plan") or {}).get("actions") or [])
    result: dict[int, str] = {}
    for action in actions:
        if isinstance(action, dict) and isinstance(action.get("source_index"), int):
            value = action.get("jianzi_text")
            result[action["source_index"]] = "" if value is None else str(value)
    return result


def surface(value: str | None) -> str:
    return "[空]" if value == "" else (f"[{value}]" if value is not None else "—")


def trace_html(stage: dict | None) -> str:
    if not stage:
        return "<p>无该阶段结果。</p>"
    out = []
    for item in stage.get("trace") or []:
        round_no = item.get("round", "?")
        suffix = "｜达到 max_new_tokens=10240，已截断" if item.get("truncated") else ""
        out.append(f'<section class="round"><h3>第 {esc(round_no)} 轮{esc(suffix)}</h3>')
        out.append(details("模型原始输出", item.get("raw_output", ""), opened=True))
        calls = item.get("tool_calls") or []
        tool_records = item.get("tool_results") or []
        # The trace stores name/arguments/result together for auditing.  The
        # next model round receives only each record's ``result`` as the tool
        # message content; avoid presenting the audit wrapper as model input.
        observations = [record.get("result") for record in tool_records]
        out.append(details("解析出的 tool calls（模型上一轮输出）", calls, opened=True))
        out.append(details(
            "模型下一轮实际收到的 tool observation（role=tool 的 content）",
            observations,
            opened=True,
        ))
        out.append("</section>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--annotation-md", type=Path, required=True)
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    pred = load_one(args.prediction, args.sample_id)
    source = load_one(args.input, args.sample_id)
    annotation = parse_md_annotation(args.annotation_md)
    base = pred.get("base") or {}
    guqinizer = pred.get("guqinizer") or {}
    base_rows, gq_rows = action_map(base), action_map(guqinizer)
    runtime = source.get("runtime_item") or {}
    notes = (runtime.get("input") or {}).get("notes_without_jianzi") or []
    if not notes:
        notes = (runtime.get("input") or {}).get("notes") or []
    note_map = {int(n["index"]): n for n in notes if isinstance(n, dict) and str(n.get("index", "")).isdigit()}
    # The mapped annotation file covers the whole score; this viewer renders
    # one eval phrase, so keep only annotation rows present in that phrase.
    annotation = {index: text for index, text in annotation.items() if index in note_map}
    indices = sorted(set(note_map) | set(annotation) | set(base_rows) | set(gq_rows))
    rows = []
    for index in indices:
        note = note_map.get(index, {})
        ref = annotation.get(index, "")
        b, g = base_rows.get(index), gq_rows.get(index)
        def cell(value: str | None) -> str:
            shown = surface(value)
            cls = "same" if value is not None and value == ref else "diff"
            return f'<td class="{cls}">{esc(shown)}</td>'
        rows.append(
            f'<tr><th>{index}</th><td>{esc(note.get("jianpu", ""))}</td>'
            f'<td>{esc(note.get("abc", ""))}</td>{cell(b)}{cell(g)}'
            f'<td>{esc(surface(ref))}</td></tr>'
        )
    public_prompt = source.get("public_prompt") or ""
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(args.sample_id)}｜两阶段轨迹</title>
<style>:root{{--bg:#f5f2eb;--fg:#28251f;--line:#d9d2c4;--surface:#fffdf8;--same:#dff2df;--diff:#fff0e8;--accent:#e8f1f4}}
*{{box-sizing:border-box}}body{{max-width:1250px;margin:24px auto;padding:0 16px;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
h1{{font-size:22px;margin-bottom:4px}}h2,h3{{font-weight:550;margin:16px 0 8px}}p{{color:#756d61}}details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--surface);overflow:hidden}}summary{{cursor:pointer;padding:8px 11px;background:var(--accent);font-weight:550}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:12px;background:#fff;font:12px/1.45 ui-monospace,"Cascadia Code",monospace}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:850px;background:var(--surface)}}th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}}thead th{{background:#eee9df;position:sticky;top:0}}tbody th{{font-weight:550}}td.same{{background:var(--same)}}td.diff{{background:var(--diff)}}.round{{border-top:1px solid var(--line);padding-top:8px;margin-top:12px}}.warning{{color:#a33b25;font-weight:600}}
</style></head><body><h1>{esc(args.sample_id)}｜两阶段评估轨迹</h1>
<p>顶部为 Base/Fingering、Guqinizer 最终结果与标注对比；绿色表示文字一致，浅红表示不同或模型没有提交该行。</p>
<h2>最终结果 vs 标注</h2><div class="table-wrap"><table><thead><tr><th>序号</th><th>简谱</th><th>ABC</th><th>Base/Fingering</th><th>Guqinizer</th><th>标注</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<details><summary>公开输入 prompt</summary><pre>{esc(public_prompt)}</pre></details>
<h2>Base/Fingering 轨迹</h2>{trace_html(base)}
<h2>Guqinizer 轨迹</h2>{trace_html(guqinizer)}
</body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8", newline="\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
