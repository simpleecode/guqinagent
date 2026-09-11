#!/usr/bin/env python3
"""Build a compact review page for Guqinizer no-op candidates."""
from __future__ import annotations

import html
import json
import re
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "ABC_J/agent_training/text_jianzi_guqinizer_selection_v2/selection_report.json"
INTERMEDIATES = ROOT / "ABC_J/agent_training/text_jianzi_guqinizer_selection_v2/fingering_intermediates.jsonl"
SOURCE = ROOT / "ABC_J/agent_training/inferred_v3/inferred_trajectories_train.jsonl"
OUTPUT = Path(
    r"C:\Users\30343\.codex\visualizations\2026\08\27\01a04324-2f64-76f1-9983-9acfa9c17711\no-op-jianzi-review.html"
)
SOURCE_REPORT = OUTPUT.with_name("no-op-source-links.md")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def surface(action: dict | None) -> str:
    if not action:
        return ""
    value = action.get("jianzi_text")
    if value is None:
        value = action.get("text")
    return "" if value is None else str(value)


def canonical(value: str) -> str:
    return re.sub(r"\s+", "", value).translate(str.maketrans("一二三四五六七", "1234567"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--intermediates", type=Path, default=INTERMEDIATES)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--fallback-source", type=Path,
                        help="optional inferred JSONL used for IDs absent from --source")
    parser.add_argument("--messages", type=Path,
                        help="messages_train.jsonl containing reasoned no-op assistant replies")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=0,
                        help="show only the first N trajectories (0 = all)")
    parser.add_argument("--stratified", action="store_true",
                        help="when limiting, prioritize non-empty annotation samples and distinct score keys")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    no_op_ids = report.get("no_op_ids")
    # Newer no-op generation reports only record the count; recover the
    # concrete phrase IDs from the generated public rows when available.
    if no_op_ids is None and args.messages:
        no_op_ids = [
            row.get("provenance", {}).get("source_trajectory_id")
            for row in read_jsonl(args.messages)
            if row.get("provenance", {}).get("source_trajectory_id")
        ]
    if not no_op_ids:
        raise ValueError("report has no_op_ids and --messages did not provide source IDs")
    intermediates = {row["trajectory_id"]: row for row in read_jsonl(args.intermediates)}
    references = {row["trajectory_id"]: row for row in read_jsonl(args.source)}
    if args.fallback_source:
        for row in read_jsonl(args.fallback_source):
            references.setdefault(row["trajectory_id"], row)
    if args.limit:
        if not args.stratified:
            no_op_ids = no_op_ids[:args.limit]
        else:
            def has_reference_text(trajectory_id: str) -> bool:
                return any(surface(action) for action in references[trajectory_id]["reference_plan"].get("actions", []))
            nonempty = [trajectory_id for trajectory_id in no_op_ids if has_reference_text(trajectory_id)]
            empty = [trajectory_id for trajectory_id in no_op_ids if not has_reference_text(trajectory_id)]
            ordered = nonempty + empty
            selected = []
            seen_keys = set()
            # First cover distinct score keys, then fill remaining slots by
            # annotation state. This avoids a prefix dominated by one all-empty score.
            for trajectory_id in ordered:
                key = references[trajectory_id].get("score_key")
                if key not in seen_keys:
                    selected.append(trajectory_id)
                    seen_keys.add(key)
                if len(selected) >= args.limit:
                    break
            if len(selected) < args.limit:
                selected.extend(trajectory_id for trajectory_id in ordered if trajectory_id not in selected)
            no_op_ids = selected[:args.limit]
    replies = {}
    if args.messages:
        for row in read_jsonl(args.messages):
            replies[row["provenance"]["source_trajectory_id"]] = row["messages"][-1].get("content", "")
    trajectories = []
    source_groups: dict[str, dict] = {}
    exact = empty = different = missing_final = rows_total = 0
    for trajectory_id in no_op_ids:
        final_actions = {
            int(a["source_index"]): a
            for a in intermediates[trajectory_id]["plan"].get("actions", [])
        }
        reference_actions = {
            int(a["source_index"]): a
            for a in references[trajectory_id]["reference_plan"].get("actions", [])
        }
        if not any(surface(action) for action in reference_actions.values()):
            source_item = references[trajectory_id]
            metadata = source_item.get("input", {}).get("metadata", {})
            key = source_item.get("score_key") or metadata.get("score_key") or "?"
            group = source_groups.setdefault(key, {
                "score_key": key,
                "title": metadata.get("score_title") or "",
                "score_id": metadata.get("score_id"),
                "phrases": 0,
            })
            group["phrases"] += 1
        note_map = {
            int(note["index"]): note
            for note in (references[trajectory_id].get("input", {}).get("notes_without_jianzi") or [])
            if note.get("index") is not None
        }
        both_empty = 0
        rows = []
        for index in sorted(set(final_actions) | set(reference_actions) | set(note_map)):
            final_text = surface(final_actions.get(index))
            reference_text = surface(reference_actions.get(index))
            if not final_text and not reference_text:
                status = "both_empty"
                both_empty += 1
            elif index not in final_actions or not final_text:
                status = "missing_final"
                missing_final += 1
            elif not reference_text:
                status = "annotation_empty"
                empty += 1
            elif canonical(final_text) == canonical(reference_text):
                status = "exact"
                exact += 1
            else:
                status = "different"
                different += 1
            rows_total += 1
            note = note_map.get(index, {})
            rows.append({
                "index": index,
                "jianpu": note.get("jianpu") or (reference_actions.get(index) or final_actions.get(index) or {}).get("jianpu", ""),
                "final": final_text,
                "reference": reference_text,
                "status": status,
            })
        empty += both_empty
        trajectories.append({"id": trajectory_id, "rows": rows,
                             "reasoning": replies.get(trajectory_id, "")})

    data = json.dumps(trajectories, ensure_ascii=False, separators=(",", ":"))
    source_rows = []
    for group in sorted(source_groups.values(), key=lambda item: (-item["phrases"], item["title"], item["score_key"])):
        key = group["score_key"]
        score_id = group.get("score_id")
        mapped_candidates = [
            ROOT / "ABC_J/final" / key / "jianpu_jianzi_mapped.md",
            ROOT / "ABC_J/round2/final" / key / "jianpu_jianzi_mapped.md",
        ]
        mapped_path = next((path for path in mapped_candidates if path.exists()), None)
        mapped_text = mapped_path.read_text(encoding="utf-8-sig", errors="replace") if mapped_path else ""
        app_match = re.search(r"^\|\s*url\s*\|\s*(https?://app\.sitongli\.net/[^|]+?)\s*\|", mapped_text, re.M)
        app_url = app_match.group(1).strip() if app_match else (
            f"https://app.sitongli.net/scores/{key}?shareid={score_id}" if score_id else ""
        )
        source_rows.append({
            **group,
            "gqs": str(ROOT / "ABC_J/agent_training/gqs" / key / "teacher.gqs"),
            "mapped": str(mapped_path) if mapped_path else "",
            "manifest": str(ROOT / "ABC_J/round2/final" / key / "manifest.json")
            if (ROOT / "ABC_J/round2/final" / key / "manifest.json").exists() else "",
            "url": f"https://s.sitongli.net/v2/scores/{score_id}/data" if score_id else "",
            "app_url": app_url,
        })
    source_data = json.dumps(source_rows, ensure_ascii=False, separators=(",", ":"))
    source_report = args.output.with_name("no-op-source-links.md")
    source_report.parent.mkdir(parents=True, exist_ok=True)
    report_lines = ["# 整段标注为空片段的源谱链接", "", "| 曲名 | score key | 全空片段 | mapped 文件 | app 链接 |", "|---|---|---:|---|---|"]
    report_lines.extend(
        f"| {s['title']} | `{s['score_key']}` | {s['phrases']} | `{s['mapped'] or '未找到'}` | {s['app_url']} |"
        for s in source_rows
    )
    source_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    title = "Guqinizer no-op：当前稿与复核 reasoning"
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#f7f8fa;--fg:#1f2937;--muted:#667085;--line:#d9dee7;--green:#e7f6ec;--green-fg:#146c35;--red:#fff0f0;--red-fg:#a12929;--gray:#f0f2f5;--blue:#315dbb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:24px 18px 48px}} h1{{font-size:22px;font-weight:600;margin:0 0 6px}} .sub{{color:var(--muted);margin:0 0 16px}}
.stats{{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0 18px}} .stat{{padding:10px 14px;border:1px solid var(--line);border-radius:8px;background:#fff;min-width:132px}} .stat b{{display:block;font-size:20px}} .stat span{{color:var(--muted);font-size:12px}}
.controls{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:0 0 14px}} label{{display:flex;gap:7px;align-items:center;color:var(--muted)}} input{{accent-color:var(--blue)}} #search{{padding:7px 9px;border:1px solid var(--line);border-radius:6px;min-width:260px;font:inherit}}
.trajectory{{border-top:1px solid var(--line);padding:12px 0}} .trajectory summary{{cursor:pointer;font-weight:600;display:flex;gap:12px;align-items:baseline}} .count{{font-weight:400;color:var(--muted);font-size:12px}} .reasoning{{margin-top:8px;padding:8px 10px;background:#fff;border-left:3px solid var(--blue);white-space:pre-wrap;word-break:break-word}} table{{width:100%;border-collapse:collapse;margin-top:10px;background:#fff}} th,td{{padding:6px 8px;border-bottom:1px solid #edf0f3;text-align:left;vertical-align:top}} th{{font-size:12px;color:var(--muted);font-weight:500}} td.idx{{width:72px;color:var(--muted)}} td.jianpu{{width:110px}} td.final,td.ref{{white-space:pre-wrap;word-break:break-word}} tr.exact td.final,tr.exact td.ref{{background:var(--green);color:var(--green-fg)}} tr.different td.final,tr.different td.ref{{background:var(--red);color:var(--red-fg)}} tr.annotation_empty td.final{{background:#fff7df;color:#805b00}} tr.both_empty td.final,tr.both_empty td.ref{{background:var(--gray);color:var(--muted)}} tr.missing_final td.final,tr.missing_final td.ref{{background:#fff7df;color:#805b00}} .sources{{border-top:1px solid var(--line);padding:10px 0 14px}} .sources summary{{cursor:pointer;font-weight:600}} .sources a{{color:var(--blue);word-break:break-all}} .legend{{color:var(--muted);font-size:12px;margin:0 0 12px}} .legend i{{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin:0 4px 0 10px;border:1px solid var(--line)}} .legend i:first-child{{margin-left:0;background:var(--green)}} .legend i:nth-child(2){{background:var(--red)}} .legend i:nth-child(3){{background:var(--gray)}} .legend i:nth-child(4){{background:#fff7df}} .hidden{{display:none}}
@media(max-width:640px){{main{{padding:16px 10px}} h1{{font-size:19px}} #search{{min-width:0;flex:1}} table{{font-size:13px}} th,td{{padding:5px}} td.jianpu{{width:70px}}}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="sub">绿色＝归一化后文字一致；红色＝双方都有文字但不同；灰色＝双方均为空；黄色＝只有一方有文字。</p>
<div class="stats"><div class="stat"><b>{len(trajectories)}</b><span>无操作轨迹</span></div><div class="stat"><b>{rows_total}</b><span>音符/行总数</span></div><div class="stat"><b>{exact}</b><span>文字一致</span></div><div class="stat"><b>{different}</b><span>文字仍有差异</span></div><div class="stat"><b>{empty}</b><span>标注为空</span></div><div class="stat"><b>{missing_final}</b><span>Fingering 未提交</span></div></div>
<details class="sources" open><summary>整段标注为空的来源曲谱索引（{len(source_rows)} 个 score key）</summary><table><thead><tr><th>曲名</th><th>score key</th><th>score id</th><th>全空片段</th><th>mapped 文件</th><th>app 链接</th></tr></thead><tbody>{''.join(f'<tr><td>{html.escape(str(s["title"]))}</td><td>{html.escape(str(s["score_key"]))}</td><td>{html.escape(str(s["score_id"] or ""))}</td><td>{s["phrases"]}</td><td><code>{html.escape(s["mapped"] or "未找到")}</code></td><td><a href="{html.escape(s["app_url"])}">{html.escape(s["app_url"])}</a></td></tr>' for s in source_rows)}</tbody></table></details>
<div class="controls"><input id="search" placeholder="搜索轨迹 ID 或序号"><label><input id="onlyDiff" type="checkbox">只显示红色差异</label><label><input id="openAll" type="checkbox">展开全部</label></div>
<p class="legend"><i></i>一致 <i></i>差异 <i></i>双方均为空 <i></i>单方有文字</p><section id="list"></section>
</main><script>
const data={data}; const list=document.getElementById('list'); const search=document.getElementById('search'); const onlyDiff=document.getElementById('onlyDiff'); const openAll=document.getElementById('openAll');
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function draw(){{const q=search.value.trim().toLowerCase(); const diff=onlyDiff.checked; list.innerHTML=''; data.forEach((t,ti)=>{{const rows=t.rows.filter(r=>(!diff||r.status==='different')&&(!q||(t.id+' '+r.index+' '+r.final+' '+r.reference).toLowerCase().includes(q))); if(!rows.length)return; const d=document.createElement('details'); d.className='trajectory'; d.open=openAll.checked||Boolean(q)||ti===0; const count=t.rows.filter(r=>r.status==='different').length; d.innerHTML='<summary>'+esc(t.id)+' <span class="count">'+t.rows.length+' 行 · '+count+' 行差异</span></summary><div class="reasoning"><b>no-op 最终复核 reasoning：</b><br>'+esc(t.reasoning||'[缺少 reasoning]')+'</div><table><thead><tr><th>序号</th><th>简谱</th><th>Fingering 最终减字</th><th>标注减字</th></tr></thead><tbody>'+rows.map(r=>'<tr class="'+r.status+'"><td class="idx">'+esc(r.index)+'</td><td class="jianpu">'+esc(r.jianpu)+'</td><td class="final">'+esc(r.final||'[空]')+'</td><td class="ref">'+esc(r.reference||'[空]')+'</td></tr>').join('')+'</tbody></table>'; list.appendChild(d)}})}}
search.addEventListener('input',draw); onlyDiff.addEventListener('change',draw); openAll.addEventListener('change',draw); draw();
</script></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "source_report": str(source_report), "trajectories": len(trajectories), "sources": len(source_rows), "rows": rows_total, "exact": exact, "different": different, "empty": empty, "missing_final": missing_final}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
