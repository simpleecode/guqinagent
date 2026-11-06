"""Prompt-experiment renderer using the running-eval viewer's visual language."""
from __future__ import annotations
import html, json, re, unicodedata
from pathlib import Path

def _e(x): return html.escape(str(x if x is not None else ""), quote=True)
def _rate(x): return "—" if x is None else f"{float(x):.2%}"
def _canon(x):
    value = unicodedata.normalize("NFKC", str(x or "")).strip()
    value = re.sub(r"[\[\]（）()]", "", value)
    for a, b in {"一":"1","二":"2","三":"3","四":"4","五":"5","六":"6","七":"7"}.items(): value=value.replace(a+"弦",b+"弦")
    return re.sub(r"\s+", "", value)

def render(result_path: Path, dataset_path: Path, output: Path) -> dict:
    result=json.loads(result_path.read_text(encoding="utf-8")); rows=[x for x in result.get("rows",[]) if isinstance(x,dict)]
    arms=list(result.get("arms",[])) or sorted({str(x.get("arm")) for x in rows}); wanted={str(x.get("sample_id")) for x in rows}
    sources={}
    for line in dataset_path.open(encoding="utf-8"):
        x=json.loads(line); sid=str(x.get("trajectory_id") or x.get("sample_id") or "")
        if sid in wanted: sources[sid]=x
    by={}
    for x in rows: by.setdefault(str(x.get("sample_id")),{})[str(x.get("arm"))]=x
    summary=result.get("summary") or {}; top=[]
    for arm in arms:
        s=summary.get(arm) or {}; sel=s.get("selection") or {}
        top.append(f"<tr><th>{_e(arm)}</th><td>{_e(s.get('n',0))}</td><td>{_rate(s.get('exact_cell_rate'))}</td><td>{_rate(s.get('nonempty_recall'))}</td><td>{_rate(s.get('protocol_valid_rate'))}</td><td>{'保留' if sel.get('retain_as_evidence') else '—'}</td></tr>")
    blocks=[]
    for pos,(sid,records) in enumerate(sorted(by.items())):
        source=sources.get(sid,{}) ; notes=(source.get("input") or {}).get("notes_without_jianzi") or []
        note={int(x["index"]):x for x in notes if x.get("index") is not None}
        ref={int(x["source_index"]):str(x.get("text") or x.get("jianzi_text") or "") for x in ((source.get("reference_plan") or {}).get("actions") or []) if x.get("source_index") is not None}
        pred={a:{int(c[0]):str(c[1]) for c in ((r.get("prediction") or {}).get("jianzi_rows") or []) if len(c)>=2} for a,r in records.items()}
        ids=sorted(set(note)|set(ref)|{i for v in pred.values() for i in v}); table=[]
        for i in ids:
            cells=[]
            for a in arms:
                v=pred.get(a,{}).get(i,""); cells.append(f"<td class={'same' if _canon(v)==_canon(ref.get(i,'')) else 'diff'}>{_e(v or '（空）')}</td>")
            n=note.get(i,{})
            table.append(f"<tr><th>{_e(n.get('event_index',i))}</th><td>{_e(n.get('jianpu') or '')}</td><td class='annotation'>{_e(ref.get(i,'') or '（空）')}</td>{''.join(cells)}</tr>")
        cards=[]
        for a in arms:
            r=records.get(a,{}) ; s=r.get("teacher_score") or {}
            cards.append(f"<tr><th>{_e(a)}</th><td>{'成功' if r.get('protocol_ok') else '失败'}</td><td>{_rate(s.get('exact_cell_rate'))}</td><td>{_rate(s.get('nonempty_recall'))}</td><td>{_e(r.get('rounds','—'))}</td></tr>")
        blocks.append(f"<details class='phrase'{' open' if pos==0 else ''}><summary><code>{_e(sid)}</code>｜提示词横向对比</summary><h3>本短语成绩</h3><div class='table-wrap'><table><thead><tr><th>版本</th><th>协议</th><th>逐格精确率</th><th>非空召回</th><th>轮数</th></tr></thead><tbody>{''.join(cards)}</tbody></table></div><h3>逐行减字对照</h3><div class='table-wrap'><table><thead><tr><th>音序</th><th>简谱</th><th>标注</th>{''.join('<th>'+_e(a)+'</th>' for a in arms)}</tr></thead><tbody>{''.join(table)}</tbody></table></div></details>")
    page=f"""<!doctype html><meta charset='utf-8'><title>提示词复测｜师兄评分</title><style>:root{{--bg:#181818;--fg:#e8e4dc;--line:#494949;--surface:#252525;--muted:#b9b3a8;--accent:#2f3a40;--same:#1e3a26;--diff:#40261e}}*{{box-sizing:border-box}}body{{max-width:1380px;margin:24px auto;padding:0 16px;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif}}h1,h2,h3{{font-weight:500}}p{{color:var(--muted)}}details{{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--surface);overflow:hidden}}summary{{cursor:pointer;padding:9px 12px;font-weight:500;background:var(--accent)}}.phrase>summary{{background:#2c2c2c}}.table-wrap{{overflow:auto;margin:8px 0 12px}}table{{border-collapse:collapse;width:100%;min-width:1080px;background:var(--surface)}}th,td{{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:top;white-space:pre-wrap}}thead th{{background:#303030;position:sticky;top:0}}code{{font-family:ui-monospace,monospace}}.same{{background:var(--same)}}.diff{{background:var(--diff)}}.annotation{{background:#343434}}</style><body><h1>提示词复测｜师兄逐格评分标准</h1><p>沿用运行中评测查看器的布局：上方为版本汇总，下方按短语折叠展开，绿色表示与标注规范化后完全一致。</p><h2>版本汇总</h2><div class='table-wrap'><table><thead><tr><th>提示词版本</th><th>完成样本</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th><th>Pareto</th></tr></thead><tbody>{''.join(top)}</tbody></table></div><h2>逐短语对照</h2>{''.join(blocks)}</body>"""
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(page,encoding="utf-8")
    return {"output":str(output),"arms":len(arms),"samples":len(by)}
