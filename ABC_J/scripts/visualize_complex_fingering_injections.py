#!/usr/bin/env python3
"""Create a standalone review page for phrase-level private knowledge injection."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ABC_J.scripts.generate_teacher_tool_trajectories import (  # noqa: E402
    COMPLEX_FINGERING_KNOWLEDGE_PATH,
    matched_compound_gesture_knowledge,
)


def action_text(action: dict) -> str:
    return str(action.get("jianzi_text") or action.get("text") or "")


def collect(path: Path) -> tuple[list[dict], Counter]:
    records: list[dict] = []
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            details = matched_compound_gesture_knowledge(item)
            actions = item.get("reference_plan", {}).get("actions", [])
            metadata = item.get("input", {}).get("metadata", {})
            names = [detail["name"] for detail in details]
            counts.update(names)
            records.append({
                "trajectory_id": item.get("trajectory_id", ""),
                "score_key": item.get("score_key", ""),
                "score_title": metadata.get("score_title", ""),
                "phrase_id": item.get("phrase_id", ""),
                "split": item.get("split", ""),
                "reference": [action_text(action) for action in actions if action_text(action)],
                "knowledge": details,
            })
    return records, counts


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guqinizer 私有指法知识注入审计</title>
<style>
:root{color-scheme:light dark;--bg:#f7f4ec;--fg:#27241f;--muted:#716b61;--surface:#fffdf8;--line:#d9d1c3;--accent:#8d3f2d;--soft:#f1e7d9;--none:#ece9e2} @media(prefers-color-scheme:dark){:root{--bg:#181714;--fg:#eee9df;--muted:#aaa397;--surface:#22201c;--line:#49443c;--accent:#e4967e;--soft:#332821;--none:#2b2924}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif} main{max-width:1500px;margin:auto;padding:24px} h1{font-size:24px;margin:0 0 4px} .sub{color:var(--muted);margin-bottom:18px}.stats,.controls{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.stat{background:var(--surface);border:1px solid var(--line);padding:9px 13px;border-radius:8px}.stat b{font-size:18px}.controls input,.controls select{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:7px;background:var(--surface);color:var(--fg)}.controls input{min-width:300px;flex:1}.score{border-top:1px solid var(--line);padding:10px 0}.score>summary{cursor:pointer;font-weight:600}.score-meta,.phrase-meta{color:var(--muted);font-size:13px}.phrase{margin:10px 0 14px 18px;padding-left:12px;border-left:3px solid var(--line)}.phrase.injected{border-left-color:var(--accent)}.phrase h3{font-size:15px;margin:0 0 5px}.reference{padding:8px 10px;background:var(--surface);border-radius:6px;overflow-wrap:anywhere}.knowledge{margin:7px 0 0;padding-left:20px}.knowledge li{margin:7px 0}.name{color:var(--accent);font-weight:600}.none{color:var(--muted);background:var(--none);display:inline-block;padding:3px 8px;border-radius:5px}.hidden{display:none!important} mark{background:var(--soft);color:inherit}.footer{color:var(--muted);margin-top:20px;font-size:13px}@media(max-width:600px){main{padding:14px}.controls input{min-width:100%}.phrase{margin-left:5px}}
</style></head><body><main>
<h1>Guqinizer 私有指法知识注入审计</h1><div class="sub" id="source"></div>
<div class="stats" id="stats"></div>
<div class="controls"><input id="query" aria-label="搜索" placeholder="搜索曲名、score key、phrase、减字或知识"><select id="mode" aria-label="注入状态"><option value="all">全部 phrase</option><option value="injected">仅有知识注入</option><option value="none">仅无知识注入</option></select><select id="split" aria-label="数据划分"><option value="all">全部 split</option><option>train</option><option>validation</option><option>test</option></select></div>
<div id="results" aria-live="polite"></div><div class="footer" id="footer"></div>
</main><script>
const DATA=__DATA__; const META=__META__;
const q=document.getElementById('query'),mode=document.getElementById('mode'),split=document.getElementById('split'),results=document.getElementById('results');
document.getElementById('source').textContent=`映射：${META.mapping}`;
document.getElementById('stats').innerHTML=`<div class="stat">总 phrase<br><b>${DATA.length}</b></div><div class="stat">有注入<br><b>${DATA.filter(x=>x.knowledge.length).length}</b></div><div class="stat">无注入<br><b>${DATA.filter(x=>!x.knowledge.length).length}</b></div><div class="stat">知识命中次数<br><b>${META.hits}</b></div><div class="stat">命中技法种类<br><b>${META.unique}</b></div>`;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const needle=q.value.trim().toLowerCase();const selected=DATA.filter(x=>{const has=x.knowledge.length>0;if(mode.value==='injected'&&!has||mode.value==='none'&&has)return false;if(split.value!=='all'&&x.split!==split.value)return false;const hay=[x.score_title,x.score_key,x.phrase_id,x.trajectory_id,...x.reference,...x.knowledge.flatMap(k=>[k.name,k.category,k.explanation])].join('\n').toLowerCase();return !needle||hay.includes(needle)});const groups=new Map();for(const x of selected){const key=x.score_key; if(!groups.has(key))groups.set(key,[]);groups.get(key).push(x)} results.innerHTML=[...groups].map(([key,rows])=>{const title=rows[0].score_title||'未命名';const injected=rows.filter(x=>x.knowledge.length).length;return `<details class="score" ${needle?'open':''}><summary>${esc(title)} · ${esc(key)} <span class="score-meta">${rows.length} 条，${injected} 条注入</span></summary>${rows.map(x=>`<section class="phrase ${x.knowledge.length?'injected':''}"><h3>${esc(x.phrase_id)} <span class="phrase-meta">${esc(x.trajectory_id)} · ${esc(x.split)}</span></h3><div class="reference">${x.reference.length?x.reference.map(esc).join('　'): '<span class="none">无减字标注</span>'}</div>${x.knowledge.length?`<ol class="knowledge">${x.knowledge.map(k=>`<li><span class="name">${esc(k.name)}</span> <span class="phrase-meta">${esc(k.category)}</span><br>${esc(k.explanation)}</li>`).join('')}</ol>`:'<div class="none">教师提示词不注入专有指法知识</div>'}</section>`).join('')}</details>`}).join('')||'<p class="none">没有匹配结果</p>';document.getElementById('footer').textContent=`当前显示 ${selected.length} / ${DATA.length} 条 phrase，按 score 分为 ${groups.size} 组。`;}
q.addEventListener('input',render);mode.addEventListener('change',render);split.addEventListener('change',render);render();
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "ABC_J/agent_training/inferred_gqs_v12_tuningfix_20260914/inferred_trajectories_all.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "ABC_J/agent_training/complex_fingering_injection_review_v3_20260909.html")
    args = parser.parse_args()
    records, counts = collect(args.input)
    meta = {
        "mapping": str(COMPLEX_FINGERING_KNOWLEDGE_PATH),
        "hits": sum(counts.values()),
        "unique": len(counts),
    }
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    metadata = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("__DATA__", payload).replace("__META__", metadata), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "phrases": len(records), "injected": sum(bool(r["knowledge"]) for r in records), **meta}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
