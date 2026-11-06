"""Generate a complete, offline prompt-evolution audit from a GEPA run.

The report reconstructs every proposal preserved in the text log, including
accepted, rejected, and interrupted attempts. It never invents hidden model
reasoning: the displayed decision basis comes from recorded scores and logs.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def score(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, (int, float)) else "未记录"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def mean(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def parse_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def load_text_log(run_dir: Path) -> tuple[str, str]:
    for name in ("run_log.txt", "gepa_run_log.txt"):
        path = run_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace"), name
    return "", ""


def parse_proposals(log_text: str, prompt_key: str) -> list[dict[str, Any]]:
    header = re.compile(
        rf"^Iteration\s+(\d+):\s+Proposed new text for\s+{re.escape(prompt_key)}:\s*",
        re.MULTILINE,
    )
    matches = list(header.finditer(log_text))
    attempts: list[dict[str, Any]] = []
    for occurrence, match in enumerate(matches, 1):
        boundary = matches[occurrence].start() if occurrence < len(matches) else len(log_text)
        raw_block = log_text[match.end():boundary]
        # A resume marker closes an orphaned attempt even if the same iteration is retried.
        resume_at = raw_block.find("\nLoading gepa state")
        block = raw_block[:resume_at] if resume_at >= 0 else raw_block
        prompt_end = re.search(r"^Iteration\s+\d+:\s+", block, re.MULTILINE)
        prompt = block[:prompt_end.start() if prompt_end else len(block)].strip()
        decision_text = block[prompt_end.start():] if prompt_end else ""

        before = log_text[:match.start()]
        selected = list(re.finditer(r"^Iteration\s+\d+: Selected program\s+(\d+)\s+score:\s+([-+0-9.eE]+)", before, re.MULTILINE))
        parent_idx = int(selected[-1].group(1)) if selected else 0
        parent_score = float(selected[-1].group(2)) if selected else None

        accepted = re.search(
            r"Accepted candidate \(subsample score\s+([-+0-9.eE]+)\s+->\s+([-+0-9.eE]+)\)",
            decision_text,
        )
        rejected = re.search(
            r"New subsample score\s+([-+0-9.eE]+)\s+is not better than old score\s+([-+0-9.eE]+),\s+skipping",
            decision_text,
        )
        candidate_match = re.search(r"New program candidate index:\s+(\d+)", decision_text)
        val_score = parse_float(decision_text, r"Valset score for new program:\s*([-+0-9.eE]+)")
        if accepted:
            status, old_total, new_total = "accepted", float(accepted.group(1)), float(accepted.group(2))
            basis = f"抽样总分由 {old_total:.6f} 提升到 {new_total:.6f}，因此进入完整验证。"
        elif rejected:
            status, new_total, old_total = "rejected", float(rejected.group(1)), float(rejected.group(2))
            basis = f"抽样总分由 {old_total:.6f} 降到 {new_total:.6f}，日志记录为 skipping。"
        else:
            status, old_total, new_total = "interrupted", None, None
            basis = "日志保存了提案文本，但没有保存接受或拒绝结果；通常表示运行在该提案后中断或恢复。"
        attempts.append({
            "occurrence": occurrence,
            "iteration": int(match.group(1)),
            "parent_idx": parent_idx,
            "parent_score": parent_score,
            "prompt": prompt,
            "status": status,
            "old_total": old_total,
            "new_total": new_total,
            "candidate_idx": int(candidate_match.group(1)) if candidate_match else None,
            "val_score": val_score,
            "basis": basis,
        })
    return attempts


def diff_stats(old: str, new: str) -> dict[str, int]:
    matcher = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines())
    stats = {"added": 0, "deleted": 0, "replaced_old": 0, "replaced_new": 0, "unchanged": 0}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            stats["unchanged"] += i2 - i1
        elif tag == "insert":
            stats["added"] += j2 - j1
        elif tag == "delete":
            stats["deleted"] += i2 - i1
        else:
            stats["replaced_old"] += i2 - i1
            stats["replaced_new"] += j2 - j1
    return stats


def unified_diff(old: str, new: str, old_name: str, new_name: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), fromfile=old_name, tofile=new_name,
        lineterm="", n=3,
    )
    rendered = []
    for line in lines:
        kind = "same"
        if line.startswith("+") and not line.startswith("+++"):
            kind = "add"
        elif line.startswith("-") and not line.startswith("---"):
            kind = "del"
        elif line.startswith("@@"):
            kind = "hunk"
        rendered.append(f'<span class="{kind}">{esc(line)}</span>')
    return "\n".join(rendered) or '<span class="same">提示词文本没有变化</span>'


def render_score_table(old_scores: list[Any], new_scores: list[Any], sample_ids: list[Any]) -> str:
    count = max(len(old_scores), len(new_scores), len(sample_ids))
    if not count:
        return "<p>本轮没有结构化的逐样本分数记录。</p>"
    rows = []
    for index in range(count):
        old = old_scores[index] if index < len(old_scores) else None
        new = new_scores[index] if index < len(new_scores) else None
        delta = float(new) - float(old) if isinstance(old, (int, float)) and isinstance(new, (int, float)) else None
        sample = sample_ids[index] if index < len(sample_ids) else index
        rows.append(f"<tr><td>{esc(sample)}</td><td>{score(old)}</td><td>{score(new)}</td><td>{score(delta)}</td></tr>")
    return ("<table><thead><tr><th>抽样索引</th><th>父版本</th><th>提案</th><th>变化</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def explain_changes(old: str, new: str) -> list[tuple[str, str]]:
    """Turn added prompt text into a small, plain-language topic list."""
    added = "\n".join(line for line in difflib.ndiff(old.splitlines(), new.splitlines())
                      if line.startswith("+ "))
    topics = [
        (("调弦", "正调", "紧五弦"), "调弦与音高", "提醒模型先看本曲调弦，再计算弦和徽位，减少选错音。"),
        (("前一段", "复用", "动机"), "上下文复用", "要求参考上一乐句和重复旋律，但仍逐音核对，避免机械照抄。"),
        (("音色", "按音/散音/泛音", "泛音"), "取音方式", "补充什么时候用按音、散音或泛音，避免只顾音高却选错演奏方式。"),
        (("get_pitch_candidates", "edit_plan", "协议"), "工具使用", "把查询音高、提交编辑和修复警告的顺序写得更明确。"),
        (("左手", "右手", "指法"), "左右手指法", "要求把按指和右手技法写完整，并注意连续演奏是否顺手。"),
        (("吟", "猱", "绰", "注", "装饰"), "装饰与技法", "补充装饰音和复杂技法的使用条件，避免为了丰富而乱加。"),
        (("行标识", "纯整数", "徽外"), "格式与协议", "强调提交格式必须可解析，避免格式错误导致整条样本记零分。"),
    ]
    found = [(title, description) for keys, title, description in topics
             if any(key in added for key in keys)]
    return found[:4] or [("表达与结构", "重新组织规则和措辞，让模型更清楚地按步骤完成任务。")]


def score_flow(old_value: Any, new_value: Any, status: str) -> str:
    old_text, new_text = score(old_value), score(new_value)
    arrow = "→"
    caption = {"accepted": "变好，继续验证", "rejected": "变差，停止采用",
               "interrupted": "没有完整结果"}[status]
    return (f'<div class="score-flow"><div><small>修改前</small><strong>{old_text}</strong></div>'
            f'<b>{arrow}</b><div><small>修改后</small><strong>{new_text}</strong></div>'
            f'<span class="flow-note {status}">{caption}</span></div>')


def build_report(run_dir: Path, prompt_key: str, title: str) -> str:
    candidates_raw = read_json(run_dir / "candidates.json", [])
    candidates = [item.get(prompt_key, "") if isinstance(item, dict) else "" for item in candidates_raw]
    best = read_json(run_dir / "best_candidate.json", {})
    run_entries = read_json(run_dir / "run_log.json", [])
    log_text, log_name = load_text_log(run_dir)
    attempts = parse_proposals(log_text, prompt_key)
    val_scores = best.get("val_aggregate_scores") or []
    entries_by_iteration = {int(entry.get("i", -1)) + 1: entry for entry in run_entries if isinstance(entry, dict)}

    seed = candidates[0] if candidates else ""
    overview_rows = []
    for index, candidate in enumerate(candidates):
        value = val_scores[index] if index < len(val_scores) else None
        overview_rows.append(
            f"<tr><td><b>版本 {index}</b>{' <span class=best-mark>最后采用</span>' if index == best.get('best_idx') else ''}</td>"
            f"<td>{score(value)}</td><td>{sha(candidate) if candidate else '未保存'}</td></tr>"
        )

    cards = []
    for attempt in attempts:
        entry = entries_by_iteration.get(attempt["iteration"], {})
        # A duplicated iteration before resume is not the committed JSON attempt.
        if attempt["status"] == "interrupted":
            entry = {}
        parent_idx = attempt["parent_idx"]
        parent = candidates[parent_idx] if 0 <= parent_idx < len(candidates) else ""
        new_prompt = attempt["prompt"]
        stats = diff_stats(parent, new_prompt)
        old_scores = entry.get("subsample_scores") or []
        new_scores = entry.get("new_subsample_scores") or []
        sample_ids = entry.get("subsample_ids") or []
        status_label = {"accepted": "接受", "rejected": "拒绝", "interrupted": "中断/未决"}[attempt["status"]]
        status_class = attempt["status"]
        candidate_idx = attempt["candidate_idx"]
        next_link = (f"它被保存为版本 {candidate_idx}，后面的优化可以在它的基础上继续。"
                     if candidate_idx is not None else
                     "它没有进入正式版本列表，后面仍从原来的版本继续。")
        if status_class == "accepted":
            plain_result = "这一版在本轮抽查中表现更好，所以保留下来继续做完整验证。"
        elif status_class == "rejected":
            plain_result = "这一版在本轮抽查中反而更差，所以放弃，不会覆盖之前的好版本。"
        else:
            plain_result = "这次运行中途停了，没有形成有效结论；报告保留它，方便你知道过程在哪里断过。"
        themes = explain_changes(parent, new_prompt)
        theme_html = "".join(f'<li><b>{esc(name)}</b><span>{esc(text)}</span></li>' for name, text in themes)
        cards.append(f"""
<article id="attempt-{attempt['occurrence']}" class="step-card {status_class}">
  <header class="step-head"><div class="step-number">{attempt['occurrence']}</div><div>
    <p class="eyebrow">第 {attempt['occurrence']} 步 · 从版本 {parent_idx} 开始修改</p>
    <h3>{'保留这次修改' if status_class == 'accepted' else '放弃这次修改' if status_class == 'rejected' else '这次修改中途停止'} <span class="badge {status_class}">{status_label}</span></h3>
  </div></header>
  <p class="plain-result">{esc(plain_result)}</p>
  {score_flow(attempt['old_total'], attempt['new_total'], status_class)}
  <div class="change-box"><h4>这一轮主要改了什么？</h4><ul class="themes">{theme_html}</ul></div>
  <div class="next-box"><b>接下来：</b>{esc(next_link)}</div>
  <details class="friendly-detail"><summary>看看具体改了哪些文字</summary>
    <p class="legend"><span class="legend-del">红色 − 删除</span><span class="legend-add">绿色 ＋ 新增</span> 没有颜色的是上下文。</p>
    <pre class="diff">{unified_diff(parent, new_prompt, f'修改前：版本 {parent_idx}', f'修改后：步骤 {attempt["occurrence"]}')}</pre>
  </details>
  <div class="tech-detail">
    <details><summary>查看修改前的完整提示词</summary><pre>{esc(parent) if parent else '运行记录未保存父提示词全文'}</pre></details>
    <details><summary>查看修改后的完整提示词</summary><pre>{esc(new_prompt)}</pre></details>
    <details><summary>查看每条抽查数据的分数</summary>{render_score_table(old_scores, new_scores, sample_ids)}
      <p>修改前抽查均分 {score(mean(old_scores))} → 修改后抽查均分 {score(mean(new_scores))}</p></details>
    <details><summary>查看技术记录</summary>
      <p>GEPA Iteration {attempt['iteration']}；版本指纹 {sha(parent) if parent else '缺失'} → {sha(new_prompt)}。</p>
      <p>验证集：修改前 {score(attempt['parent_score'])}；修改后 {score(attempt['val_score'])}。</p>
      <p>文本变化：新增 {stats['added']} 行，删除 {stats['deleted']} 行，替换 {stats['replaced_old']}→{stats['replaced_new']} 行，未变 {stats['unchanged']} 行。</p>
      <p>日志依据：{esc(attempt['basis'])}</p>
    </details>
  </div>
</article>""")

    limitations = []
    if not log_text:
        limitations.append("没有找到文本日志，无法恢复被拒绝或中断的提案全文。")
    if not candidates:
        limitations.append("没有找到 candidates.json，无法恢复已接受候选的父版本。")
    if not run_entries:
        limitations.append("没有找到 run_log.json，无法展示结构化抽样分数。")
    if not attempts:
        limitations.append("没有解析到任何 Proposed new text 记录。")
    warning = "；".join(limitations) if limitations else "过程文件齐全，这个页面可以还原已保存的每一次修改。"
    first_score = val_scores[0] if val_scores else None
    best_score = best.get("best_val_score")
    improvement = (float(best_score) - float(first_score)
                   if isinstance(first_score, (int, float)) and isinstance(best_score, (int, float)) else None)
    accepted_count = sum(item["status"] == "accepted" for item in attempts)
    rejected_count = sum(item["status"] == "rejected" for item in attempts)
    interrupted_count = sum(item["status"] == "interrupted" for item in attempts)
    journey = "".join(
        f'<a href="#attempt-{item["occurrence"]}" class="journey-node {item["status"]}">'
        f'<span>{item["occurrence"]}</span><b>{"保留" if item["status"] == "accepted" else "放弃" if item["status"] == "rejected" else "中断"}</b></a>'
        for item in attempts
    )

    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title>
<style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:#f6f2e9;color:#26322d;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}
.page{{max-width:1120px;margin:auto;padding:32px 22px 80px}}.hero{{background:linear-gradient(135deg,#173f36,#285d4e);color:#fff;border-radius:24px;padding:38px 42px;box-shadow:0 18px 45px #173f3628}}
.hero h1{{font-size:clamp(30px,5vw,48px);line-height:1.16;margin:8px 0 14px}}.hero p{{max-width:760px;color:#e7f0eb;margin:0}}.kicker{{letter-spacing:.12em;font-size:13px;font-weight:800;color:#bce0cd!important}}
.view-controls{{display:flex;gap:10px;flex-wrap:wrap;margin-top:24px}}button{{border:1px solid #ffffff66;background:#ffffff12;color:#fff;border-radius:999px;padding:10px 18px;font:inherit;font-weight:700;cursor:pointer}}button.active{{background:#fff;color:#173f36}}
section{{background:#fffdf8;border:1px solid #ded8ca;border-radius:18px;padding:24px 28px;margin:20px 0;box-shadow:0 5px 20px #443b2c0c}}h2{{font-size:25px;margin:0 0 14px}}h3{{font-size:22px;line-height:1.35;margin:2px 0}}h4{{margin:0 0 10px}}.notice{{background:#fff7dc;border-color:#e2c976;color:#5d4a18}}
.summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}}.metric{{background:#f0f6f1;border-radius:14px;padding:16px}}.metric strong{{display:block;font-size:26px;color:#195a47}}.metric small{{color:#607069}}
.journey{{display:flex;align-items:center;gap:12px;overflow:auto;padding:10px 2px}}.journey-node{{min-width:104px;text-decoration:none;color:#24312c;text-align:center;border:2px solid #d9d4c8;border-radius:16px;padding:12px;background:#fff}}.journey-node span{{display:block;margin:auto;width:34px;height:34px;line-height:34px;border-radius:50%;background:#e9ede9;font-weight:800}}.journey-node.accepted{{border-color:#72b596}}.journey-node.rejected{{border-color:#e39b98}}.journey-node.interrupted{{border-color:#dfbf63}}.journey-node:not(:last-child)::after{{content:"→";position:relative;left:64px;color:#887f70}}
.step-card{{background:#fffdf8;border:1px solid #ded8ca;border-left:7px solid #aaa;border-radius:18px;padding:24px 28px;margin:22px 0;box-shadow:0 8px 24px #443b2c10}}.step-card.accepted{{border-left-color:#2f8f68}}.step-card.rejected{{border-left-color:#c95d59}}.step-card.interrupted{{border-left-color:#c79b2e}}
.step-head{{display:flex;gap:16px;align-items:center}}.step-number{{flex:none;width:48px;height:48px;border-radius:50%;display:grid;place-items:center;background:#263f36;color:white;font-size:22px;font-weight:800}}.eyebrow{{margin:0;color:#6a756f;font-size:13px;font-weight:800;letter-spacing:.06em}}
.badge,.flow-note{{display:inline-block;padding:.2rem .65rem;border-radius:999px;font-size:13px}}.badge.accepted,.flow-note.accepted{{background:#dff4e9;color:#176344}}.badge.rejected,.flow-note.rejected{{background:#fde5e3;color:#922f2b}}.badge.interrupted,.flow-note.interrupted{{background:#fff0bd;color:#765714}}
.plain-result{{font-size:18px;background:#f4f2ec;border-radius:12px;padding:14px 16px}}.score-flow{{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin:18px 0}}.score-flow div{{background:#edf4ef;border-radius:12px;padding:10px 18px;min-width:130px}}.score-flow small,.score-flow strong{{display:block}}.score-flow strong{{font-size:24px;color:#195a47}}
.change-box{{background:#eef6f1;border-radius:14px;padding:18px 20px}}.themes{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0;padding:0;list-style:none}}.themes li{{background:#fff;border:1px solid #d7e5dc;border-radius:10px;padding:12px}}.themes b,.themes span{{display:block}}.themes span{{color:#58655f;font-size:14px;margin-top:3px}}.next-box{{margin:15px 0;border-left:4px solid #5a8f78;padding:9px 14px;background:#f7f4ec}}
details{{border-top:1px solid #e3ddd1;margin-top:12px;padding-top:12px}}summary{{cursor:pointer;color:#235d4b;font-weight:800}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#202a26;color:#eef4f0;border-radius:10px;padding:16px;max-height:680px;overflow:auto;font:13px/1.6 Consolas,monospace}}
.diff span{{display:block;min-height:1.4em}}.diff .add{{background:#194d37;color:#d9ffe8}}.diff .del{{background:#632d2c;color:#ffe0de}}.diff .hunk{{color:#dcecff;background:#2f4350}}.legend span{{display:inline-block;margin-right:14px;padding:2px 8px;border-radius:6px}}.legend-add{{background:#dff4e9;color:#176344}}.legend-del{{background:#fde5e3;color:#922f2b}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e1dcd2}}th{{color:#66716c;font-size:13px}}.best-mark{{background:#dff4e9;color:#176344;border-radius:999px;padding:2px 7px;font-size:12px}}code{{background:#ede9df;padding:.12rem .3rem;border-radius:4px;overflow-wrap:anywhere}}.glossary{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.glossary div{{background:#f6f3eb;border-radius:10px;padding:12px}}.glossary b{{display:block;color:#195a47}}
body.simple .tech-detail,body.simple .technical-section{{display:none}}.footer-note{{color:#69756f;font-size:14px}}@media(max-width:760px){{.page{{padding:16px 12px 50px}}.hero{{padding:26px 22px}}section,.step-card{{padding:20px}}.summary-grid{{grid-template-columns:1fr 1fr}}.themes,.glossary{{grid-template-columns:1fr}}.journey-node:not(:last-child)::after{{display:none}}}}
</style></head><body>
<div class="page"><header class="hero"><p class="kicker">PROMPT EVOLUTION · 提示词成长记录</p><h1>{esc(title)}</h1>
<p>把复杂日志翻译成一条容易看懂的路线：每次改了什么、分数有没有变好、最后为什么留下或放弃。</p>
<div class="view-controls"><button id="simpleButton" class="active" onclick="setView('simple')">只看重点</button><button id="fullButton" onclick="setView('full')">查看全部细节</button></div></header>
<section><h2>先说结论</h2><p class="plain-result">提示词从版本 0 出发，经历 {len(attempts)} 次修改记录，最后选择版本 {esc(best.get('best_idx', '未记录'))}。验证分从 {score(first_score)} 变为 {score(best_score)}，变化 {score(improvement)}。</p>
<div class="summary-grid"><div class="metric"><small>修改尝试</small><strong>{len(attempts)}</strong></div><div class="metric"><small>保留下来</small><strong>{accepted_count}</strong></div><div class="metric"><small>放弃</small><strong>{rejected_count}</strong></div><div class="metric"><small>中途中断</small><strong>{interrupted_count}</strong></div></div></section>
<section><h2>一眼看懂整个过程</h2><p>点击任意节点，可以直接跳到那一次修改。</p><div class="journey"><div class="journey-node accepted"><span>0</span><b>原始版</b></div>{journey}<div class="journey-node accepted"><span>✓</span><b>最终版</b></div></div></section>
<section><h2>几个词先解释清楚</h2><div class="glossary"><div><b>版本 / 候选</b>一份完整提示词。只有表现较好的修改才会成为新版本。</div><div><b>抽查分数</b>先用少量样本快速试验，判断这次修改值不值得继续。</div><div><b>验证分数</b>通过抽查后，再用固定验证集进行更完整的比较。</div><div><b>中断</b>提案已经生成，但运行没有完成，所以不能判断好坏。</div></div></section>
<section><h2>逐步查看每一次修改</h2><p>每一步先给通俗结论；想看原文时，再展开“具体改了哪些文字”。</p>{''.join(cards) if cards else '<p>没有可展示的提案。</p>'}</section>
<section class="technical-section"><h2>版本总表</h2><table><thead><tr><th>版本</th><th>验证分数</th><th>文本指纹</th></tr></thead><tbody>{''.join(overview_rows)}</tbody></table>
<p class="footer-note">运行目录：<code>{esc(run_dir)}</code><br>提示词字段：<code>{esc(prompt_key)}</code>；日志：<code>{esc(log_name or '未找到')}</code>。</p></section>
<section class="technical-section"><h2>最初的完整提示词</h2><p>这是所有修改的起点，也就是版本 0。</p><details><summary>展开原始提示词全文</summary><pre>{esc(seed) if seed else '未保存'}</pre></details></section>
<p class="notice">{esc(warning)} 页面只解释日志中真实保存的修改和分数，不会编造模型没有留下的思考内容。</p></div>
<script>function setView(mode){{document.body.className=mode;document.getElementById('simpleButton').classList.toggle('active',mode==='simple');document.getElementById('fullButton').classList.toggle('active',mode==='full');localStorage.setItem('prompt-report-view',mode)}}setView(localStorage.getItem('prompt-report-view')||'simple')</script>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 GEPA 的种子、每轮提案、逐行差异、评分和接受/拒绝决策生成完整 HTML。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="GEPA 运行目录，需含 candidates.json/run_log.json/文本日志")
    parser.add_argument("--output", type=Path, default=Path("prompt_evolution_report.html"), help="输出 HTML 路径")
    parser.add_argument("--prompt-key", default="single_stage", help="candidates.json 中的提示词字段名")
    parser.add_argument("--title", default="GEPA 提示词完整修改过程", help="HTML 页面标题")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"运行目录不存在：{run_dir}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(run_dir, args.prompt_key, args.title), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
