# -*- coding: utf-8 -*-
"""Render a self-contained, plain-language audit from production prompt to V9."""
from __future__ import annotations

import difflib
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def load_text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8").strip()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def score(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def average(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def arm_summary(result: dict, arm: str) -> dict:
    return result["summary"][arm]


def paired_rows(result: dict, candidate: str = "v9") -> list[dict]:
    rows = []
    for sample_id in result["sample_ids"]:
        production = next(row for row in result["rows"]
                          if row["sample_id"] == sample_id and row["arm"] == "production")
        trial = next(row for row in result["rows"]
                     if row["sample_id"] == sample_id and row["arm"] == candidate)
        rows.append({
            "sample": sample_id,
            "production": production["score"],
            "candidate": trial["score"],
            "delta": trial["score"] - production["score"],
            "production_rules": production["rule_violations"],
            "candidate_rules": trial["rule_violations"],
            "production_calls": production["actor_calls"],
            "candidate_calls": trial["actor_calls"],
        })
    return rows


def component_mean(rows: list[dict], arm: str, component: str) -> float | None:
    values = [row["components"].get(component) for row in rows if row["arm"] == arm]
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def completed_candidate_pairs(results: list[dict], baseline: str, candidate: str) -> list[tuple]:
    pairs = []
    for result in results:
        for sample_id in result["sample_ids"]:
            rows = [row for row in result["rows"] if row["sample_id"] == sample_id]
            arms = {row["arm"] for row in rows}
            if baseline in arms and candidate in arms:
                left = next(row for row in rows if row["arm"] == baseline)
                right = next(row for row in rows if row["arm"] == candidate)
                pairs.append((left, right))
    return pairs


def prompt_details(label: str, filename: str) -> str:
    return (
        f'<details class="prompt"><summary>{esc(label)}：查看完整提示词补丁</summary>'
        f'<pre>{esc(load_text(filename))}</pre></details>'
    )


def text_details(label: str, text: str, note: str = "") -> str:
    note_html = f'<p class="small">{esc(note)}</p>' if note else ""
    return (
        f'<details class="prompt"><summary>{esc(label)}</summary>{note_html}'
        f'<pre>{esc(text)}</pre></details>'
    )


def diff_details(old_label: str, old_file: str, new_label: str, new_file: str) -> str:
    diff = "\n".join(difflib.unified_diff(
        load_text(old_file).splitlines(), load_text(new_file).splitlines(),
        fromfile=old_label, tofile=new_label, lineterm="",
    ))
    return (
        f'<details class="diff"><summary>逐行查看：{esc(old_label)} → {esc(new_label)}</summary>'
        f'<pre>{esc(diff)}</pre></details>'
    )


def main() -> None:
    pilot = load_json("glm_dataset_pilot_result.json")
    v3_result = load_json("glm_prompt_v3_train_result.json")
    v4_result = load_json("glm_prompt_v4_train_result.json")
    v5_result = load_json("glm_prompt_v5_train_result.json")
    v6_dev = load_json("prompt_v6_dev_paired_result.json")
    v7_diag = load_json("prompt_v7_diagnostic_result.json")
    v7_fresh = load_json("prompt_v7_fresh_dev_result.json")
    v7_test = load_json("prompt_v7_test_pilot_result.json")
    v8_dev = load_json("prompt_v8_fresh_dev_result.json")
    v9_dev = load_json("prompt_v9_dev_replay_result.json")
    v10_dev = load_json("prompt_v10_dev_replay_result.json")
    v11_dev = load_json("prompt_v11_targeted_dev_result.json")
    v12_dev = load_json("prompt_v12_targeted_replay_result.json")
    v13_dev = load_json("prompt_v13_targeted_replay_result.json")
    v14_dev = load_json("prompt_v14_clean_broad_dev_result.json")
    v15_dev_a = load_json("prompt_v15_integrated_dev_result.json")
    v15_dev_b = load_json("prompt_v15_confirmation_dev_result.json")
    v16_dev = load_json("prompt_v16_dev_replay_a_result.json")
    repair_unsafe = load_json("prompt_two_stage_repair_dev_result.json")
    repair_safe = load_json("prompt_two_stage_safe_repair_dev_result.json")
    holdout_a = load_json("prompt_v9_final_holdout_result.json")
    holdout_b = load_json("prompt_v9_confirmation_holdout_result.json")

    production_record = load_json(
        "guqinagent/baseline/gepa_guqin/runs/pilot_final_20261008/best_candidate.json"
    )
    production_prompt = production_record["best_candidate"]["single_stage"]
    v2_prompt = load_text("guqinagent/baseline/gepa_guqin/prompts/single_stage_curated_v2.txt")
    pilot_groups = {}
    for row in pilot["rows"]:
        pilot_groups.setdefault(row["prompt"], []).append(row["score"])
    pilot_production = statistics.fmean(pilot_groups["production_seed_plus_handbook"])
    pilot_v2 = statistics.fmean(pilot_groups["gpt_curated_v2_plus_handbook"])
    v3_mean = v3_result["mean_score"]
    v4_mean = v4_result["mean_score"]
    v5_pilot_mean = v5_result["mean_score"]
    v6_prod = arm_summary(v6_dev, "production")
    v6_v5 = arm_summary(v6_dev, "v5")
    v6 = arm_summary(v6_dev, "v6")
    v7_diagnostic = arm_summary(v7_diag, "v7")
    v7_fresh_prod = arm_summary(v7_fresh, "production")
    v7_fresh_candidate = arm_summary(v7_fresh, "v7")

    final_rows = paired_rows(holdout_a) + paired_rows(holdout_b)
    deltas = [row["delta"] for row in final_rows]
    all_eval_rows = holdout_a["rows"] + holdout_b["rows"]
    prod_mean = statistics.fmean(row["production"] for row in final_rows)
    v9_mean = statistics.fmean(row["candidate"] for row in final_rows)
    delta_mean = statistics.fmean(deltas)
    median_delta = statistics.median(deltas)
    wins = sum(value > 1e-9 for value in deltas)
    ties = sum(abs(value) <= 1e-9 for value in deltas)
    losses = sum(value < -1e-9 for value in deltas)
    largest = max(final_rows, key=lambda row: row["delta"])
    without_largest = statistics.fmean(
        row["delta"] for row in final_rows if row is not largest
    )
    prod_rules = sum(row["production_rules"] or 0 for row in final_rows)
    v9_rules = sum(row["candidate_rules"] or 0 for row in final_rows)
    prod_calls = statistics.fmean(row["production_calls"] for row in final_rows)
    v9_calls = statistics.fmean(row["candidate_calls"] for row in final_rows)

    component_labels = {
        "pitch": "音高",
        "tone_type": "按 / 散 / 泛",
        "fingering": "左右手指法",
        "ornament": "装饰技法",
        "playability": "可演奏性",
    }
    component_rows = []
    for key, label in component_labels.items():
        p = component_mean(all_eval_rows, "production", key)
        v = component_mean(all_eval_rows, "v9", key)
        component_rows.append((label, p, v, None if p is None or v is None else v - p))

    sample_table = "".join(
        f'<tr><td><code>{esc(row["sample"])}</code></td>'
        f'<td>{score(row["production"])}</td><td>{score(row["candidate"])}</td>'
        f'<td class="{"good" if row["delta"] > 0 else "bad"}">{row["delta"]:+.4f}</td>'
        f'<td>{row["production_rules"]} → {row["candidate_rules"]}</td>'
        f'<td>{row["production_calls"]} → {row["candidate_calls"]}</td></tr>'
        for row in final_rows
    )
    component_table = "".join(
        f'<tr><td>{esc(label)}</td><td>{score(p)}</td><td>{score(v)}</td>'
        f'<td class="{"good" if delta is not None and delta >= 0 else "bad"}">'
        f'{"—" if delta is None else f"{delta:+.4f}"}</td></tr>'
        for label, p, v, delta in component_rows
    )

    v7_prod = arm_summary(v7_test, "production")
    v7 = arm_summary(v7_test, "v7")
    v8_prod = arm_summary(v8_dev, "production")
    v8 = arm_summary(v8_dev, "v8")
    v9 = arm_summary(v9_dev, "v9")
    v10 = arm_summary(v10_dev, "v10")
    v11_base, v11 = arm_summary(v11_dev, "v9"), arm_summary(v11_dev, "v11")
    v14_base, v14 = arm_summary(v14_dev, "v9"), arm_summary(v14_dev, "v14")
    v15_pairs = completed_candidate_pairs([v15_dev_a, v15_dev_b], "v9", "v15")
    v15_base_mean = statistics.fmean(left["score"] for left, _ in v15_pairs)
    v15_mean = statistics.fmean(right["score"] for _, right in v15_pairs)
    v15_wins = sum(right["score"] > left["score"] for left, right in v15_pairs)
    v15_losses = sum(right["score"] < left["score"] for left, right in v15_pairs)
    v16_pairs = []
    for right in v16_dev["rows"]:
        left = next(row for row in v15_dev_a["rows"]
                    if row["sample_id"] == right["sample_id"] and row["arm"] == "v9")
        v16_pairs.append((left, right))
    v16_base_mean = statistics.fmean(left["score"] for left, _ in v16_pairs)
    v16_mean = statistics.fmean(right["score"] for _, right in v16_pairs)

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>古琴提示词：原始版 → V9 完整优化记录</title>
<style>
:root{{--ink:#17221b;--muted:#637067;--paper:#f7f3e8;--card:#fffdf7;--line:#d8d0bd;--green:#176b4d;--green2:#dff2e8;--red:#a84439;--red2:#f9e5df;--gold:#af7a22;--blue:#315f7d}}
*{{box-sizing:border-box}} body{{margin:0;color:var(--ink);background:var(--paper);font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;line-height:1.75}}
.hero{{padding:64px 24px 52px;background:linear-gradient(135deg,#143c31,#245b43 60%,#92702f);color:white}}
.wrap{{max-width:1120px;margin:auto}} h1{{font-size:clamp(34px,6vw,68px);line-height:1.08;margin:0 0 18px;letter-spacing:-2px}} h2{{font-size:30px;margin:58px 0 18px}} h3{{font-size:21px;margin:0 0 10px}} p{{margin:8px 0 14px}} .lead{{max-width:820px;font-size:19px;color:#eaf3ec}}
.result-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:28px}} .result-strip div{{background:#ffffff18;border:1px solid #ffffff33;border-radius:16px;padding:16px}} .result-strip strong{{display:block;font-size:28px}} .result-strip span{{font-size:13px;color:#e0ece4}}
main{{padding:0 24px 70px}} .notice{{margin-top:28px;padding:18px 20px;border-left:5px solid var(--gold);background:#fff4d8;border-radius:10px}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 8px 30px #253a2710}}
.good{{color:var(--green);font-weight:700}} .bad{{color:var(--red);font-weight:700}} .pill{{display:inline-block;padding:4px 10px;border-radius:999px;font-size:13px;font-weight:700}} .pass{{background:var(--green2);color:var(--green)}} .fail{{background:var(--red2);color:var(--red)}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}} .step{{position:relative;padding:18px;background:var(--card);border:1px solid var(--line);border-radius:16px}} .step b{{display:block;color:var(--blue);font-size:14px}} .step strong{{display:block;font-size:22px;margin:5px 0}} .step p{{font-size:14px;color:var(--muted)}}
.route{{position:relative;margin:22px 0;padding-left:44px}} .route:before{{content:"";position:absolute;left:16px;top:12px;bottom:12px;width:3px;background:#c8bea9}} .route-card{{position:relative;margin:0 0 18px;padding:20px 22px;background:var(--card);border:1px solid var(--line);border-radius:16px}} .route-card:before{{content:"";position:absolute;left:-38px;top:24px;width:17px;height:17px;border-radius:50%;background:var(--blue);border:4px solid var(--paper)}} .route-card.kept:before{{background:var(--green)}} .route-card.dropped:before{{background:var(--red)}} .route-card h3{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}} .why-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px}} .why-grid div{{padding:12px 14px;border-radius:10px;background:#f2eee3;font-size:14px}} .why-grid b{{display:block;color:var(--blue);margin-bottom:3px}} .formula{{padding:17px 20px;background:#e8f0e8;border-radius:14px;text-align:center;font-size:18px}} .arrow{{color:var(--gold);font-weight:800;padding:0 8px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border-radius:14px;overflow:hidden}} th,td{{padding:11px 12px;border-bottom:1px solid #e7e0d1;text-align:left}} th{{background:#ebe4d4;font-size:13px}} tr:last-child td{{border-bottom:0}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}}
details{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:12px 0}} summary{{cursor:pointer;padding:14px 16px;font-weight:700}} pre{{margin:0;padding:16px;overflow:auto;background:#18231d;color:#e8efe9;font-family:Consolas,"Microsoft YaHei",monospace;font-size:13px;line-height:1.65;white-space:pre-wrap}} code{{font-family:Consolas,monospace}} .plain{{font-size:18px;background:#e8f0e8;border-radius:16px;padding:20px}} .small{{font-size:13px;color:var(--muted)}}
@media(max-width:850px){{.result-strip,.grid{{grid-template-columns:1fr 1fr}}.flow{{grid-template-columns:1fr}}}} @media(max-width:560px){{.result-strip,.grid,.why-grid{{grid-template-columns:1fr}} h2{{font-size:25px}}.route{{padding-left:34px}}.route-card:before{{left:-30px}}}}
</style>
</head>
<body>
<header class="hero"><div class="wrap">
  <div class="pill" style="background:#ffffff22;color:white">冻结版本：V9 · 模型：GLM-5.3</div>
  <h1>从原始版到 V9<br>提示词是怎样一步步修出来的</h1>
  <p class="lead">从最初生产提示词开始，完整展示 V2 预实验、V3–V6 的试错、V7 的转折、V8 的回退和 V9 的最小修复。每一步都回答：为什么改、改了什么、数据怎样、为什么保留或淘汰。</p>
  <div class="result-strip">
    <div><strong>{prod_mean:.4f}</strong><span>原提示词 · 16 条测试</span></div>
    <div><strong>{v9_mean:.4f}</strong><span>V9 · 16 条测试</span></div>
    <div><strong>+{delta_mean:.4f}</strong><span>绝对提升（约 {delta_mean/prod_mean*100:.2f}%）</span></div>
    <div><strong>{wins} 胜 {losses} 负</strong><span>{ties} 条持平</span></div>
  </div>
</div></header>
<main><div class="wrap">
  <div class="notice"><b>先说结论：</b>V9 在两批冻结留出测试、共 16 条样本上超过原提示词；第二批独立确认集也是 7 胜 1 负。它不是每条都赢，也还不是完整顺序测试，因此当前最准确的说法是“V9 通过小规模冻结测试，值得进入完整 benchmark”，不是“已经证明全面替代原版”。</div>

  <h2>先看懂“版本”到底是什么</h2>
  <div class="formula"><b>一次实际运行的提示词</b><span class="arrow">＝</span>原始生产提示词<span class="arrow">＋</span>当前版本补丁<span class="arrow">＋</span>固定手册</div>
  <p>V3–V9 的补丁通常是<b>替换关系</b>，不是全部叠加。也就是说，V9 不是“原版 + V3 + V4 + … + V9”，而是“原版 + V9 补丁 + 同一份固定手册”。V2 是一次完整重写的预实验分支；V7 回到 V5 的可解析格式思路，V9 又回到 V7 只修两个明确错误。</p>
  <div class="notice"><b>为什么必须讲清楚：</b>如果把所有补丁叠在一起，会制造一份从未测试过的新提示词，页面上的分数也就无法对应真实实验。</div>

  <h2>原始版 → V9：完整修改路线</h2>
  <div class="route">
    <section class="route-card kept"><h3>原始生产版 <span class="pill pass">正式基线</span></h3><p>它已经包含音高优先、警告修复、字段继承、工具调用顺序和局部编辑，是一个能力较完整的起点。3 条早期样本均分 <b>{pilot_production:.4f}</b>。</p><div class="why-grid"><div><b>保留了什么</b>原版主体始终不改，所有候选在同一主体和固定手册上比较。</div><div><b>发现的问题</b>输出写法有时过于省略；双音、多弦与连续泛音容易产生解析或声类歧义。</div></div></section>
    <section class="route-card dropped"><h3>V2 预实验 <span class="pill fail">完整重写失败</span></h3><p>尝试把音高、声类、协议、指法和流程全部写得更详尽。同一批 3 条样本从原版 <b>{pilot_production:.4f}</b> 降到 <b>{pilot_v2:.4f}</b>。</p><div class="why-grid"><div><b>修改方法</b>用一份很长的完整提示词替换原版。</div><div><b>得到的经验</b>一次重写太多变量，分数下降后无法定位是哪条规则造成的；后续改用“小补丁”。</div></div></section>
    <section class="route-card dropped"><h3>V3 · 保守编配 <span class="pill fail">未采用</span></h3><p>限制无依据的双弦、走手和装饰，鼓励简单可演奏方案。3 条样本均分 <b>{v3_mean:.4f}</b>，低于原版的 {pilot_production:.4f}。</p><div class="why-grid"><div><b>想解决</b>模型为了“丰富”而堆复杂技法、重复发声。</div><div><b>为什么放弃</b>约束面太宽，可能把合理的音乐处理一起压掉，而且没有解决字符串可解析性这个更直接的问题。</div></div></section>
    <section class="route-card dropped"><h3>V4 · 只防新增违规 <span class="pill fail">明显回退</span></h3><p>把 V3 缩成“不改变音乐判断，只拦截非法编辑”的护栏，但 3 条均分降到 <b>{v4_mean:.4f}</b>；其中一条只有 0.1955。</p><div class="why-grid"><div><b>想解决</b>避免 V3 的“过度简化”干扰声类与艺术处理。</div><div><b>为什么放弃</b>即使措辞说“不改变”，护栏仍会影响模型决策；小样本出现灾难性回退，不能继续。</div></div></section>
    <section class="route-card kept"><h3>V5 · 只管最终格式 <span class="pill pass">关键转向</span></h3><p>不再管音乐风格，只要求弦、徽、按/散/泛等字段写得可解析。早期 3 条均分 <b>{v5_pilot_mean:.4f}</b>；随后 6 条配对实验为 <b>{v6_v5['mean_score']:.4f}</b>，高于同批原版 {v6_prod['mean_score']:.4f}。</p><div class="why-grid"><div><b>改了什么</b>明确“弦”字、按音/散音/泛音和推荐写法。</div><div><b>为什么成为主线</b>改动目标单一，能对应评分器和解析器；但仍有 2 处违规，所以只是过渡版本。</div></div></section>
    <section class="route-card dropped"><h3>V6 · 加入整句声类决策 <span class="pill fail">均分高但风险不合格</span></h3><p>6 条配对均分 <b>{v6['mean_score']:.4f}</b>，表面高于 V5；但音高从 1.0000 降到 {v6['component_means']['pitch']:.4f}，违规增至 {v6['total_rule_violations']} 处。</p><div class="why-grid"><div><b>想解决</b>先决定按、散、泛，再格式化，避免为写法方便改变音色。</div><div><b>为什么放弃</b>出现“散历456弦 / 散历4321弦”一类把多个不同音高压成一个动作的写法；综合均分掩盖了音高和违规红线。</div></div></section>
    <section class="route-card kept"><h3>V7 · 回到 V5，补双音和“历” <span class="pill pass">开发集通过</span></h3><p>在同一 6 条诊断集上均分 <b>{v7_diagnostic['mean_score']:.4f}</b>、音高恢复 1.0000、违规回到 {v7_diagnostic['total_rule_violations']}；新 6 条开发集又从原版 {v7_fresh_prod['mean_score']:.4f} 升到 <b>{v7_fresh_candidate['mean_score']:.4f}</b>。</p><div class="why-grid"><div><b>改了什么</b>规定双音的标准复合外壳；没有逐音核验时禁止用“历”压缩多个音。</div><div><b>新问题</b>首批 6 条测试反而低 0.0009：格式规则被误读成音乐规则，连续泛音和双散音会被改坏。</div></div></section>
    <section class="route-card dropped"><h3>V8 · 把两阶段讲得更完整 <span class="pill fail">解释过重</span></h3><p>要求先锁定音乐决策、再格式化。8 条开发集为 <b>{v8['mean_score']:.4f}</b>，虽然略高于原版 {v8_prod['mean_score']:.4f}，却低于同批 V7 的 {arm_summary(v8_dev,'v7')['mean_score']:.4f}；声类只有 {v8['component_means']['tone_type']:.4f}。</p><div class="why-grid"><div><b>想解决</b>彻底阻止格式规则反过来改变音乐选择。</div><div><b>为什么放弃</b>流程说明太长、约束太强，增加模型负担，也制造新的决策扰动。</div></div></section>
    <section class="route-card kept"><h3>V9 · 回到 V7，只做两处最小修复 <span class="pill pass">最终冻结</span></h3><p>只补“连续泛音不要被逐行格式化打断”和“复合双音保持成员原定声类”。同批开发集达到 <b>{v9['mean_score']:.4f}</b>，最终 16 条冻结测试达到 <b>{v9_mean:.4f}</b>，超过原版 {prod_mean:.4f}。</p><div class="why-grid"><div><b>生成方法</b>从失败样本提取两个可复现错误，一条错误只加一条局部规则。</div><div><b>为什么保留</b>11 胜 5 负、平均提升 {delta_mean:+.4f}；去掉最大提升样本后仍为正。</div></div></section>
  </div>

  <h2>每一版到底改了哪些原文</h2>
  <p>下面默认折叠，想核查时再展开。原始主体与固定手册不随 V3–V9 改动；每版只替换自己的补丁。</p>
  {text_details('原始生产提示词全文', production_prompt, '这是所有候选共享的主体，不是 vanilla ReAct。')}
  {text_details('V2 预实验的完整重写提示词', v2_prompt, '这是独立分支，失败后没有继续作为父版本。')}
  {prompt_details('V3', 'single_stage_conservative_v3_addendum.txt')}
  {prompt_details('V4', 'single_stage_guardrails_v4_addendum.txt')}
  {prompt_details('V5', 'single_stage_parseable_v5_addendum.txt')}
  {prompt_details('V6', 'single_stage_parseable_tone_v6_addendum.txt')}
  {prompt_details('V7', 'single_stage_parseable_compound_v7_addendum.txt')}
  {prompt_details('V8', 'single_stage_tone_state_compound_v8_addendum.txt')}
  {prompt_details('V9', 'single_stage_minimal_tone_compound_v9_addendum.txt')}
  {diff_details('V3', 'single_stage_conservative_v3_addendum.txt', 'V4', 'single_stage_guardrails_v4_addendum.txt')}
  {diff_details('V4', 'single_stage_guardrails_v4_addendum.txt', 'V5', 'single_stage_parseable_v5_addendum.txt')}
  {diff_details('V5', 'single_stage_parseable_v5_addendum.txt', 'V6', 'single_stage_parseable_tone_v6_addendum.txt')}
  {diff_details('V6', 'single_stage_parseable_tone_v6_addendum.txt', 'V7', 'single_stage_parseable_compound_v7_addendum.txt')}
  {diff_details('V7', 'single_stage_parseable_compound_v7_addendum.txt', 'V8', 'single_stage_tone_state_compound_v8_addendum.txt')}
  {diff_details('V7', 'single_stage_parseable_compound_v7_addendum.txt', 'V9', 'single_stage_minimal_tone_compound_v9_addendum.txt')}

  <h2>V7 以后：30 秒看懂验证过程</h2>
  <div class="flow">
    <div class="step"><b>起点</b><strong>V7</strong><span class="pill fail">测试未过</span><p>格式更稳，但声类被带偏；6 条测试比原版低 0.0009。</p></div>
    <div class="step"><b>尝试 1</b><strong>V8</strong><span class="pill fail">淘汰</span><p>解释太重，声类降到 0.5138，均分 0.8161。</p></div>
    <div class="step"><b>尝试 2</b><strong>V9</strong><span class="pill pass">冻结</span><p>回到 V7，只修连续泛音和双音音色误读；开发集 0.8438。</p></div>
    <div class="step"><b>反向验证</b><strong>V10</strong><span class="pill fail">淘汰</span><p>把多弦核对写得更强，反而增加负担；回落到 0.8249。</p></div>
    <div class="step"><b>最终</b><strong>V9</strong><span class="pill pass">测试通过</span><p>16 条：0.8007 对 0.7549，11 胜 5 负。</p></div>
  </div>

  <h2>V7 为什么不如原提示词？</h2>
  <div class="grid">
    <section class="card"><h3>表面结果</h3><p>原版 {score(v7_prod['mean_score'])}，V7 {score(v7['mean_score'])}，差值 {v7['paired_mean_delta_vs_production']:+.4f}。</p><p>音高和可演奏性没有崩，主要损失来自按、散、泛判断。</p></section>
    <section class="card"><h3>真正原因</h3><p><b>连续泛音状态被打断：</b>“泛起…泛止”段中间被误改成按音。</p><p><b>双音规则有歧义：</b>“按在前、散在后”容易被误读成所有“撮”都必须一按一散，双散音被强行改写。</p></section>
  </div>
  <div class="plain"><b>通俗理解：</b>V7 本来只是想把答案写规范，但规范写得像“演奏指令”，模型为了满足格式，反过来改了原本的音乐选择。</div>

  <h2>每一轮为什么这样改</h2>
  <section class="card"><h3>V8：把“先判断、后格式化”讲得很完整</h3><p>开发集原版 {score(v8_prod['mean_score'])}，V8 {score(v8['mean_score'])}，但 V7 同批是 {score(arm_summary(v8_dev,'v7')['mean_score'])}。</p><p class="bad">淘汰原因：文字太多、过程约束太强。它虽然比原版略高，却丢掉 V7 的大部分优势，声类也低于原版。</p></section>
  {diff_details('V7', 'single_stage_parseable_compound_v7_addendum.txt', 'V8', 'single_stage_tone_state_compound_v8_addendum.txt')}
  <section class="card"><h3>V9：回退到 V7，只保留两处最小修复</h3><p>同批开发集 V9 {score(v9['mean_score'])}，高于原版 {score(v8_prod['mean_score'])} 和 V7 {score(arm_summary(v8_dev,'v7')['mean_score'])}；声类、指法、装饰也都高于原版。</p><p class="good">保留原因：规则更短，修改点能直接对应 V7 的两个失败模式，综合分最高。</p></section>
  {diff_details('V7', 'single_stage_parseable_compound_v7_addendum.txt', 'V9', 'single_stage_minimal_tone_compound_v9_addendum.txt')}
  <section class="card"><h3>V10：尝试消除 V9 的一个多弦音高错误</h3><p>V10 把“逐音核对”推广到历、滚、拂，但开发集回落到 {score(v10['mean_score'])}，声类也明显下降。</p><p class="bad">淘汰原因：目标错误没有稳定消失，其他样本却被额外规则干扰。这个反例说明：提示词不是越长越好。</p></section>
  {diff_details('V9', 'single_stage_minimal_tone_compound_v9_addendum.txt', 'V10', 'single_stage_verified_multistring_v10_addendum.txt')}

  <h2>最终 16 条测试：分项发生了什么</h2>
  <div class="grid">
    <section class="card"><h3>总体</h3><p><b>原版：</b>{prod_mean:.4f}<br><b>V9：</b>{v9_mean:.4f}<br><b>平均差：</b><span class="good">{delta_mean:+.4f}</span><br><b>中位差：</b>{median_delta:+.4f}</p></section>
    <section class="card"><h3>稳健性核查</h3><p>最大单条提升：<code>{esc(largest['sample'])}</code>，{largest['delta']:+.4f}。</p><p>去掉这条后，其余 15 条平均仍提升 <b>{without_largest:+.4f}</b>，说明合并 16 条后的优势不再完全依赖它。</p></section>
    <section class="card"><h3>格式违规</h3><p>原版共 {prod_rules} 处，V9 共 {v9_rules} 处。V9 整体更稳，但确认集中仍有一条出现 2 处违规，并非彻底消除。</p></section>
    <section class="card"><h3>调用成本</h3><p>原版平均 {prod_calls:.2f} 次模型调用，V9 平均 {v9_calls:.2f} 次。V9 质量提升伴随一定额外推理成本。</p></section>
  </div>
  <div class="table-wrap"><table><thead><tr><th>评分项</th><th>原版</th><th>V9</th><th>变化</th></tr></thead><tbody>{component_table}</tbody></table></div>

  <h2>逐条看：哪里赢，哪里输</h2>
  <div class="table-wrap"><table><thead><tr><th>样本</th><th>原版</th><th>V9</th><th>差值</th><th>违规 原→V9</th><th>调用 原→V9</th></tr></thead><tbody>{sample_table}</tbody></table></div>
  <p class="small">绿色代表 V9 更高，红色代表 V9 更低。首批 8 条为冻结留出测试；第二批 8 条是在 V9 不再修改的前提下追加的独立确认集。</p>

  <h2>最终采用的 V9 提示词</h2>
  {text_details('展开实际送给模型的主体 + V9 补丁', production_prompt + '\n\n' + load_text('single_stage_minimal_tone_compound_v9_addendum.txt'), '运行时还会附带同一份固定手册；这里展示本轮真正发生变化的系统提示词部分。')}
  {prompt_details('V7', 'single_stage_parseable_compound_v7_addendum.txt')}
  {prompt_details('V8', 'single_stage_tone_state_compound_v8_addendum.txt')}
  {prompt_details('V9（最终候选）', 'single_stage_minimal_tone_compound_v9_addendum.txt')}
  {prompt_details('V10（已淘汰）', 'single_stage_verified_multistring_v10_addendum.txt')}

  <h2>V9 之后继续尝试了什么？</h2>
  <div class="notice"><b>结论：</b>V11–V16 都按预先门槛被淘汰，V9 仍是当前最佳冻结候选。失败版本没有进入新测试集，避免为了追分反复消耗测试数据。</div>
  <div class="grid">
    <section class="card"><h3>V11 · 主动选择整句音色路线</h3><p>V9 {score(v11_base['mean_score'])}，V11 {score(v11['mean_score'])}；声类下降，并新增 1 处违规。</p><p class="bad">原因：主动重规划会改坏 V9 已正确的乐句。</p></section>
    <section class="card"><h3>V12 · 强制重复动机一致</h3><p>首条样本 8 轮后仍没有合法 edit_plan，协议成功率 0/1，立即停止。</p><p class="bad">原因：强制复用多个字段导致工具校验循环。</p></section>
    <section class="card"><h3>V13 · 把一致性降级为平局裁决</h3><p>8 条均分 {score(arm_summary(v13_dev,'v13')['mean_score'])}，其中一条出现 15 处违规。</p><p class="bad">原因：错误字段模板被复制到整段，可演奏性崩溃。</p></section>
    <section class="card"><h3>V14 · 严格字段顺序</h3><p>完成 7 组后，V14 {score(v14['mean_score'])}，V9 {score(v14_base['mean_score'])}；3 胜 1 平 3 负，无法达到 5 胜门槛。</p><p class="bad">原因：格式提示仍会扰动音乐选择，违规数也没有减少。</p></section>
    <section class="card"><h3>V15 · 重写为精简完整提示词</h3><p>完成 {len(v15_pairs)} 组：V15 {v15_mean:.4f}，V9 {v15_base_mean:.4f}，但只有 {v15_wins} 胜 {v15_losses} 负。</p><p class="bad">原因：减少了一次 V9 协议失败，却在多数正常样本上不够稳定。</p></section>
    <section class="card"><h3>V16 · 原版不动，只压缩增量</h3><p>前 {len(v16_pairs)} 组 V16 {v16_mean:.4f}，V9 {v16_base_mean:.4f}；一条出现 15 处违规后提前停止。</p><p class="bad">原因：V9 中看似重复的示例与逐行检查实际上承担了防错作用。</p></section>
  </div>
  {prompt_details('V11（已淘汰）', 'single_stage_phrase_route_v11_addendum.txt')}
  {prompt_details('V12（已淘汰）', 'single_stage_motif_consistency_v12_addendum.txt')}
  {prompt_details('V13（已淘汰）', 'single_stage_motif_tiebreak_v13_addendum.txt')}
  {prompt_details('V14（已淘汰）', 'single_stage_canonical_order_v14_addendum.txt')}
  {prompt_details('V15（已淘汰的完整提示词）', 'single_stage_integrated_v15_prompt.txt')}
  {prompt_details('V16（已淘汰）', 'single_stage_minimal_format_v16_addendum.txt')}

  <h2>两阶段提示为什么也没有超过 V9</h2>
  <div class="grid">
    <section class="card"><h3>方案 A · LLM 自由修格式</h3><p>原草稿 {repair_unsafe['summary']['source_mean']:.4f}，V9 {repair_unsafe['summary']['comparison_mean']:.4f}，修复后只有 {repair_unsafe['summary']['repaired_mean']:.4f}。</p><p class="bad">失败原因：模型把“九徽”改成“9徽”，现有解析器不再识别，造成整段音高和声类崩溃。</p></section>
    <section class="card"><h3>方案 B · 确定性白名单保护</h3><p>只允许新增缺失的“弦”字，其他修改自动回退。修复后 {repair_safe['summary']['repaired_mean']:.4f}，V9 {repair_safe['summary']['comparison_mean']:.4f}，3 胜 5 负。</p><p class="bad">安全但无增益：这些样本原本几乎没有可机械修复的字符，V9 的优势来自更好的方案选择。</p></section>
  </div>
  <div class="plain"><b>经验：</b>音乐决策和格式约束无法简单地完全拆开；但把格式交给另一个无保护的 LLM 又不安全。下一阶段若继续提升，需要对评分器可识别的动作建立结构化中间表示，再由确定性渲染器生成减字字符串，而不是继续叠自然语言规则。</div>

  <h2>评分和测试口径</h2>
  <div class="grid">
    <section class="card"><h3>评分权重</h3><p>音高 45% ＋ 声类 15% ＋ 指法 15% ＋ 装饰 10% ＋ 可演奏性 15%。缺失分项按现有评分器规则重新归一化。</p></section>
    <section class="card"><h3>固定条件</h3><p>GLM-5.3；温度 0；每条最多 8 轮；原版与候选使用同一批样本，并交替运行顺序。</p></section>
    <section class="card"><h3>防止“偷看测试集”</h3><p>最早看过的 6 条测试样本降级为错误分析集。V8/V9/V10 的选择只用训练开发集；V9 冻结后才运行后续 16 条。</p></section>
    <section class="card"><h3>当前限制</h3><p>这仍是 p0001 首乐句的小规模测试，不是 43 首测试曲的完整顺序 benchmark；也没有估计统计置信区间。</p></section>
  </div>

  <h2>脚本怎么用</h2>
  <section class="card"><p>PowerShell 中先把密钥放进当前进程环境变量（不要写进脚本或提交 Git），再运行：</p>
<pre>$env:GLM_API_KEY='你的密钥'
$env:GLM_BASE_URL='https://open.bigmodel.cn/api/anthropic'
$env:GLM_MODEL='glm-5.3'
$env:PYTHONPATH="$PWD\\.gepa_shims;$PWD\\.gepa_packages;$PWD\\guqinagent\\baseline\\gepa\\src;$PWD\\guqinagent"
&amp; .\\.venv-gepa-run\\Scripts\\python.exe prompt_candidate_paired_experiment.py --dataset train --limit 8 --seed your-seed --arms production,v9 --output result.json
&amp; .\\.venv-gepa-run\\Scripts\\python.exe render_prompt_v9_evolution.py</pre>
  <p>实验脚本每完成一条就更新 JSON；渲染脚本读取已有 JSON，重新生成本页面，不会再次调用 API。</p></section>

  <h2>下一步建议</h2>
  <div class="plain">把 V9 当作“进入正式 benchmark 的候选”，不要立刻覆盖生产提示词。下一步应锁死 V9，运行完整的顺序测试（保留 phrase handoff），并提前写好通过门槛。若完整测试仍高于原版，再升级生产版本；若失败，只能回到训练集继续优化，不能根据同一测试集反复改。</div>
  <p class="small">本页展示可复核的实验决策、提示词差异和评分证据，不声称展示模型不可验证的私有思维链。生成文件不包含 API key。</p>
</div></main>
</body></html>"""

    output = ROOT / "gepa_prompt_v9_full_evolution.html"
    output.write_text(page, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
