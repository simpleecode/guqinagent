"""Render the V20-V23 evidence-led prompt-optimization audit."""
from __future__ import annotations
import html, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(n): return json.loads((ROOT/n).read_text(encoding='utf-8'))
def s(x): return f'{x:.4f}'
def main():
    v20=load('prompt_v20_candidate_smoke_result.json')['summary']['v20']
    v21=load('prompt_v21_route_smoke_result.json')['summary']['v21']
    v22=load('prompt_v22_public_constraint_smoke_result.json')['summary']['v22']
    v23=load('prompt_v23_multistring_targeted_result.json')['summary']
    v22partial=load('prompt_v22_candidate_constraint_dev_result.json')['rows']
    partial=''.join(f'<tr><td>{html.escape(r["sample_id"])}</td><td>{r["arm"]}</td><td>{s(r["score"])}</td><td>{r.get("rule_violations")}</td></tr>' for r in v22partial)
    prompt='\n\n'.join((ROOT/n).read_text(encoding='utf-8') for n in ['single_stage_candidate_v20_addendum.txt','single_stage_candidate_v21_addendum.txt','single_stage_candidate_v22_addendum.txt','single_stage_local_multistring_repair_v23_addendum.txt'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V20–V23 提示词优化复盘</title><style>
body{{margin:0;background:#f6f2e9;color:#17221b;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}.hero{{background:linear-gradient(135deg,#143c31,#854d46);color:#fff;padding:52px 24px}}.wrap{{max-width:1000px;margin:auto}}h1{{font-size:clamp(32px,6vw,58px);line-height:1.1;margin:0 0 12px}}main{{padding:0 24px 70px}}h2{{margin-top:44px}}.notice,.take{{background:#fff2d3;border-left:5px solid #af7a22;border-radius:12px;padding:18px;margin:24px 0}}.take{{background:#e5f1e9;border-color:#176b4d}}.timeline{{border-left:3px solid #cdbfa7;padding-left:28px}}.card{{position:relative;background:#fffdf7;border:1px solid #d8d0bd;border-radius:14px;padding:18px;margin:16px 0}}.card:before{{content:"";position:absolute;left:-38px;top:22px;width:16px;height:16px;background:#a84439;border:4px solid #f6f2e9;border-radius:50%}}.tag{{background:#f9e5df;color:#a84439;border-radius:999px;padding:3px 9px;font-size:13px;font-weight:700}}table{{width:100%;border-collapse:collapse;background:#fffdf7}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e7e0d1}}th{{background:#ebe4d4}}details{{background:#fffdf7;border:1px solid #d8d0bd;border-radius:12px;margin:12px 0}}summary{{padding:14px;font-weight:700;cursor:pointer}}pre{{margin:0;padding:15px;white-space:pre-wrap;overflow:auto;background:#18231d;color:#e8efe9;font:13px/1.6 Consolas,"Microsoft YaHei",monospace}}</style><body><header class=hero><div class=wrap><h1>V20 → V23：<br>为什么这轮没有超过 V9</h1><p>真实展示每个假设、实验结果和淘汰理由。目标不是“每次都声称变好”，而是用硬门槛尽快排除错误方向。</p></div></header><main><div class=wrap>
<div class=notice><b>总判断：</b>候选 ID 和公开状态约束能修复某些具体错误，但把它们提升为全局提示词会干扰 V9 已经正确的音乐决策。V20–V23 均未超过 V9，均不进入测试集。</div>
<div class=timeline><section class=card><h3>V20 · 候选 ID 选择 <span class=tag>淘汰</span></h3><p>假设：让程序决定合法弦徽，模型只选 ID，可消除字段拼接错误。最大失败样本分数 {s(v20['mean_score'])}，声类仍为 0。</p><p><b>结论：</b>候选保证音高，不保证模型选对整段音色。</p></section><section class=card><h3>V21 · 模型先声明路线 <span class=tag>淘汰</span></h3><p>假设：模型先选 continuous_harmonic，程序拒绝混入按/散候选。分数 {s(v21['mean_score'])}；模型选择 mixed 后仍能绕开约束。</p><p><b>结论：</b>“要求模型声明路线”仍是自然语言软约束。</p></section><section class=card><h3>V22 · 公开 harmonic_scope 变硬约束 <span class=tag>条件有效、全局淘汰</span></h3><p>定向样本升到 {s(v22['mean_score'])}，声类修复到 {v22['component_means']['tone_type']:.4f}；但全局开发前 5 对仅 1 胜 4 负，且最后一条新增 2 处违规。</p><p><b>结论：</b>公开状态约束可作局部修复，不可替代 V9 全局策略。</p></section><section class=card><h3>V23 · 仅修复明确的滚/历音高 warning <span class=tag>淘汰</span></h3><p>假设：把 V10 的广泛多弦规则缩成只修工具明确标记的一行。定向比较：V9 {s(v23['v9']['mean_score'])}、V23 {s(v23['v23']['mean_score'])}；V23 违规 {v23['v23']['total_rule_violations']}。</p><p><b>结论：</b>模型仍会重做局部音乐方案，单靠提示词无法保证最小编辑。</p></section></div>
<h2>V22 已完成的全局配对证据</h2><table><thead><tr><th>样本</th><th>版本</th><th>分数</th><th>违规</th></tr></thead><tbody>{partial}</tbody></table>
<h2>下一阶段：更高效的正确方向</h2><div class=take><b>保留 V9 作为唯一音乐决策器。</b>先让 V9 正常生成；程序仅检测可确定的格式、候选合法性和公开连续状态；对于命中问题的行，程序生成可验证的最小修复建议，而不是再给第二个模型一个重写整段的提示词。这样避免第二阶段破坏 V9 的正确行。</div>
<h2>四版提示词全文</h2><details><summary>展开 V20–V23 文本</summary><pre>{html.escape(prompt)}</pre></details><p>本页不包含 API key，也不编造模型内部思维过程。</p></div></main></html>'''
    out=ROOT/'gepa_v20_v23_evolution.html';out.write_text(page,encoding='utf-8');print(out)
if __name__=='__main__':main()
