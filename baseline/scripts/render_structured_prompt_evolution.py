"""Render the auditable V17-V19 structured-prompt experiment report."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def score(value: float) -> str:
    return f"{value:.4f}"


def detail(title: str, content: str) -> str:
    return f"<details><summary>{html.escape(title)}</summary><pre>{html.escape(content)}</pre></details>"


def main() -> None:
    paired = load("prompt_v17_structured_dev_paired_2048.json")
    v18 = load("prompt_v18_harmonic_targeted_result.json")
    v19 = load("prompt_v19_harmonic_targeted_result.json")
    grouped = {}
    for row in paired["rows"]:
        grouped.setdefault(row["sample_id"], {})[row["arm"]] = row
    pairs = [(sid, arms["v9"], arms["v17"]) for sid, arms in grouped.items()]
    v9_mean = paired["summary"]["v9"]["mean_score"]
    v17_mean = paired["summary"]["v17"]["mean_score"]
    wins = sum(new["score"] > old["score"] for _, old, new in pairs)
    losses = len(pairs) - wins
    rows = "".join(
        f"<tr><td><code>{html.escape(sid)}</code></td><td>{score(old['score'])}</td>"
        f"<td>{score(new['score'])}</td><td class={'good' if new['score'] > old['score'] else 'bad'}>{new['score']-old['score']:+.4f}</td>"
        f"<td>{old['rule_violations']} → {new['rule_violations']}</td></tr>"
        for sid, old, new in pairs
    )
    failure = next((item for item in pairs if item[0] == "S3JNAahW-p0001"), None)
    source = "\n\n".join((ROOT / name).read_text(encoding="utf-8").strip() for name in [
        "single_stage_structured_v17_addendum.txt",
        "single_stage_structured_v18_addendum.txt",
        "single_stage_structured_v19_addendum.txt",
    ])
    page = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>V17–V19 结构化提示词实验过程</title><style>
:root{{--ink:#17221b;--paper:#f6f2e9;--card:#fffdf7;--line:#d8d0bd;--green:#176b4d;--red:#a84439;--gold:#af7a22}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}.hero{{background:linear-gradient(130deg,#143c31,#315f7d);color:white;padding:54px 24px}}.wrap{{max-width:1050px;margin:auto}}h1{{font-size:clamp(32px,6vw,58px);line-height:1.12;margin:0 0 12px}}h2{{margin:46px 0 16px;font-size:28px}}.lead{{font-size:18px;max-width:800px}}main{{padding:0 24px 70px}}.notice,.plain{{padding:18px 20px;border-radius:14px;background:#fff2d3;border-left:5px solid var(--gold);margin:25px 0}}.plain{{background:#e5f1e9;border-color:var(--green)}}.stats,.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.box{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}}.box strong{{display:block;font-size:27px;color:var(--green)}}.timeline{{border-left:3px solid #c7bca5;margin:18px 0;padding-left:28px}}.step{{position:relative;margin:0 0 17px;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}.step:before{{content:"";position:absolute;width:15px;height:15px;border-radius:50%;background:#315f7d;left:-38px;top:24px;border:4px solid var(--paper)}}.step.fail:before{{background:var(--red)}}.step h3{{margin:0}}.tag{{display:inline-block;font-size:13px;font-weight:700;border-radius:999px;padding:3px 9px;background:#dff2e8;color:var(--green)}}.tag.fail{{background:#f9e5df;color:var(--red)}}table{{width:100%;border-collapse:collapse;background:var(--card)}}th,td{{padding:10px;border-bottom:1px solid #e7e0d1;text-align:left}}th{{background:#ebe4d4}}.good{{color:var(--green);font-weight:700}}.bad{{color:var(--red);font-weight:700}}details{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:12px 0}}summary{{padding:14px 16px;cursor:pointer;font-weight:700;color:#235d4b}}pre{{white-space:pre-wrap;overflow:auto;margin:0;padding:16px;background:#18231d;color:#e8efe9;font:13px/1.6 Consolas,"Microsoft YaHei",monospace}}@media(max-width:700px){{.stats,.grid{{grid-template-columns:1fr 1fr}}}}@media(max-width:450px){{.stats,.grid{{grid-template-columns:1fr}}}}
</style><body><header class=hero><div class=wrap><div class=tag style=\"background:#ffffff22;color:#fff\">STRUCTURED EXPERIMENT · V17 → V19</div><h1>结构化方案：<br>为什么暂时没有超过 V9</h1><p class=lead>这一页记录模型不再直接拼减字字符串、改为提交结构字段后的三轮实验。它展示真实分数、失败原因和停止条件，不把“架构更漂亮”误说成“效果更高”。</p></div></header><main><div class=wrap>
<div class=notice><b>结论：</b>V17 的 8 条开发集均分 {score(v17_mean)}，略低于 V9 的 {score(v9_mean)}（{v17_mean-v9_mean:+.4f}）。虽然 V17 有 {wins} 胜 {losses} 负、违规从 2 降到 0，但没有通过“均分必须超过 V9”的门槛，因此 V17/V18/V19 都没有使用测试集。</div>
<div class=stats><div class=box><span>V9 均分</span><strong>{score(v9_mean)}</strong></div><div class=box><span>V17 均分</span><strong>{score(v17_mean)}</strong></div><div class=box><span>逐条结果</span><strong>{wins} 胜 {losses} 负</strong></div><div class=box><span>协议成功</span><strong>8 / 8</strong></div></div>
<h2>这次到底改了什么</h2><div class=plain><b>原路径：</b>模型直接提交“减字文字”。<br><b>结构化路径：</b>模型提交按/散/泛、弦、徽、左右手指、双音成员、续作等字段 → 程序确定性渲染减字 → 原有音高与可演奏性校验器评分。</div>
<div class=timeline><section class=step><h3>V17 · 第一次结构化提交 <span class=tag>协议成功</span></h3><p>新增 <code>edit_structured_plan</code>：single、compound、continuation、hidden 四种动作；程序渲染成标准减字。结果是 8/8 合法、音高 1.0、零违规，但均分低 V9 0.0005。</p><p><b>收获：</b>字符串格式和违规确实更稳定。<br><b>问题：</b>模型没有稳定做出整段音色状态选择。</p></section><section class=step fail><h3>V18 · 禁止自由文本回退 <span class=tag fail>未提升</span></h3><p>结构化实验不再提供 edit_plan，确保模型不能绕开字段接口。针对连续泛音失败样本，V18 得分 0.7222，仍未修复声类。</p></section><section class=step fail><h3>V19 · 加入泛音区间状态字段 <span class=tag fail>未提升</span></h3><p>新增 <code>harmonic_state = start / inside / end</code>，程序能渲染泛起、泛音、泛止；提示词要求参考公开的 harmonic_scope 状态。但同一失败样本仍为 0.7222、声类 0.0。</p><p><b>真实结论：</b>字段能表达泛音边界，不等于模型会正确选择泛音。问题是音乐判断，不是渲染器。</p></section></div>
<h2>8 条配对评分</h2><table><thead><tr><th>样本</th><th>V9</th><th>V17</th><th>差值</th><th>违规 V9 → V17</th></tr></thead><tbody>{rows}</tbody></table>
<h2>最大失败案例：为什么不能进入测试集</h2><div class=grid><section class=box><h3>样本</h3><p><code>S3JNAahW-p0001</code></p><p>V9：{score(failure[1]['score'])}<br>V17：{score(failure[2]['score'])}</p></section><section class=box><h3>评分变化</h3><p>音高仍为 1.0、违规仍为 0；但按/散/泛声类从 {failure[1]['components']['tone_type']:.4f} 降为 {failure[2]['components']['tone_type']:.4f}。</p></section><section class=box><h3>根因</h3><p>该段应维持连续泛音。结构化模型却选择了散音和按音；程序忠实渲染了错误的音乐选择。</p></section><section class=box><h3>停止理由</h3><p>继续只靠自然语言催模型“选泛音”没有带来改善。为了不污染测试集，停止候选迭代。</p></section></div>
<h2>代码层的可复核改动</h2><div class=grid><section class=box><h3>工具 schema</h3><p>新增结构化工具与字段定义；普通生产工具未被替换。</p></section><section class=box><h3>确定性渲染</h3><p>字段转换为单音、双音、续作、隐藏和泛音边界字符串，再走原校验器。</p></section><section class=box><h3>Rollout</h3><p>实验 arm 才开启结构化工具；V9 与生产路径保持不变。</p></section><section class=box><h3>下一阶段</h3><p>程序提供合法候选 ID 和公开状态约束，模型只在候选中做音乐选择。</p></section></div>
<h2>提示词与原始结果</h2>{detail('展开 V17 / V18 / V19 提示词', source)}{detail('展开 V17 的完整 8 条 JSON 结果', json.dumps(paired, ensure_ascii=False, indent=2))}{detail('展开 V18 定向结果', json.dumps(v18, ensure_ascii=False, indent=2))}{detail('展开 V19 定向结果', json.dumps(v19, ensure_ascii=False, indent=2))}
<p style=\"color:#637067;font-size:14px\">本页只展示可验证的提示词、代码机制和实验结果；不包含 API key，也不伪造模型内部思维过程。</p></div></main></body></html>"""
    output = ROOT / "gepa_structured_v17_v19_evolution.html"
    output.write_text(page, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
