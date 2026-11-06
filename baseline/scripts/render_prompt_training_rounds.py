"""Render the manual GLM prompt-optimization rounds as a readable HTML report."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "glm_prompt_optimization_rounds.html"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def mean(rows: list[dict], key: str = "score") -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def row_map(data: dict) -> dict[str, dict]:
    return {row["sample_id"]: row for row in data["rows"]}


def build() -> str:
    baseline = load("glm_dataset_pilot_result.json")
    v3 = load("glm_prompt_v3_train_result.json")
    v4 = load("glm_prompt_v4_train_result.json")
    v5 = load("glm_prompt_v5_train_result.json")
    baseline_rows = [row for row in baseline["rows"] if row["prompt"] == "production_seed_plus_handbook"]
    v2_rows = [row for row in baseline["rows"] if row["prompt"] == "gpt_curated_v2_plus_handbook"]
    rounds = [
        ("原生产版", baseline_rows, "作为比较起点", "仓库当前 single_stage 提示词＋固定手册。"),
        ("v2 · 全面重写", v2_rows, "失败", "改动范围过大，强调写满和复杂技法；一条样本出现大量可演奏性违规。"),
        ("v3 · 保守简化", v3["rows"], "接近但未超过", "消除了违规，但“优先简单”误伤了按/散/泛音色选择。"),
        ("v4 · 收窄护栏", v4["rows"], "失败", "模型产生“散勾六”等缩写，执行器接受但评分器无法解析音高。"),
        ("v5 · 可解析格式", v5["rows"], "当前候选", "不改变音乐判断，只要求弦、徽、音色字段写成解析器能识别的形式。"),
    ]
    cards = []
    base_mean = mean(baseline_rows)
    for index, (name, rows, status, reason) in enumerate(rounds):
        value = mean(rows)
        violations = sum(int(row.get("rule_violations") or 0) for row in rows)
        cards.append(f"""<article class="round {'winner' if name.startswith('v5') else ''}">
<div class="round-no">{index}</div><div><p class="label">{esc(status)}</p><h3>{esc(name)}</h3>
<p>{esc(reason)}</p><div class="score"><strong>{fmt(value)}</strong><span>三条平均分</span></div>
<p class="delta">相对生产版：{fmt(value - base_mean)}{f'；记录到的违规数：{violations}' if index >= 2 else ''}</p></div></article>""")
    sample_ids = sorted(row["sample_id"] for row in baseline_rows)
    maps = [row_map({"rows": rows}) for _, rows, _, _ in rounds]
    sample_table = []
    for sample_id in sample_ids:
        values = [mapping.get(sample_id, {}).get("score") for mapping in maps]
        sample_table.append("<tr><td>" + esc(sample_id) + "</td>" + "".join(f"<td>{fmt(value)}</td>" for value in values) + "</tr>")
    v5_mean = mean(v5["rows"])
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM 提示词优化训练记录</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f1e8;color:#26322d;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1080px;margin:auto;padding:30px 20px 70px}}
header{{background:linear-gradient(135deg,#163e35,#36715c);color:white;border-radius:24px;padding:38px;box-shadow:0 18px 45px #163e3528}}header h1{{font-size:clamp(30px,5vw,46px);line-height:1.2;margin:6px 0}}header p{{color:#e2f0e9;max-width:760px}}section{{background:#fffdf8;border:1px solid #ded8ca;border-radius:18px;padding:25px 28px;margin:20px 0}}h2{{margin-top:0}}.answer{{font-size:19px;background:#e6f4ec;border-left:5px solid #27805d;padding:16px;border-radius:10px}}.warning{{background:#fff4cf;border-left-color:#c18b18}}
.rounds{{display:grid;gap:13px}}.round{{display:grid;grid-template-columns:54px 1fr;gap:16px;border:1px solid #ded8ca;border-radius:15px;padding:18px;background:#fff}}.round.winner{{border:2px solid #2b8a64;background:#f2faf6}}.round-no{{width:44px;height:44px;border-radius:50%;display:grid;place-items:center;background:#284f43;color:#fff;font-weight:800;font-size:20px}}.label{{margin:0;color:#718078;font-size:13px;font-weight:800;letter-spacing:.08em}}h3{{margin:0;font-size:22px}}.score{{display:flex;align-items:baseline;gap:10px}}.score strong{{font-size:31px;color:#176247}}.score span,.delta{{color:#66746d}}
table{{border-collapse:collapse;width:100%;overflow:auto}}th,td{{padding:10px;border-bottom:1px solid #e1dcd2;text-align:left}}th{{font-size:13px;color:#66746d}}details{{border-top:1px solid #e1dcd2;margin-top:13px;padding-top:13px}}summary{{cursor:pointer;color:#1d674f;font-weight:800}}pre{{white-space:pre-wrap;background:#202a26;color:#eaf4ee;border-radius:10px;padding:16px;max-height:520px;overflow:auto}}code{{background:#eee9df;padding:2px 5px;border-radius:4px}}@media(max-width:650px){{header,section{{padding:21px}}.table-wrap{{overflow:auto}}}}
</style></head><body><main><header><p>TRAINING DIAGNOSIS</p><h1>提示词为什么变差，又是怎样修回来的？</h1><p>同一个 GLM 5.3、同样的 3 条训练样本和同一评分方式，只改变提示词。</p></header>
<section><h2>结论先看这里</h2><p class="answer">v2 下降的直接原因是提示词鼓励模型写得过满、过复杂，造成额外演奏违规；后续又发现，过度强调“简单”会损失音色判断，而不规范的减字缩写会让评分器无法解析。v5 最终只约束输出格式，平均分达到 <b>{fmt(v5_mean)}</b>，略高于生产版 <b>{fmt(base_mean)}</b>。</p>
<p class="answer warning">提升只有 <b>{fmt(v5_mean-base_mean)}</b>，3 条样本远不足以证明稳定提升。v5 可以进入扩大验证，但现在不能替换生产提示词。</p></section>
<section><h2>每一轮发生了什么</h2><div class="rounds">{''.join(cards)}</div></section>
<section><h2>逐条样本对比</h2><div class="table-wrap"><table><thead><tr><th>样本</th>{''.join(f'<th>{esc(item[0])}</th>' for item in rounds)}</tr></thead><tbody>{''.join(sample_table)}</tbody></table></div></section>
<section><h2>根因拆解</h2><ol><li><b>策略改动太大：</b>v2 替换了原提示词的行为习惯，模型为了覆盖和丰富而增加动作。</li><li><b>评分口径会放大违规：</b>可演奏性使用违规事件除以可评音高事件，分母很小时可产生负分；但违规本身确实由 v2 新增。</li><li><b>“简单优先”不是普遍正确：</b>v3 虽然消除了违规，却把部分泛音或散音改成普通按音，音色分下降。</li><li><b>执行成功不等于可评分：</b>v4 的“散勾六”等文本能通过 edit_plan，却缺少解析器要求的“弦”等字段，导致音高不可评。</li><li><b>最有效的是边界清楚的小改动：</b>v5 保留生产提示词的音乐决策，只增加可解析输出契约。</li></ol></section>
<section><h2>下一步怎么做</h2><p>冻结 v5，不再根据这 3 条继续调参；从训练集另取未参与本轮设计的样本做扩大验证。只有在配对均分、失败率和违规数都稳定不差于生产版时，才进入正式验证集。</p>
<details><summary>查看 v5 新增的提示词</summary><pre>{esc((ROOT/'single_stage_parseable_v5_addendum.txt').read_text(encoding='utf-8').strip())}</pre></details></section>
</main></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)
