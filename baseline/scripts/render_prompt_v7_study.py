"""Render the evidence trail for the v5-v7 prompt study."""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "prompt_v7_evidence_report.html"


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def f(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def summary_rows(result: dict) -> str:
    return "".join(
        f"<tr><td>{esc(name)}</td><td>{data['protocol_successes']}/{data['n']}</td>"
        f"<td>{f(data['mean_score'])}</td><td>{data['total_rule_violations']}</td>"
        f"<td>{f((data.get('component_means') or {}).get('pitch'))}</td>"
        f"<td>{f((data.get('component_means') or {}).get('tone_type'))}</td>"
        f"<td>{f((data.get('component_means') or {}).get('fingering'))}</td></tr>"
        for name, data in result["summary"].items()
    )


def paired_rows(result: dict) -> str:
    by_id: dict[str, dict[str, dict]] = {}
    for row in result["rows"]:
        by_id.setdefault(row["sample_id"], {})[row["arm"]] = row
    rows = []
    for sample_id, arms in by_id.items():
        production = arms.get("production")
        candidate = arms.get("v7")
        if not production or not candidate:
            continue
        delta = candidate["score"] - production["score"]
        rows.append(f"<tr><td>{esc(sample_id)}</td><td>{f(production['score'])}</td>"
                    f"<td>{f(candidate['score'])}</td><td class='{'up' if delta > 0 else 'down' if delta < 0 else ''}'>{f(delta)}</td></tr>")
    return "".join(rows)


def main() -> None:
    first_dev = load("prompt_v6_dev_paired_result.json")
    v7_diag = load("prompt_v7_diagnostic_result.json")
    fresh_dev = load("prompt_v7_fresh_dev_result.json")
    test = load("prompt_v7_test_pilot_result.json")
    prompt = (ROOT / "single_stage_parseable_compound_v7_addendum.txt").read_text(encoding="utf-8").strip()
    fresh_delta = fresh_dev["summary"]["v7"]["paired_mean_delta_vs_production"]
    test_delta = test["summary"]["v7"]["paired_mean_delta_vs_production"]
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>提示词 v7 优化证据报告</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f5f1e8;color:#26322d;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1100px;margin:auto;padding:28px 20px 70px}}header{{background:linear-gradient(135deg,#173f36,#326c58);color:#fff;border-radius:24px;padding:38px;box-shadow:0 16px 42px #173f3626}}header h1{{font-size:clamp(30px,5vw,46px);line-height:1.18;margin:6px 0}}header p{{color:#e4f0ea;max-width:800px}}section{{background:#fffdf8;border:1px solid #ded8ca;border-radius:18px;padding:25px 28px;margin:20px 0}}h2{{margin-top:0}}.verdict{{font-size:20px;background:#fff0d4;border-left:6px solid #c47c1d;padding:17px;border-radius:10px}}.flow{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.flow div{{background:#eef5f1;border-radius:13px;padding:15px}}.flow b{{display:block;color:#1d654f;font-size:18px}}.metric-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{background:#f1f6f3;border-radius:13px;padding:15px}}.metric strong{{display:block;font-size:28px;color:#1b604a}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #e1dcd2;text-align:left}}th{{font-size:13px;color:#65736c}}.table-wrap{{overflow:auto}}.up{{color:#13794f;font-weight:700}}.down{{color:#b3453f;font-weight:700}}.callout{{background:#e9f4ee;border-left:5px solid #2c8563;padding:14px;border-radius:9px}}.danger{{background:#fdebe8;border-left-color:#bc514b}}details{{border-top:1px solid #ddd6c9;padding-top:12px;margin-top:14px}}summary{{cursor:pointer;color:#1d654f;font-weight:800}}pre{{white-space:pre-wrap;background:#202a26;color:#eef5f1;padding:16px;border-radius:10px;max-height:560px;overflow:auto}}code{{background:#eee9df;padding:2px 5px;border-radius:4px}}@media(max-width:760px){{header,section{{padding:21px}}.flow,.metric-grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><p>EVIDENCE-BASED PROMPT OPTIMIZATION</p><h1>v7 为什么在开发集成功，却没有通过测试？</h1><p>从评分标准、问题证据、最小修改、确认集到冻结测试，完整保留每一步的判断依据。</p></header>
<section><h2>最终结论</h2><p class="verdict"><b>保留原生产提示词，暂不采用 v7。</b><br>v7 在全新开发确认集提升 {f(fresh_delta)}，但在冻结测试 pilot 上变化为 {f(test_delta)}，3 胜 3 负，不能证明泛化提升。</p>
<div class="metric-grid"><div class="metric"><span>测试生产版</span><strong>{f(test['summary']['production']['mean_score'])}</strong></div><div class="metric"><span>测试 v7</span><strong>{f(test['summary']['v7']['mean_score'])}</strong></div><div class="metric"><span>测试配对差</span><strong>{f(test_delta)}</strong></div></div></section>
<section><h2>这套优化流程</h2><div class="flow"><div><b>1. 找问题</b>从预测与确定性违规记录定位，不凭感觉改提示词。</div><div><b>2. 最小修改</b>只修复有证据的问题，避免一次重写全部策略。</div><div><b>3. 新开发集确认</b>排除参与调试的样本，按预设门槛判断。</div><div><b>4. 冻结后测试</b>测试结果只用于报告，不再反向修改提示词。</div></div></section>
<section><h2>评分口径</h2><p>沿用仓库既有组合分：音高 45%、音色 15%、左右手指法均值 15%、装饰 10%、可演奏性 15%。候选门槛在看结果前固定：协议成功率不降、违规不增、音高下降不超过 0.005、配对总分至少提升 0.005。</p></section>
<section><h2>第一轮开发集：为什么 v5/v6 被拒绝</h2><div class="table-wrap"><table><thead><tr><th>版本</th><th>协议成功</th><th>均分</th><th>违规</th><th>音高</th><th>音色</th><th>指法</th></tr></thead><tbody>{summary_rows(first_dev)}</tbody></table></div>
<p class="callout danger">v5 总分提高但违规增加；v6 总分提高更多，但音高下降且违规更多。两者都没有通过安全门槛。</p></section>
<section><h2>v7 的修改依据</h2><ul><li>把双音写成解析器支持的标准格式：<code>撮（按音成员＋散音成员）</code>，按音在前。</li><li>禁止两根弦直接并列，避免 <code>ambiguous_multiple_strings</code>。</li><li>禁止把装饰音组仅凭外形压成“散历456弦”等未经逐音验证的写法。</li><li>保留 v5 已验证有效的弦、徽、音色显式字段。</li></ul>
<p>同批诊断复核：v7 均分 {f(v7_diag['summary']['v7']['mean_score'])}，音高 {f(v7_diag['summary']['v7']['component_means']['pitch'])}，违规 {v7_diag['summary']['v7']['total_rule_violations']}。</p><details><summary>查看冻结的 v7 增量提示词</summary><pre>{esc(prompt)}</pre></details></section>
<section><h2>全新开发确认集</h2><div class="table-wrap"><table><thead><tr><th>版本</th><th>协议成功</th><th>均分</th><th>违规</th><th>音高</th><th>音色</th><th>指法</th></tr></thead><tbody>{summary_rows(fresh_dev)}</tbody></table></div>
<p class="callout">v7 6/6 成功，生产版 5/6；v7 5 胜 1 平，配对提升 {f(fresh_delta)}，且音高和违规门槛均通过，因此获准进入冻结测试。</p></section>
<section><h2>冻结测试 pilot</h2><div class="table-wrap"><table><thead><tr><th>测试样本</th><th>生产版</th><th>v7</th><th>差值</th></tr></thead><tbody>{paired_rows(test)}</tbody></table></div>
<p class="callout danger">v7 测试均分 {f(test['summary']['v7']['mean_score'])}，生产版 {f(test['summary']['production']['mean_score'])}，配对差 {f(test_delta)}。虽然两者音高均为 1.0、违规均为 0，但 v7 音色均值更低，抵消了指法与装饰收益。</p></section>
<section><h2>方法本身还有哪些问题？</h2><ol><li><b>测试规模仍小：</b>本次只有 6 条首乐句，是 test pilot，不是 1379 条完整测试集。</li><li><b>只测 p0001：</b>规避了前缀状态污染，但没有覆盖跨乐句 handoff；正式评测必须按整曲顺序运行。</li><li><b>单次调用有波动：</b>即使温度为 0，远端模型和工具循环仍可能产生不同路径；正式开发比较应至少重复两次或扩大样本。</li><li><b>可演奏性公式可能出现负值：</b>违规事件数可能大于可评音高事件数。当前为保持与既有标准一致没有改分，但正式论文应报告这一限制。</li><li><b>开发集协议失败会放大均分：</b>v7 的开发提升中包含修复一次生产版 0 分；应同时报告成功率和仅成功样本指标。</li></ol></section>
<section><h2>下一轮合理方案</h2><p>不再查看这 6 条测试样本来生成 v8。下一轮应重新从训练半边建立更大的固定 train/val split，优化目标加入“格式可解析率、协议成功率、音高、违规数”的多门槛；在至少 30–50 条、包含非首乐句的开发集上确认。提示词彻底冻结后，再使用仓库的 <code>run_final_eval.py</code> 对完整 43 曲 / 1379 乐句顺序评测一次。</p></section>
</main></body></html>"""
    OUT.write_text(document, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
