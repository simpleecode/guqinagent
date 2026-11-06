"""Render the recorded GLM train-pilot results as an offline HTML audit.

Usage: python render_glm_dataset_pilot.py [result.json] [report.html]
Only recorded inputs, scores, and decisions are shown; no hidden reasoning is inferred.
"""
from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "glm_dataset_pilot_result.json"
TARGET = ROOT / "glm_dataset_pilot_report.html"
LABELS = {
    "production_seed_plus_handbook": "当前生产提示词＋固定手册",
    "gpt_curated_v2_plus_handbook": "优化候选 v2＋同一手册",
}


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def numeric(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def render(data: dict) -> str:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data["rows"]:
        grouped[row["prompt"]].append(row)
    averages = {name: sum(row["score"] for row in rows) / len(rows)
                for name, rows in grouped.items() if rows}
    baseline = averages.get("production_seed_plus_handbook")
    candidate = averages.get("gpt_curated_v2_plus_handbook")
    delta = candidate - baseline if baseline is not None and candidate is not None else None
    decision = ("保留当前生产提示词；该候选没有通过本次小样本对照。"
                if delta is not None and delta <= 0 else
                "本次小样本未否定候选，但不足以支持替换生产提示词。")
    cards = []
    for index, sample_id in enumerate(data["sample_ids"], 1):
        rows = [row for row in data["rows"] if row["sample_id"] == sample_id]
        details = []
        for row in rows:
            components = " · ".join(f"{escape(k)} {numeric(v)}"
                                    for k, v in row["components"].items())
            details.append(
                f"<div class='result'><h4>{escape(LABELS.get(row['prompt'], row['prompt']))}: "
                f"{numeric(row['score'])}</h4><p>协议通过：{escape(row['protocol_ok'])}；"
                f"模型调用：{escape(row['actor_calls'])}；工具轮数：{escape(row.get('rounds', '—'))}</p>"
                f"<p>{components}</p>"
                f"{f'<p>错误：{escape(row["error"])}</p>' if row.get('error') else ''}</div>"
            )
        cards.append(f"<article><h3>第 {index} 条：{escape(sample_id)}</h3>{''.join(details)}</article>")
    usage = data.get("usage") or {}
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>GLM 数据集试跑审计</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1080px;margin:auto;padding:2rem;background:#f4f7fb;color:#1d2939}}
section,article{{background:white;border:1px solid #d8e0e9;border-radius:12px;padding:1rem 1.4rem;margin:1rem 0}}
.result{{border-left:3px solid #698eb9;padding:.15rem .9rem;margin:.8rem 0}}h4{{margin:.35rem 0}}p{{margin:.5rem 0}}
.warn{{background:#fff4db;border:1px solid #e7ca80;border-radius:8px;padding:.8rem}}code{{overflow-wrap:anywhere}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:.45rem;text-align:left;border-bottom:1px solid #dde3ec}}</style></head><body>
<h1>GLM 5.3 数据集试跑</h1>
<p class="warn">这是从训练集中选取 3 条首乐句的探索性对照，不是完整数据集或正式 baseline；
候选提示词并未通过 GEPA 自动优化。本页展示可核查的操作和结果，不展示或编造模型的隐藏思维链。</p>
<section><h2>过程与口径</h2><ol>
<li>从 <code>inferred_trajectories_train.jsonl</code> 选取 3 条较短的 <code>p0001</code> 首乐句，避免跨乐句历史状态混入。</li>
<li>用相同的 GLM 模型、固定手册、温度 0、最多 8 轮和 1 次尝试，依次运行当前生产提示词及优化候选 v2。</li>
<li>通过实际古琴工具执行，再用封存参考答案计算音高、音色、指法、装饰音与可演奏性分项。</li>
<li>比较逐条得分和均分；只作试跑决策，不以这 3 条替代正式验证集。</li></ol>
<p>模型：{escape(data['model'])}；输入 tokens：{escape(usage.get('actor_tokens_in', '—'))}；输出 tokens：{escape(usage.get('actor_tokens_out', '—'))}；调用：{escape(usage.get('actor_calls', '—'))}</p>
</section>
<section><h2>结果概览</h2><table><thead><tr><th>提示词</th><th>样本数</th><th>平均分</th></tr></thead><tbody>
{''.join(f'<tr><td>{escape(LABELS.get(name, name))}</td><td>{len(grouped[name])}</td><td>{numeric(score)}</td></tr>' for name, score in averages.items())}
</tbody></table><p>候选相对生产版：{numeric(delta)}；结论：{escape(decision)}</p>
<p class="warn">可演奏性可能出现负值：当前评分把违规事件率除以可评音高事件数，分母较小时可能超过 1。正式比较前应审查这一口径。</p></section>
<section><h2>逐条运行记录</h2>{''.join(cards)}</section>
<section><h2>下一步</h2><p>先修订并冻结评分口径，再在固定验证集上运行生产版 baseline；
候选应在相同样本、相同模型参数和相同手册下配对比较，通过后才考虑封存测试集。</p></section>
</body></html>"""


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else TARGET
    target.write_text(render(json.loads(source.read_text(encoding="utf-8"))), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
