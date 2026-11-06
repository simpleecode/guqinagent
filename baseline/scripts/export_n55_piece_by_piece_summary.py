#!/usr/bin/env python3
"""Export a standalone, whole-score-by-whole-score N55 annotation summary."""
from __future__ import annotations

import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "guqinagent"
OUT = REPO / "baseline/runs/N_gepa16_from_original/test50_top3"
sys.path[:0] = [str(REPO)]

from baseline.scripts.prompt_data_sources import TEST50  # noqa: E402
from train.scripts.score_agent_predictions import canonical  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def phrase_number(phrase_id: str) -> int:
    try:
        return int(str(phrase_id).removeprefix("p"))
    except ValueError:
        return 10**9


def main() -> int:
    rows = json.loads((OUT / "rows_N55.json").read_text(encoding="utf-8")).get("rows") or []
    references = {row["sample_id"]: row for row in read_jsonl(TEST50 / "evaluation_pairs_test.jsonl")}
    pieces: dict[str, dict] = defaultdict(lambda: {
        "phrases": [], "generated": Counter(), "counts": Counter(),
        "predicted_nonempty": 0, "reference_nonempty": 0,
        "exact_cells": 0, "rows_compared": 0, "nonempty_hits": 0,
        "reference_nonempty_metric": 0, "protocol_ok": 0,
    })
    for row in rows:
        sample_id = str(row["sample_id"])
        reference_row = references.get(sample_id)
        if reference_row is None:
            continue
        score_key = str(row.get("score_key") or reference_row.get("score_key") or "未知曲谱")
        phrase_id = str(row.get("phrase_id") or reference_row.get("phrase_id") or "")
        piece = pieces[score_key]
        reference = {int(item["source_index"]): str(item.get("jianzi_text") or item.get("text") or "")
                     for item in (reference_row.get("reference") or {}).get("actions") or []}
        prediction = {int(index): str(text or "")
                      for index, text in (row.get("prediction") or {}).get("jianzi_rows") or []}
        for text in reference.values():
            piece["reference_nonempty"] += int(bool(canonical(text)))
        for text in prediction.values():
            if canonical(text):
                piece["predicted_nonempty"] += 1
                piece["generated"][text] += 1
        for index in set(reference) | set(prediction):
            ref, pred = reference.get(index, ""), prediction.get(index, "")
            if canonical(ref) == canonical(pred):
                piece["counts"]["完全一致"] += 1
            elif ref and not pred:
                piece["counts"]["漏写"] += 1
            elif pred and not ref:
                piece["counts"]["多写"] += 1
            else:
                piece["counts"]["文字不同"] += 1
        metric = row.get("teacher_score") or {}
        for name in ("exact_cells", "rows_compared", "nonempty_hits", "reference_nonempty"):
            piece[f"{name}_metric" if name == "reference_nonempty" else name] += int(metric.get(name) or 0)
        piece["protocol_ok"] += int(bool(row.get("protocol_ok")))
        piece["phrases"].append((phrase_id, metric, bool(row.get("protocol_ok"))))

    nav = "".join(f"<a href='#piece-{html.escape(key)}'>{html.escape(key)}</a>" for key in sorted(pieces))
    sections = []
    for score_key in sorted(pieces):
        piece = pieces[score_key]
        phrase_rows = sorted(piece["phrases"], key=lambda item: phrase_number(item[0]))
        exact = piece["exact_cells"] / piece["rows_compared"] if piece["rows_compared"] else None
        recall = (piece["nonempty_hits"] / piece["reference_nonempty_metric"]
                  if piece["reference_nonempty_metric"] else None)
        protocol = piece["protocol_ok"] / len(phrase_rows) if phrase_rows else None
        cards = [
            ("乐句", str(len(phrase_rows))), ("逐格精确率", pct(exact)),
            ("非空召回", pct(recall)), ("协议合规", pct(protocol)),
            ("N55 非空减字", f"{piece['predicted_nonempty']:,}"),
            ("标注非空减字", f"{piece['reference_nonempty']:,}"),
            ("完全一致", f"{piece['counts']['完全一致']:,}"),
            ("文字不同", f"{piece['counts']['文字不同']:,}"),
            ("多写 / 漏写", f"{piece['counts']['多写']:,} / {piece['counts']['漏写']:,}"),
        ]
        card_html = "".join(f"<div class='card'><b>{html.escape(name)}</b><strong>{html.escape(value)}</strong></div>"
                            for name, value in cards)
        top_html = "".join(f"<tr><td>{html.escape(text)}</td><td>{count}</td></tr>"
                               for text, count in piece["generated"].most_common(10)) or "<tr><td colspan='2'>无非空生成</td></tr>"
        phrase_html = "".join(
            f"<tr><td>{html.escape(phrase_id)}</td><td>{pct(metric.get('exact_cell_rate'))}</td>"
            f"<td>{pct(metric.get('nonempty_recall'))}</td><td>{'✓' if ok else '✗'}</td></tr>"
            for phrase_id, metric, ok in phrase_rows
        )
        sections.append(f"""<section id='piece-{html.escape(score_key)}'><h2>{html.escape(score_key)}</h2>
<p>这一节把整首曲谱的所有乐句汇总在一起。文字不同不代表音高一定错；详细的逐音高核对请查看全量报告。</p>
<div class='cards'>{card_html}</div>
<h3>N55 在本曲最常生成的减字</h3><table><tr><th>减字</th><th>次数</th></tr>{top_html}</table>
<h3>本曲各乐句评分</h3><table><tr><th>乐句</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th></tr>{phrase_html}</table></section>""")
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>N55 test50 按曲谱生成标注总结</title><style>
body{{font:15px/1.65 system-ui,'Microsoft YaHei';max-width:1350px;margin:28px auto;padding:0 18px;background:#f4f6fa;color:#263146}}section{{background:#fff;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:16px 0}}h1,h2,h3{{margin-top:0}}nav{{display:flex;gap:8px;flex-wrap:wrap}}nav a{{padding:4px 8px;background:#eaf2ff;border-radius:6px;color:#174a8b;text-decoration:none}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:9px}}.card{{background:#f8fafc;border:1px solid #dce3ed;border-radius:8px;padding:9px}}.card b{{display:block;font-size:12px;color:#59677a}}.card strong{{display:block;font-size:20px;color:#12624d;margin-top:3px}}table{{border-collapse:collapse;width:100%;margin:8px 0}}td,th{{padding:7px;border-bottom:1px solid #e1e6ee;text-align:left}}th{{background:#f5f8fc}}</style>
<h1>N55 · test50 按整首曲谱的生成标注总结</h1><section><p>共 {len(rows)} 句、{len(pieces)} 首曲谱。此文件是稳定静态报告，按曲谱而非按最近完成顺序组织。</p><nav>{nav}</nav></section>{''.join(sections)}</html>"""
    target = OUT / "N55_test50_PIECE_BY_PIECE_SUMMARY.html"
    target.write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(target), "pieces": len(pieces), "phrases": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
