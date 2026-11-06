#!/usr/bin/env python3
"""Export whole-score original annotations alongside N55 generated annotations."""
from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "guqinagent"
OUT = REPO / "baseline/runs/N_gepa16_from_original/test50_top3"
sys.path[:0] = [str(REPO)]

from baseline.scripts.prompt_data_sources import TEST50  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def phrase_number(value: str) -> int:
    try:
        return int(str(value).removeprefix("p"))
    except ValueError:
        return 10**9


def phrase_text(items: dict[int, str]) -> str:
    """Keep every source index, including empty cells, so both columns align."""
    return "  |  ".join(f"{index}: {text if text else '（空）'}" for index, text in sorted(items.items()))


def main() -> int:
    result = json.loads((OUT / "rows_N55.json").read_text(encoding="utf-8"))
    rows = {str(row["sample_id"]): row for row in result.get("rows") or []}
    public = {str(row["sample_id"]): row for row in read_jsonl(TEST50 / "test_runtime.jsonl")}
    references = {str(row["sample_id"]): row for row in read_jsonl(TEST50 / "evaluation_pairs_test.jsonl")}
    pieces: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for sample_id, public_row in public.items():
        reference_row = references.get(sample_id)
        output_row = rows.get(sample_id)
        if reference_row is None or output_row is None:
            continue
        score_key = str(public_row["score_key"])
        phrase_id = str(public_row["phrase_id"])
        original = {int(item["source_index"]): str(item.get("jianzi_text") or item.get("text") or "")
                    for item in (reference_row.get("reference") or {}).get("actions") or []}
        generated = {int(index): str(text or "")
                     for index, text in (output_row.get("prediction") or {}).get("jianzi_rows") or []}
        # Include missing generated indices as blanks and unexpected generated
        # indices as extra cells, so the two complete-score columns stay aligned.
        all_indices = set(original) | set(generated)
        original = {index: original.get(index, "") for index in all_indices}
        generated = {index: generated.get(index, "") for index in all_indices}
        pieces[score_key].append((phrase_id, phrase_text(original), phrase_text(generated)))

    nav = "".join(f"<a href='#score-{html.escape(key)}'>{html.escape(key)}</a>" for key in sorted(pieces))
    sections = []
    for score_key in sorted(pieces):
        phrases = sorted(pieces[score_key], key=lambda row: phrase_number(row[0]))
        original_full = "\n\n".join(f"【{phrase_id}】\n{original}" for phrase_id, original, _ in phrases)
        generated_full = "\n\n".join(f"【{phrase_id}】\n{generated}" for phrase_id, _, generated in phrases)
        sections.append(f"""<section id='score-{html.escape(score_key)}'><h2>{html.escape(score_key)} · {len(phrases)} 句</h2>
<p>每行格式为“音序: 减字”；“（空）”代表该音序没有减字。两栏按相同乐句、相同音序排列，可直接逐项对照。</p>
<div class='compare'><article><h3>原版标注</h3><pre>{html.escape(original_full)}</pre></article><article><h3>N55 生成标注</h3><pre>{html.escape(generated_full)}</pre></article></div></section>""")
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>N55 与原版标注：整首曲谱对照</title><style>
body{{font:15px/1.65 system-ui,'Microsoft YaHei';max-width:1600px;margin:28px auto;padding:0 18px;background:#f4f6fa;color:#263146}}section{{background:#fff;border:1px solid #dce3ed;border-radius:12px;padding:18px;margin:16px 0}}h1,h2,h3{{margin-top:0}}nav{{display:flex;gap:8px;flex-wrap:wrap}}nav a{{padding:4px 8px;background:#eaf2ff;border-radius:6px;color:#174a8b;text-decoration:none}}.compare{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}article{{border:1px solid #dce3ed;border-radius:8px;overflow:hidden}}article h3{{padding:10px 12px;margin:0;background:#f5f8fc}}pre{{margin:0;padding:12px;white-space:pre-wrap;word-break:break-word;font:13px/1.6 ui-monospace,'Cascadia Mono','Microsoft YaHei',monospace;max-height:680px;overflow:auto}}@media(max-width:900px){{.compare{{grid-template-columns:1fr}}}}</style>
<h1>N55 生成标注与原版标注：按整首曲谱对照</h1><section><p>共 {len(pieces)} 首曲谱、{sum(len(value) for value in pieces.values())} 句。此文件是独立的稳定静态对照文档；不替代原有的评分报告和逐句音高校对报告。</p><nav>{nav}</nav></section>{''.join(sections)}</html>"""
    target = OUT / "N55_test50_ORIGINAL_VS_GENERATED_BY_FULL_SCORE.html"
    target.write_text(page, encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(target), "scores": len(pieces),
                      "phrases": sum(len(value) for value in pieces.values())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
