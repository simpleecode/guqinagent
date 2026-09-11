#!/usr/bin/env python3
import argparse
import html
import json
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = {row["sample_id"]: row for row in load_jsonl(args.inputs)}
    predictions = {row["sample_id"]: row for row in load_jsonl(args.predictions)}
    references = {row["sample_id"]: row for row in load_jsonl(args.references)}
    sample_ids = sorted(
        sample_id for sample_id in predictions
        if sample_id.startswith(args.score + "-") and sample_id in inputs
    )
    title = ""
    sections = []
    for position, sample_id in enumerate(sample_ids):
        source = inputs[sample_id]["input"]
        title = title or source.get("metadata", {}).get("score_title", "")
        predicted = dict(predictions[sample_id].get("jianzi_rows") or [])
        actions = references.get(sample_id, {}).get("reference", {}).get("actions") or []
        annotated = {item["source_index"]: item.get("text") or "" for item in actions}
        rows = []
        for note in source.get("notes") or []:
            index = note.get("index")
            pred = predicted.get(index, "")
            ref = annotated.get(index, "")
            same = pred == ref
            rows.append(
                "<tr>"
                f"<td class='text-nowrap'>{html.escape(str(index))}</td>"
                f"<td>{html.escape(str(note.get('jianpu') or ''))}</td>"
                f"<td>{html.escape(str(note.get('abc') or ''))}</td>"
                f"<td>{html.escape(str(note.get('duration') or ''))}</td>"
                f"<td class='{'same' if same else ''}'>{html.escape(pred or '∅')}</td>"
                f"<td class='{'same' if same else ''}'>{html.escape(ref or '∅')}</td>"
                "</tr>"
            )
        raw = html.escape(predictions[sample_id].get("raw_output") or "")
        sections.append(
            f"<details {'open' if position == 0 else ''}>"
            f"<summary><code>{html.escape(sample_id)}</code> · {len(rows)} 行</summary>"
            "<div class='table-responsive'><table class='table table-sm'>"
            "<thead><tr><th>序号</th><th>简谱</th><th>ABC</th><th>时值</th><th>模型</th><th>标注</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<details><summary>模型完整输出</summary>"
            f"<pre><code>{raw}</code></pre></details></details>"
        )
    fragment = f"""<div id="eval-score-complete">
<style>
#eval-score-complete {{ color: var(--foreground); }}
#eval-score-complete .same {{ background: color-mix(in srgb, var(--green) 16%, transparent); }}
#eval-score-complete details {{ margin-block: .45rem; }}
#eval-score-complete summary {{ cursor: pointer; }}
#eval-score-complete pre {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--foreground); }}
</style>
<h2>{html.escape(args.score)}｜{html.escape(title)}｜已完成 {len(sample_ids)} 个片段</h2>
{''.join(sections)}
</div>
"""
    args.output.write_text(fragment, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
