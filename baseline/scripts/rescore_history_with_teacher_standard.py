#!/usr/bin/env python3
"""Re-score saved prompt experiments with the repository's sealed-score rules.

This intentionally mirrors ``guqinagent/train/scripts/score_agent_predictions.py``:
NFKC text normalisation, bracket removal, Chinese string-number normalisation,
exact-cell accuracy over the union of reference/prediction indexes, non-empty
reference recall, and protocol-valid rate.  It does not call a model.

Historical experiments used different fixed samples.  Results are therefore
reported per experiment and arm; a cross-experiment total ranking is omitted.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
DATASETS = (
    ROOT / "_tmp_zip_inspect" / "pitch_eligible_two_or_half" / "inferred_trajectories_train.jsonl",
    ROOT / "_tmp_zip_inspect" / "pitch_eligible_two_or_half" / "inferred_trajectories_test.jsonl",
)


def canonical(text: Any) -> str:
    """Exact copy of score_agent_predictions.canonical()."""
    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    value = re.sub(r"[\[\]（）()]", "", value)
    for source, target in {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7"}.items():
        value = value.replace(source + "弦", target + "弦")
    return re.sub(r"\s+", "", value)


def load_result_rows(paths: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], set[str], list[dict[str, str]]]:
    experiments: dict[str, list[dict[str, Any]]] = {}
    sample_ids: set[str] = set()
    skipped: list[dict[str, str]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows")
        if not isinstance(rows, list):
            skipped.append({"file": path.name, "reason": "no row-level saved prediction output to re-score"})
            continue
        valid = [row for row in rows if isinstance(row, dict) and row.get("sample_id") and isinstance(row.get("prediction"), dict)]
        if valid:
            experiments[path.stem] = valid
            sample_ids.update(str(row["sample_id"]) for row in valid)
        else:
            skipped.append({"file": path.name, "reason": "no saved valid prediction.jianzi_rows to re-score"})
    return experiments, sample_ids, skipped


def load_references(target_ids: set[str]) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for path in DATASETS:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("trajectory_id") or row.get("sample_id") or "")
                if sample_id in target_ids:
                    references[sample_id] = row
    return references


def reference_map(row: dict[str, Any]) -> dict[int, str]:
    actions = (row.get("reference_plan") or {}).get("actions") or []
    return {
        int(item["source_index"]): str(item.get("text") or item.get("jianzi_text") or "")
        for item in actions if isinstance(item, dict) and item.get("source_index") is not None
    }


def prediction_map(row: dict[str, Any]) -> dict[int, str]:
    cells = (row.get("prediction") or {}).get("jianzi_rows") or []
    return {
        int(cell[0]): str(cell[1])
        for cell in cells if isinstance(cell, (list, tuple)) and len(cell) >= 2
    }


def score_arm(rows: list[dict[str, Any]], references: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    missing: list[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        reference = references.get(sample_id)
        if reference is None:
            missing.append(sample_id)
            continue
        refs, generated = reference_map(reference), prediction_map(row)
        indices = set(refs) | set(generated)
        exact = sum(canonical(refs.get(index)) == canonical(generated.get(index)) for index in indices)
        nonempty_ref = {index for index, text in refs.items() if canonical(text)}
        nonempty_pred = {index for index, text in generated.items() if canonical(text)}
        totals["samples"] += 1
        totals["protocol_valid"] += int(bool(row.get("protocol_ok")))
        totals["rows_compared"] += len(indices)
        totals["exact_cells"] += exact
        totals["reference_nonempty"] += len(nonempty_ref)
        totals["predicted_nonempty"] += len(nonempty_pred)
        totals["nonempty_hits"] += len(nonempty_ref & nonempty_pred)
    samples = totals["samples"]
    return {
        **totals,
        "missing_reference_samples": sorted(missing),
        "protocol_valid_rate": totals["protocol_valid"] / samples if samples else None,
        "exact_cell_rate": totals["exact_cells"] / totals["rows_compared"] if totals["rows_compared"] else None,
        "nonempty_recall": totals["nonempty_hits"] / totals["reference_nonempty"] if totals["reference_nonempty"] else None,
    }


def render_html(report: dict[str, Any]) -> str:
    def pct(value: Any) -> str:
        return "—" if value is None else f"{float(value):.2%}"
    rows = []
    for experiment in report["experiments"]:
        for arm in experiment["arms"]:
            score = arm["score"]
            rows.append(
                "<tr>"
                f"<td>{html.escape(experiment['name'])}</td><td>{html.escape(arm['name'])}</td>"
                f"<td>{score['samples']}/{experiment['saved_rows']}</td>"
                f"<td>{pct(score['exact_cell_rate'])}</td><td>{pct(score['nonempty_recall'])}</td>"
                f"<td>{pct(score['protocol_valid_rate'])}</td><td>{score['rows_compared']}</td>"
                "</tr>"
            )
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>历史提示词｜师兄标准重评分</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 18px;font:15px/1.6 system-ui,'Microsoft YaHei',sans-serif;color:#17212b;background:#f6f8fb}}h1{{margin-bottom:4px}}.card{{background:#fff;border:1px solid #dce3eb;border-radius:10px;padding:16px;margin:16px 0}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #dce3eb;padding:8px;text-align:left}}th{{background:#eef3f8}}code{{background:#eef3f8;padding:2px 4px;border-radius:4px}}.note{{color:#536271}}</style>
<h1>历史提示词重评分</h1><p class='note'>生成于 {html.escape(report['generated_at'])}。本报告没有重新调用模型，只重算已保存预测。</p>
<section class='card'><h2>严格采用的规则</h2><p>复刻 <code>guqinagent/train/scripts/score_agent_predictions.py</code>：NFKC、去括号、中文弦号转阿拉伯数字、去空白后，按“参考与预测索引并集”逐格精确匹配；同时报告非空标注召回和协议合规率。</p><p><b>重要：</b>不同历史实验的样本集、随机种子不相同，不能跨实验把数字拼成总排行榜；只有同一实验内的多个 arm 可以直接比较。</p></section>
<section class='card'><h2>结果</h2><table><thead><tr><th>历史实验</th><th>提示词</th><th>已评分/保存行</th><th>逐格精确率</th><th>非空召回</th><th>协议合规</th><th>比较格数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=RESULTS / "teacher_standard_history_rescore.json")
    parser.add_argument("--html", type=Path, default=REPORTS / "teacher_standard_history_rescore.html")
    args = parser.parse_args()
    files = sorted(args.results_dir.glob("prompt_*result.json"))
    experiments, sample_ids, skipped = load_result_rows(files)
    references = load_references(sample_ids)
    output_experiments = []
    for name, rows in experiments.items():
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_arm.setdefault(str(row.get("arm") or "unknown"), []).append(row)
        output_experiments.append({
            "name": name,
            "saved_rows": len(rows),
            "arms": [{"name": arm, "score": score_arm(arm_rows, references)} for arm, arm_rows in sorted(by_arm.items())],
        })
    report = {
        "schema_version": "teacher-standard-history-rescore-1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_scorer": "guqinagent/train/scripts/score_agent_predictions.py",
        "result_files_seen": len(files),
        "referenced_samples_found": len(references),
        "skipped_files": skipped,
        "experiments": output_experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "html": str(args.html), "experiments": len(output_experiments), "references": len(references)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
