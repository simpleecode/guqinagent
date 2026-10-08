#!/usr/bin/env python3
"""Export structured evaluation results as a compact Markdown report.

The evaluator remains the single source of metric definitions.  This script
only groups its event-level output by score and formats the resulting metrics.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.aggregate import metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return str(value)


def metric_rows(m: dict[str, Any]) -> list[tuple[str, Any, bool]]:
    tone = m.get("tone_type") or {}
    ornament = m.get("ornament") or {}
    technique = m.get("performance_technique_usage") or {}
    left = m.get("left_hand_fingering") or {}
    right = m.get("right_hand_fingering") or {}
    string = m.get("string") or {}
    return [
        ("Pitch Accuracy @ 50 cents", m.get("pitch_accuracy_at_50_cents"), True),
        ("Pitch MAE (cents)", m.get("pitch_mae_cents"), False),
        ("Pitch evaluable events", m.get("pitch_evaluable_events"), False),
        ("String Accuracy", string.get("accuracy"), True),
        ("Left-hand Fingering Accuracy", left.get("accuracy"), True),
        ("Left-hand Fingering F1", left.get("f1"), True),
        ("Right-hand Fingering Accuracy", right.get("accuracy"), True),
        ("Right-hand Fingering F1", right.get("f1"), True),
        ("Tone-type Accuracy", tone.get("accuracy"), True),
        ("Tone-type F1", tone.get("f1"), True),
        ("Tone-type Distribution Similarity", tone.get("distribution_similarity"), True),
        ("Ornament Precision", ornament.get("precision"), True),
        ("Ornament Recall", ornament.get("recall"), True),
        ("Ornament F1", ornament.get("f1"), True),
        ("Technique Distribution Similarity", technique.get("distribution_similarity"), True),
        ("Technique Unique Vocabulary (prediction)", (technique.get("technique_diversity") or {}).get("prediction", {}).get("unique_techniques"), False),
        ("Technique Effective Vocabulary (prediction)", (technique.get("technique_diversity") or {}).get("prediction", {}).get("effective_techniques"), False),
        ("Technique Unique Vocabulary (reference)", (technique.get("technique_diversity") or {}).get("reference", {}).get("unique_techniques"), False),
        ("Technique Effective Vocabulary (reference)", (technique.get("technique_diversity") or {}).get("reference", {}).get("effective_techniques"), False),
        ("Rule Violation Rate", m.get("rule_violation_rate"), True),
        ("Rule Violations", m.get("rule_violations"), False),
    ]


def report(summary: dict[str, Any], events: list[dict[str, Any]], violations: list[dict[str, Any]],
           experiment: str, source: str) -> str:
    tolerance = float((summary.get("metrics") or {}).get("pitch_tolerance_cents", 50.0))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event.get("piece_id") or "")] .append(event)
    violations_by_piece: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for violation in violations:
        violations_by_piece[str(violation.get("piece_id") or "")].append(violation)

    lines = [
        f"# Adapter 评估分数｜{experiment}", "",
        f"- 评估结果：`{source}`",
        f"- 预测样本：{summary.get('prediction_samples')}；匹配样本：{summary.get('matched_samples')}",
        f"- 结构化事件：{summary.get('events')}；音高容差：{fmt(tolerance)} cents",
        "",
        "## 总体指标",
        "",
        "| 指标 | 分数 |",
        "|---|---:|",
    ]
    for name, value, percent in metric_rows(summary.get("metrics") or {}):
        lines.append(f"| {name} | {fmt(value, percent)} |")
    warning = summary.get("final_warning") or {}
    lines.append(f"| Final warning rate | {fmt(warning.get('warning_rate'), True)} |")
    lines.append(f"| Final warning phrases | {warning.get('warning_phrases', '—')} / {warning.get('phrases', '—')} |")

    lines += ["", "## 按曲谱指标", "", "以下按 `piece_id` 汇总其全部 phrase；空值表示该曲谱没有足够可评估事件。", ""]
    headers = [
        "曲谱", "事件数", "Pitch@50c", "MAE(cents)", "String Acc.",
        "左手 F1", "右手 F1", "Tone Acc.", "Tone Dist.", "Orn. F1",
        "Technique Dist.", "Tech Eff. Vocab", "Rule Violation",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---:|" * len(headers))
    for piece_id in sorted(grouped):
        m = metrics(grouped[piece_id], violations_by_piece.get(piece_id, []), tolerance)
        tone = m.get("tone_type") or {}
        ornament = m.get("ornament") or {}
        technique = m.get("performance_technique_usage") or {}
        left = m.get("left_hand_fingering") or {}
        right = m.get("right_hand_fingering") or {}
        row = [
            piece_id,
            str(len(grouped[piece_id])),
            fmt(m.get("pitch_accuracy_at_50_cents"), True),
            fmt(m.get("pitch_mae_cents")),
            fmt((m.get("string") or {}).get("accuracy"), True),
            fmt(left.get("f1"), True),
            fmt(right.get("f1"), True),
            fmt(tone.get("accuracy"), True),
            fmt(tone.get("distribution_similarity"), True),
            fmt(ornament.get("f1"), True),
            fmt(technique.get("distribution_similarity"), True),
            fmt(((technique.get("technique_diversity") or {}).get("prediction") or {}).get("effective_techniques")),
            fmt(m.get("rule_violation_rate"), True),
        ]
        lines.append("| " + " | ".join(row) + " |")

    incomplete = summary.get("tone_type_replay_incomplete_scores") or []
    lines += ["", "## 评估备注", "", f"- Tone replay 不完整曲谱：{len(incomplete)} 条。" ]
    if incomplete:
        lines.append("- ID：" + ", ".join(f"`{x}`" for x in incomplete))
    unmatched = summary.get("unmatched_prediction_ids") or []
    lines.append(f"- 未匹配 prediction：{len(unmatched)} 条。")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("train/eval_score"))
    parser.add_argument("--name", default="adapter_metrics.md")
    args = parser.parse_args()
    evaluation_dir = args.evaluation_dir
    summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    events = load_jsonl(evaluation_dir / "per_event.jsonl")
    violations = load_jsonl(evaluation_dir / "violations.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.name
    output.write_text(report(summary, events, violations, args.name.removesuffix(".md"), str(evaluation_dir)), encoding="utf-8")
    print(json.dumps({"output": str(output), "scores": len({e.get('piece_id') for e in events}), "events": len(events)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
