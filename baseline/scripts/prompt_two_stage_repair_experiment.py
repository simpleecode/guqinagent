"""Evaluate a frozen musical draft followed by a format-only GLM repair pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import prompt_candidate_paired_experiment as experiment
from baseline.gepa_guqin.glm_client import GlmAnthropicClient, GlmBackend, tool_calls
from baseline.gepa_guqin.metric import DEFAULT_WEIGHTS, phrase_report


REPAIR_SYSTEM = """你是减字谱字符串的最小校对器，不是重新编配的演奏 Agent。
输入已经包含第一阶段确定的 event_index 与减字字符串。你唯一允许的修改是：当一个明确的弦序数字后面缺少“弦”字时，只插入一个“弦”字。

除此之外所有字符必须逐字保持：不得改中文/阿拉伯数字，不得改徽位、弦号、字段顺序、空格、按/散/泛、左右手指法、技法、事件位置或空行。特别禁止把`九徽`改成`9徽`。已经含“弦”字的行原样返回；无法唯一判断时原样返回。所有输入行必须按原索引、原顺序返回。只调用 submit_repair。"""


REPAIR_TOOL = {
    "name": "submit_repair",
    "description": "提交仅做机械格式修复后的完整减字行",
    "input_schema": {
        "type": "object",
        "properties": {
            "jianzi_rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [{"type": "integer"}, {"type": "string"}],
                    "minItems": 2,
                    "maxItems": 2,
                },
            }
        },
        "required": ["jianzi_rows"],
        "additionalProperties": False,
    },
}


def row_lookup(result: dict, arm: str) -> dict[str, dict]:
    return {row["sample_id"]: row for row in result["rows"] if row["arm"] == arm}


def insertion_only(original: str, repaired: str) -> bool:
    """Only additional 弦 characters are permitted; all other text is immutable."""
    return (repaired.replace("弦", "") == original.replace("弦", "")
            and repaired.count("弦") >= original.count("弦"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("train", "test"), required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--source-arm", default="production")
    parser.add_argument("--comparison-result", type=Path, required=True)
    parser.add_argument("--comparison-arm", default="v9")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_result = json.loads(args.source_result.read_text(encoding="utf-8"))
    comparison_result = json.loads(args.comparison_result.read_text(encoding="utf-8"))
    sources = row_lookup(source_result, args.source_arm)
    comparisons = row_lookup(comparison_result, args.comparison_arm)
    sample_ids = [sample_id for sample_id in source_result["sample_ids"]
                  if sample_id in sources and sample_id in comparisons]
    raw_rows = experiment.select_explicit_rows(args.dataset, sample_ids)
    examples = {item["sample_id"]: item for item in map(experiment.prepare, raw_rows)}

    client = GlmAnthropicClient(min_interval=0.5, max_retries=1)
    backend = GlmBackend(client)
    result = {
        "kind": "two_stage_format_repair_pilot",
        "model": client.model,
        "dataset": args.dataset,
        "source_arm": args.source_arm,
        "comparison_arm": args.comparison_arm,
        "sample_ids": sample_ids,
        "weights": DEFAULT_WEIGHTS,
        "rows": [],
    }

    for sample_id in sample_ids:
        source = sources[sample_id]
        comparison = comparisons[sample_id]
        original_rows = source["prediction"]["jianzi_rows"]
        expected_indices = [int(row[0]) for row in original_rows]
        user_payload = {
            "instruction": "只做允许的机械格式修复，保持全部音乐决策不变。",
            "jianzi_rows": original_rows,
        }
        error = None
        repaired_prediction = None
        try:
            response = backend.generate(
                system=REPAIR_SYSTEM,
                messages=[{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
                tools=[REPAIR_TOOL],
                temperature=0.0,
            )
            calls = [call for call in tool_calls(response) if call["name"] == "submit_repair"]
            if len(calls) != 1:
                raise ValueError(f"expected one submit_repair call, got {len(calls)}")
            repaired_rows = calls[0]["arguments"].get("jianzi_rows") or []
            normalized = [[int(row[0]), str(row[1])] for row in repaired_rows
                          if isinstance(row, list) and len(row) == 2]
            if [row[0] for row in normalized] != expected_indices:
                raise ValueError("repair changed, dropped, or reordered source indices")
            safe_rows = []
            reverted = 0
            for original, repaired in zip(original_rows, normalized):
                text = repaired[1] if insertion_only(str(original[1]), repaired[1]) else str(original[1])
                reverted += int(text != repaired[1])
                safe_rows.append([repaired[0], text])
            repaired_prediction = {
                "score_key": source["prediction"]["score_key"],
                "phrase_id": source["prediction"]["phrase_id"],
                "jianzi_rows": safe_rows,
            }
            report = phrase_report(examples[sample_id], repaired_prediction, {})
            repaired_score = float(report["score"]) if report["score"] is not None else 0.5
            components = report["metrics"]["components"]
            violations = report["metrics"]["rule_violations"]
            feedback = report["feedback"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            repaired_score = 0.0
            components = {}
            violations = None
            feedback = ""
        row = {
            "sample_id": sample_id,
            "source_score": source["score"],
            "comparison_score": comparison["score"],
            "repaired_score": repaired_score,
            "delta_vs_source": repaired_score - source["score"],
            "delta_vs_comparison": repaired_score - comparison["score"],
            "components": components,
            "rule_violations": violations,
            "feedback": feedback,
            "source_prediction": source["prediction"],
            "repaired_prediction": repaired_prediction,
            "error": error,
            "unsafe_rows_reverted": reverted if error is None else None,
        }
        result["rows"].append(row)
        deltas = [item["delta_vs_comparison"] for item in result["rows"]]
        result["summary"] = {
            "n": len(result["rows"]),
            "source_mean": sum(item["source_score"] for item in result["rows"]) / len(result["rows"]),
            "comparison_mean": sum(item["comparison_score"] for item in result["rows"]) / len(result["rows"]),
            "repaired_mean": sum(item["repaired_score"] for item in result["rows"]) / len(result["rows"]),
            "mean_delta_vs_comparison": sum(deltas) / len(deltas),
            "wins_ties_losses_vs_comparison": [sum(d > 1e-9 for d in deltas),
                                                 sum(abs(d) <= 1e-9 for d in deltas),
                                                 sum(d < -1e-9 for d in deltas)],
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"sample_id": sample_id, "source": source["score"],
                          "v9": comparison["score"], "repaired": repaired_score,
                          "error": error}, ensure_ascii=False), flush=True)

    result["usage"] = client.snapshot()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
