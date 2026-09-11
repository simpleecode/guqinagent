#!/usr/bin/env python3
"""Use a teacher model to rewrite observable decision summaries, then revalidate."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def clean_summary(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value[:180]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "ABC_J/agent_training/messages")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ABC_J/agent_training/messages_model")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                       base_url=os.environ["ANTHROPIC_BASE_URL"].rstrip("/"),
                       timeout=90.0, max_retries=1)
    model = args.model or os.getenv("MINIMAX_MODEL", "MiniMax-M3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_rows = [json.loads(line) for line in
                   (args.input_dir / "messages_train.jsonl").open(encoding="utf-8")]
    private_by_id = {row["sample_id"]: row for row in (
        json.loads(line) for line in
        (args.input_dir / "teacher_trajectory_audit.jsonl").open(encoding="utf-8"))}
    selected = public_rows[:args.limit]
    failures, refined = [], []
    with (args.output_dir / "messages_train.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for item in selected:
            private = private_by_id[item["sample_id"]]
            assistant_indices = [index for index, message in enumerate(item["messages"])
                                 if message.get("role") == "assistant" and message.get("tool_calls")]
            prompt = {
                "instruction": (
                    "你是教师模型。根据私有目标和真实工具结果，为每个工具调用改写一句简短、可核验的"
                    "外显决策摘要。不得提到标注、答案、teacher、reference、target patch，不能添加工具结果中"
                    "不存在的事实。仅返回 JSON：{\"summaries\":[字符串...]}，数量必须准确。"
                ),
                "agent_stage": item["agent_stage"],
                "messages": [item["messages"][index:index + 2] for index in assistant_indices],
                "teacher_private": private["teacher_private"],
            }
            try:
                options = {"model": model, "max_tokens": 1200, "temperature": 0.8,
                           "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]}
                if model == "MiniMax-M3":
                    options["thinking"] = {"type": "disabled"}
                response = client.messages.create(**options)
                text = "".join(block.text for block in response.content
                               if getattr(block, "type", None) == "text")
                start, end = text.find("{"), text.rfind("}")
                payload = json.loads(text[start:end + 1])
                summaries = payload["summaries"]
                if len(summaries) != len(assistant_indices):
                    raise ValueError("summary count mismatch")
                banned = ("标注", "答案", "teacher", "reference", "target patch", "目标patch")
                for index, summary in zip(assistant_indices, summaries):
                    summary = clean_summary(str(summary))
                    if not summary or any(token.lower() in summary.lower() for token in banned):
                        raise ValueError("summary contains private-answer language")
                    item["messages"][index]["content"] = summary
                item["generation_mode"] = "teacher_forced_model_refined"
                item["teacher_model"] = model
                refined.append(item["sample_id"])
            except Exception as exc:
                failures.append({"sample_id": item["sample_id"],
                                 "error": f"{type(exc).__name__}: {exc}"})
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_dir / "teacher_trajectory_audit.jsonl").open(
        "w", encoding="utf-8", newline="\n") as out:
        for item in selected:
            out.write(json.dumps(private_by_id[item["sample_id"]], ensure_ascii=False) + "\n")
    report = {"schema_version": "model-refinement-1.0", "model": model,
              "requested": len(selected), "refined": len(refined),
              "fallback_template": len(failures), "failures": failures}
    (args.output_dir / "model_refinement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
