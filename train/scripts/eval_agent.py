#!/usr/bin/env python3
"""Generate Base/Fingering predictions for public evaluation inputs.

The input intentionally contains no reference annotation.  This script only
loads a finished LoRA adapter and writes raw model continuations plus a
recoverable ``jianzi_rows`` parse.  Scoring against the sealed references is
performed locally by ``score_agent_predictions.py``.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EDIT_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_plan",
        "description": (
            "批量预览并设置减字文本。唯一输入为 "
            "jianzi_rows=[[序号,减字文字],...]；可以只提交本次决定修改的行。"
        ),
        "parameters": {
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
    },
}


BASE_SYSTEM = (
    "你是减字谱初稿 Agent。阅读简谱、ABC、时值和前一段减字，"
    "为当前段每个演奏事件填写基础减字；也可以填写\"再作标记\"/\"再作\"/\"从ㄱ再作\"等表示省略的减字。"
    "只使用 edit_plan.jianzi_rows；工具的音高结果是机会式警告：仅在双方可可靠解析时参考，复杂动作无法解析不算错误。"
    "基础初稿只使用泛音、按音、散音三种取音方式，右手只使用抹、挑、勾、剔、擘、托、打、摘、撮；"
    "填写或编辑曲谱只能使用 edit_plan.jianzi_rows；尽量为当前段每个发音事件填写减字，"
    "暂不主动加入绰、注、吟、猱、走手或其他复杂技法。"
)


def render_public_input(row: dict[str, Any]) -> str:
    if row.get("public_prompt"):
        return str(row["public_prompt"])
    payload = row.get("input") or ((row.get("runtime_item") or {}).get("input")) or {}
    metadata = payload.get("metadata") or {}
    # Support both the legacy v1 envelope and the current text-protocol v2
    # runtime payload when a pre-rendered public_prompt is unavailable.
    tuning = payload.get("normalized_tuning") or payload.get("tuning") or {}
    notes = payload.get("notes") or payload.get("notes_without_jianzi") or []
    title = metadata.get("score_title") or row.get("score_key") or ""
    tuning_name = tuning.get("name") or ""
    open_midi = tuning.get("open_midi") or []
    event_range = payload.get("event_range") or {}
    lines = [f"谱名｜{title}", f"调弦｜{tuning_name}｜{json.dumps(open_midi, ensure_ascii=False)}"]
    if event_range:
        lines.append(
            f"当前段｜{event_range.get('start', '')}-{event_range.get('end_exclusive', '')}"
        )
    lines.append("序号｜简谱｜ABC｜时值")
    for note in notes:
        lines.append("｜".join(
            str(note.get(key) or "")
            for key in ("index", "jianpu", "abc", "duration")
        ))
    lines.append("要求｜通过 edit_plan.jianzi_rows 提交减字文字。")
    return "\n".join(lines)


def _first_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON value found")


def _normalize_rows(value: Any) -> list[list[Any]]:
    if isinstance(value, dict):
        if "jianzi_rows" in value:
            value = value["jianzi_rows"]
        elif "arguments" in value:
            return _normalize_rows(value["arguments"])
    if not isinstance(value, list):
        raise ValueError("jianzi_rows is not an array")
    rows: list[list[Any]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], str):
            raise ValueError("row is not [source_index, jianzi_text]")
        index = row[0]
        if isinstance(index, str) and re.fullmatch(r"[0-9]+", index):
            index = int(index)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("source_index is not an integer")
        rows.append([index, row[1]])
    return rows


def parse_jianzi_rows(text: str) -> tuple[list[list[Any]], str | None]:
    blocks = re.findall(r"<function\s*=\s*edit_plan>(.*?)</function>", text, flags=re.S)
    candidates = list(blocks)
    if not candidates:
        candidates = [text]
    last_error = "no edit_plan function call"
    for block in reversed(candidates):
        parameter = re.search(
            r"<parameter\s*=\s*jianzi_rows>(.*?)</parameter>", block, flags=re.S
        )
        candidate = parameter.group(1) if parameter else block
        candidate = candidate.replace("```json", "").replace("```", "").strip()
        try:
            value = _first_json_value(candidate)
            if isinstance(value, dict) and "tool_calls" in value:
                for call in reversed(value["tool_calls"]):
                    rows, _ = parse_jianzi_rows(json.dumps(call, ensure_ascii=False))
                    return rows, None
            return _normalize_rows(value), None
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return [], last_error


def generate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        from scripts.adapter_loading import load_adapter_checked
        from scripts.qwen35_generation import qwen35_eos_token_ids, trim_qwen35_assistant_turn
    except ModuleNotFoundError:
        from adapter_loading import load_adapter_checked
        from qwen35_generation import qwen35_eos_token_ids, trim_qwen35_assistant_turn

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.adapter:
        model = load_adapter_checked(model, args.adapter)
    model.eval()
    generation_eos_ids = qwen35_eos_token_ids(tokenizer)
    completed_ids: set[tuple[str, str | None]] = set()
    if args.resume and args.output.exists():
        for line in args.output.open(encoding="utf-8"):
            if line.strip():
                prior = json.loads(line)
                if prior.get("protocol_valid") and prior.get("jianzi_rows"):
                    completed_ids.add((prior["sample_id"], prior.get("input_sha256")))
    output = args.output.open("a" if args.resume else "w", encoding="utf-8", newline="\n")
    count = 0

    def infer_batch(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
        conversations = [
            [
                {"role": "system", "content": BASE_SYSTEM},
                {"role": "user", "content": render_public_input(row)},
            ]
            for row in rows
        ]
        try:
            encoded = tokenizer.apply_chat_template(
                conversations,
                tools=[EDIT_PLAN_TOOL],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )
            encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
            padded_input_length = encoded["input_ids"].shape[-1]
            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=generation_eos_ids,
                )
            return [
                (
                    trim_qwen35_assistant_turn(tokenizer.decode(
                        generated[index, padded_input_length:], skip_special_tokens=False
                    )),
                    int(prompt_lengths[index]),
                )
                for index in range(len(rows))
            ]
        except torch.OutOfMemoryError:
            if len(rows) == 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(rows) // 2
            print(
                f"batch OOM at {len(rows)}; retrying as {midpoint}+{len(rows) - midpoint}",
                flush=True,
            )
            return infer_batch(rows[:midpoint]) + infer_batch(rows[midpoint:])

    try:
        pending = [
            json.loads(line)
            for line in args.input.open(encoding="utf-8")
            if line.strip() and (
                json.loads(line)["sample_id"], json.loads(line).get("input_sha256")
            ) not in completed_ids
        ]
        for start in range(0, len(pending), args.batch_size):
            rows = pending[start:start + args.batch_size]
            results = infer_batch(rows)
            for row, (continuation, prompt_tokens) in zip(rows, results):
                parsed_rows, parse_error = parse_jianzi_rows(continuation)
                output.write(json.dumps({
                    "schema_version": "agent-eval-prediction-1.0",
                    "sample_id": row["sample_id"],
                    "split": row.get("split"),
                    "stage": args.stage,
                    "input_sha256": row.get("input_sha256"),
                    "prompt_tokens": prompt_tokens,
                    "raw_output": continuation,
                    "jianzi_rows": parsed_rows,
                    "protocol_valid": parse_error is None,
                    "parse_error": parse_error,
                }, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
            if count % 20 == 0 or count == len(pending):
                print(f"generated {count}; resumed {len(completed_ids)}", flush=True)
    finally:
        output.close()
    print(json.dumps({"rows": count, "output": str(args.output)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", default="fingering_agent")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
