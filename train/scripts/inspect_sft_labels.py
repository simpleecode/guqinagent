#!/usr/bin/env python3
"""Randomly audit full-trajectory labels with the installed LLaMA-Factory processor."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def sample_jsonl(path: Path, count: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    rng = random.Random(seed)
    selected: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = (line_number, json.loads(line))
            if len(selected) < count:
                selected.append(row)
            else:
                position = rng.randrange(line_number)
                if position < count:
                    selected[position] = row
    return sorted(selected)


def align_openai_row(row: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], str, str]:
    messages = list(row["messages"])
    if not messages or messages[0].get("role") != "system":
        raise ValueError("expected a leading system message")
    system = str(messages.pop(0).get("content") or "")
    tools = row.get("tools") or ""
    aligned: list[dict[str, str]] = []
    tool_responses: list[str] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "tool":
            tool_responses.append(content)
            continue
        if tool_responses:
            aligned.append({
                "role": "observation",
                "content": "\n</tool_response>\n<tool_response>\n".join(tool_responses),
            })
            tool_responses = []
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported role after export: {role}")
        aligned.append({"role": role, "content": content})
    if tool_responses:
        aligned.append({
            "role": "observation",
            "content": "\n</tool_response>\n<tool_response>\n".join(tool_responses),
        })
    if not aligned or aligned[-1]["role"] != "assistant":
        raise ValueError("aligned trajectory does not end in assistant")
    # Mirrors the local 0.9.6.dev0 OpenAIDatasetConverter behavior when tools exist.
    if tools:
        system += "\n"
    return aligned[:-1], aligned[-1:], system, tools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llama-factory-home", type=Path, required=True)
    parser.add_argument("--template", default="qwen3_5")
    parser.add_argument("--cutoff-len", type=int, default=18432)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--dump-decoded", action="store_true",
        help="store the full decoded input_ids and supervised assistant targets",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.llama_factory_home / "src"))
    from transformers import AutoProcessor
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.processor_utils import infer_seqlen
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.hparams import DataArguments

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    data_args = DataArguments(
        template=args.template,
        cutoff_len=args.cutoff_len,
        mask_history=False,
        train_on_prompt=False,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_processor = SupervisedDatasetProcessor(
        template=template,
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )

    errors: list[str] = []
    audits: list[dict[str, Any]] = []
    for line_number, row in sample_jsonl(args.input, args.samples, args.seed):
        try:
            prompt, response, system, tools = align_openai_row(row)
            all_messages = template.mm_plugin.process_messages(
                prompt + response, [], [], [], processor
            )
            pairs = template.encode_multiturn(tokenizer, all_messages, system, tools, False)
            input_ids, labels = dataset_processor._encode_data_example(
                prompt=prompt,
                response=response,
                system=system,
                tools=tools,
                images=[],
                videos=[],
                audios=[],
            )
            assistant_messages = [message for message in row["messages"] if message["role"] == "assistant"]
            cursor = 0
            turn_reports = []
            decoded_targets: list[str] = []
            expected_labels: list[int] = []
            total_length = 1 if template.efficient_eos else 0
            for turn_index, (source_ids, target_ids) in enumerate(pairs):
                if total_length >= args.cutoff_len:
                    break
                source_len, target_len = infer_seqlen(
                    len(source_ids), len(target_ids), args.cutoff_len - total_length
                )
                source_ids = source_ids[:source_len]
                target_ids = target_ids[:target_len]
                total_length += source_len + target_len
                source_labels = [IGNORE_INDEX] * source_len
                if template.efficient_eos and turn_index != 0 and source_len:
                    source_labels[0] = tokenizer.eos_token_id
                target_labels = target_ids
                expected_labels.extend(source_labels + target_labels)
                actual_source = labels[cursor:cursor + source_len]
                cursor += source_len
                actual_target = labels[cursor:cursor + target_len]
                cursor += target_len
                source_content_masked = all(
                    label == IGNORE_INDEX or (
                        template.efficient_eos and position == 0 and turn_index != 0
                        and label == tokenizer.eos_token_id
                    )
                    for position, label in enumerate(actual_source)
                )
                target_fully_supervised = actual_target == target_labels
                decoded_target = tokenizer.decode(target_ids, skip_special_tokens=False)
                decoded_targets.append(decoded_target)
                original_content = assistant_messages[turn_index]["content"]
                content_preserved = original_content.strip() in decoded_target
                turn_report = {
                    "assistant_turn": turn_index + 1,
                    "source_tokens": source_len,
                    "source_content_masked": source_content_masked,
                    "target_tokens": target_len,
                    "target_fully_supervised": target_fully_supervised,
                    "target_complete_before_cutoff": target_len == len(pairs[turn_index][1]),
                    "assistant_content_preserved": content_preserved,
                    "has_tool_call": "<tool_call>" in original_content,
                    "target_preview": decoded_target[:300],
                    "think_open_count": decoded_target.count("<think>"),
                    "think_close_count": decoded_target.count("</think>"),
                }
                if args.dump_decoded:
                    turn_report["decoded_target"] = decoded_target
                turn_reports.append(turn_report)
            if template.efficient_eos:
                expected_labels.append(tokenizer.eos_token_id)
            exact_label_match = labels == expected_labels
            all_turns_covered = len(turn_reports) == len(assistant_messages)
            passed = (
                exact_label_match
                and all_turns_covered
                and all(turn["source_content_masked"] for turn in turn_reports)
                and all(turn["target_fully_supervised"] for turn in turn_reports)
                and all(turn["target_complete_before_cutoff"] for turn in turn_reports)
                and all(turn["assistant_content_preserved"] for turn in turn_reports)
            )
            if not passed:
                errors.append(f"line {line_number}: label coverage or masking check failed")
            audit = {
                "line": line_number,
                "message_count": len(row["messages"]),
                "assistant_turns_expected": len(assistant_messages),
                "assistant_turns_covered": len(turn_reports),
                "input_tokens": len(input_ids),
                "supervised_tokens": sum(label != IGNORE_INDEX for label in labels),
                "exact_processor_label_match": exact_label_match,
                "passed": passed,
                "turns": turn_reports,
                "think_open_count": sum(text.count("<think>") for text in decoded_targets),
                "think_close_count": sum(text.count("</think>") for text in decoded_targets),
            }
            if args.dump_decoded:
                audit["decoded_input_ids"] = tokenizer.decode(
                    input_ids, skip_special_tokens=False
                )
            audits.append(audit)
        except Exception as exc:
            errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")

    report = {
        "schema_version": "guqin-sft-label-audit-1.0",
        "valid": not errors,
        "mask_history": False,
        "train_on_prompt": False,
        "cutoff_len": args.cutoff_len,
        "seed": args.seed,
        "sample_count": len(audits),
        "errors": errors,
        "audits": audits,
        "template_token_summary": {
            "assistant_targets": sum(
                audit["assistant_turns_covered"] for audit in audits
            ),
            "think_open_count": sum(audit["think_open_count"] for audit in audits),
            "think_close_count": sum(audit["think_close_count"] for audit in audits),
            "targets_with_think_open": sum(
                turn["think_open_count"] > 0
                for audit in audits for turn in audit["turns"]
            ),
            "targets_with_think_close": sum(
                turn["think_close_count"] > 0
                for audit in audits for turn in audit["turns"]
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "audits"}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
