#!/usr/bin/env python3
"""Export one schema-stable Qwen3.5 SFT example per complete Agent trajectory.

Assistant reasoning and tool calls are serialized together as assistant text so
LLaMA-Factory's OpenAI converter cannot overwrite the reasoning.  Tool results
remain as context.  Legacy accepted trajectories that end at a successful tool
result receive a deterministic assistant completion instead of losing that
result.  A reasoned no-op trajectory may contain no tool call at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "ABC_J" / "agent_training" / "messages_batch_glm_40_redacted"
DEFAULT_OUTPUT = ROOT / "train" / "data" / "guqin_agent_sft_v2_full_trajectory"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def qwen35_tool_text(tool_calls: list[dict[str, Any]]) -> str:
    """Use the exact Qwen3.5 function surface used by LLaMA-Factory 0.9.6."""
    blocks: list[str] = []
    for call in tool_calls:
        function = call.get("function") or {}
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not name or not isinstance(arguments, dict):
            raise ValueError("tool call requires a function name and object arguments")
        lines = ["<tool_call>", f"<function={name}>"]
        for key, value in arguments.items():
            surface = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            lines.extend([f"<parameter={key}>", str(surface), "</parameter>"])
        lines.extend(["</function>", "</tool_call>"])
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def normalize_tools(tools: list[dict[str, Any]]) -> str:
    """Keep tools a JSON string so Arrow never infers a heterogeneous schema."""
    normalized = []
    for tool in tools:
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(tool)
            continue
        normalized.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
            },
        })
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def public_message(message: dict[str, Any], *, wrap_noop_reasoning: bool = False) -> dict[str, str]:
    role = str(message.get("role") or "")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported message role: {role}")
    content = str(message.get("content") or "")
    if role == "assistant" and message.get("tool_calls"):
        # Qwen3.5's reasoning-aware template otherwise inserts an *empty*
        # <think> block and leaves our public reasoning in the answer channel.
        # Keep reasoning supervised as reasoning, while tool calls remain after
        # </think> in the model's native surface form.
        reasoning = content.strip()
        if reasoning and not (reasoning.startswith("<think>") and "</think>" in reasoning):
            content = f"<think>\n{reasoning}\n</think>"
        call_text = qwen35_tool_text(message["tool_calls"])
        content = f"{content}{'' if content.endswith(chr(10)) else chr(10)}{call_text}" if content else call_text
    elif role == "assistant" and wrap_noop_reasoning and content.strip():
        reasoning = content.strip()
        if not (reasoning.startswith("<think>") and "</think>" in reasoning):
            content = f"<think>\n{reasoning}\n</think>\n无需修改。"
    return {"role": role, "content": content}


def merge_adjacent_assistants(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce adjacent assistant records for OpenAI-style role alternation.

    Some accepted teacher logs contain a short reasoning-only assistant record
    immediately before the next tool-call assistant record.  Keeping those as
    two messages violates LLaMA-Factory's user/assistant alternation, while
    merging them preserves every reasoning token and every tool call in order.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        current = dict(message)
        if merged and merged[-1].get("role") == "assistant" and current.get("role") == "assistant":
            previous = merged[-1]
            previous_content = str(previous.get("content") or "")
            current_content = str(current.get("content") or "")
            if previous_content and current_content:
                previous["content"] = previous_content + "\n" + current_content
            elif current_content:
                previous["content"] = current_content
            previous_calls = list(previous.get("tool_calls") or [])
            current_calls = list(current.get("tool_calls") or [])
            if previous_calls or current_calls:
                previous["tool_calls"] = previous_calls + current_calls
            else:
                previous.pop("tool_calls", None)
            continue
        merged.append(current)
    return merged


def build_trajectory_example(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = merge_adjacent_assistants(list(row.get("messages") or []))
    if len(messages) < 3 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("expected system/user prefix")
    appended_terminal_assistant = False
    if messages and messages[-1].get("role") == "tool":
        if (row.get("termination") or {}).get("kind") != "accepted_edit_plan":
            raise ValueError("terminal tool result is only valid for an accepted edit plan")
        stage = str(row.get("agent_stage") or "")
        messages.append({
            "role": "assistant",
            "content": (
                "工具预览已通过，当前段基础减字填写完成。"
                if stage == "fingering_agent"
                else ("工具预览已通过，当前段最终减字填写完成。"
                      if stage == "single_stage"
                      else "工具预览已通过，当前段减字润色完成。")
            ),
        })
        appended_terminal_assistant = True
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("trajectory must end with assistant")
    assistant_turns = sum(message.get("role") == "assistant" for message in messages)
    tool_call_turns = sum(
        message.get("role") == "assistant" and bool(message.get("tool_calls")) for message in messages
    )
    termination_kind = (row.get("termination") or {}).get("kind")
    if not assistant_turns:
        raise ValueError("trajectory has no trainable assistant turn")
    if not tool_call_turns and termination_kind != "no_changes":
        raise ValueError("trajectory without tool calls must be an explicit no_changes sample")
    example = {
        "messages": [
            public_message(
                message,
                wrap_noop_reasoning=(
                    termination_kind == "no_changes"
                    and index == len(messages) - 1
                    and message.get("role") == "assistant"
                    and not message.get("tool_calls")
                ),
            )
            for index, message in enumerate(messages)
        ],
        "tools": normalize_tools(row.get("tools") or []),
    }
    metadata = {
        "source_sample_id": row["sample_id"],
        "source_trajectory_id": (row.get("provenance") or {}).get("source_trajectory_id"),
        "score_key": (row.get("provenance") or {}).get("score_key"),
        "split": (row.get("provenance") or {}).get("split"),
        "agent_stage": row.get("agent_stage"),
        "assistant_turn_count": assistant_turns,
        "tool_call_turn_count": tool_call_turns,
        "appended_terminal_assistant": appended_terminal_assistant,
    }
    return example, metadata


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit-source", type=int, help="For smoke exports only.")
    args = parser.parse_args()
    source_path = args.input_dir / "messages_train.jsonl"
    validation_path = args.input_dir / "messages_validation_report.json"
    selection_path = args.input_dir / "selection_manifest.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("valid"):
            raise SystemExit("input messages_validation_report.json is not valid")
    elif selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("schema_version") != "pitch-eligible-trajectory-export-1.0":
            raise SystemExit("unsupported selection_manifest.json")
        expected_rows = int((selection.get("counts") or {}).get("exported_message_rows", -1))
        actual_rows = sum(1 for line in source_path.open(encoding="utf-8") if line.strip())
        if actual_rows != expected_rows:
            raise SystemExit(
                f"selection manifest row mismatch: expected {expected_rows}, got {actual_rows}"
            )
    else:
        raise SystemExit("input needs a valid messages_validation_report.json or supported selection_manifest.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    by_stage: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    assistant_turns = 0
    tool_call_turns = 0
    appended_terminal_assistants = 0
    source_rows = 0
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if args.limit_source is not None and source_rows >= args.limit_source:
                break
            source_rows += 1
            row = json.loads(line)
            example, meta = build_trajectory_example(row)
            exported.append(example)
            meta["training_trajectory_id"] = meta["source_sample_id"]
            metadata.append(meta)
            by_stage[str(meta["agent_stage"])] += 1
            by_split[str(meta["split"])] += 1
            assistant_turns += int(meta["assistant_turn_count"])
            tool_call_turns += int(meta["tool_call_turn_count"])
            appended_terminal_assistants += int(meta["appended_terminal_assistant"])
    if not exported:
        raise SystemExit("no complete assistant trajectories exported")
    if set(by_split) != {"train"}:
        raise SystemExit(f"training export must contain only train split; got {dict(by_split)}")

    dataset_path = args.output_dir / "guqin_agent_train.jsonl"
    metadata_path = args.output_dir / "guqin_agent_train_metadata.jsonl"
    write_jsonl(dataset_path, exported)
    write_jsonl(metadata_path, metadata)
    dataset_info = {
        "guqin_agent_train": {
            "file_name": dataset_path.name,
            "formatting": "openai",
            "columns": {"messages": "messages", "tools": "tools"},
            "tags": {
                "role_tag": "role", "content_tag": "content", "user_tag": "user",
                "assistant_tag": "assistant", "observation_tag": "tool",
                "function_tag": "function", "system_tag": "system",
            },
        }
    }
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "guqin-agent-sft-export-2.0",
        "source_dir": str(args.input_dir),
        "source_messages_sha256": sha256(source_path),
        "source_rows": source_rows,
        "training_rows": len(exported),
        "training_rows_by_stage": dict(sorted(by_stage.items())),
        "assistant_turns": assistant_turns,
        "assistant_tool_call_turns": tool_call_turns,
        "appended_terminal_assistants": appended_terminal_assistants,
        "split_policy": "source train only; no validation/test labels are exported",
        "serialization": "one complete trajectory per row; tool-turn and no-change public reasoning is enclosed in <think>...</think>; Qwen3.5 XML tool calls and final answers remain outside",
        "terminal_policy": "preserve tool observations; append a deterministic assistant stop to legacy accepted trajectories; allow reasoned no_changes without tools",
        "dataset_sha256": sha256(dataset_path),
        "metadata_sha256": sha256(metadata_path),
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
