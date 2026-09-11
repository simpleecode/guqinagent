#!/usr/bin/env python3
"""Evaluate held-out scores with the production two-stage public agent loop.

For each score the order is strictly:
Base -> Guqinizer -> next phrase.  No annotation/reference plan is loaded.
The finalized Guqinizer plan becomes the read-only previous phrase and the
only expandable history for later phrases in that score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_SCHEMA_VERSION = "agent-two-stage-eval-prediction-1.1"


def first_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[pos:])
            return value
        except json.JSONDecodeError:
            pass
    raise ValueError("no JSON value found")


def normalize_call(call: Any) -> dict | None:
    if not isinstance(call, dict):
        return None
    payload = call.get("function") if isinstance(call.get("function"), dict) else call
    name = payload.get("name")
    if not name:
        return None
    arguments = payload.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {"name": str(name), "arguments": arguments}


def parse_calls(text: str) -> list[dict]:
    try:
        value = first_json(text)
        if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
            return [normalized for call in value["tool_calls"]
                    if (normalized := normalize_call(call)) is not None]
    except ValueError:
        pass
    calls = []
    for name, body in re.findall(r"<function\s*=\s*([^>]+)>(.*?)</function>", text, re.S):
        arguments = {}
        for key, raw in re.findall(r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>", body, re.S):
            raw = raw.strip()
            try:
                arguments[key.strip()] = json.loads(raw)
            except json.JSONDecodeError:
                arguments[key.strip()] = raw
        calls.append({"name": name.strip(), "arguments": arguments})
    if calls:
        return calls
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        try:
            value = first_json(block)
            if isinstance(value, dict) and value.get("name"):
                normalized = normalize_call(value)
                if normalized is not None:
                    calls.append(normalized)
        except ValueError:
            pass
    return calls


def visible_reasoning(text: str) -> str:
    if "<tool_call>" in text:
        return text.split("<tool_call>", 1)[0].strip()
    try:
        value = first_json(text)
        if isinstance(value, dict):
            if isinstance(value.get("decision_summary"), str):
                return value["decision_summary"].strip()
            if isinstance(value.get("tool_calls"), list):
                return ""
    except ValueError:
        pass
    return text.strip()


def openai_tools(tools: list[dict]) -> list[dict]:
    result = []
    for tool in tools:
        if tool.get("type") == "function":
            result.append(tool)
        else:
            parameters = deepcopy(tool.get("input_schema") or {"type": "object"})
            # Qwen3.5's bundled chat template cannot render JSON-Schema oneOf
            # branches.  Runtime accepts both forms, so expose the simpler
            # scalar variant to the model.
            for prop in (parameters.get("properties") or {}).values():
                if isinstance(prop, dict) and isinstance(prop.get("oneOf"), list):
                    replacement = deepcopy(prop["oneOf"][0])
                    prop.clear()
                    prop.update(replacement)
            result.append({"type": "function", "function": {
                "name": tool["name"], "description": tool.get("description", ""),
                "parameters": parameters,
            }})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--score-key", help="evaluate one score; omit for every score in the input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="render Qwen3.5 generation prompts with enable_thinking=False",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from ABC_J.scripts.generate_teacher_tool_trajectories import (
        RealToolRuntime, blank_plan_from_item, harmonic_region_at_phrase_start,
        public_system_for, public_tools_for, render_public_prompt,
        validate_jianzi_only,
    )
    from scripts.adapter_loading import load_adapter_checked
    from scripts.qwen35_generation import qwen35_eos_token_ids, trim_qwen35_assistant_turn

    source = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in source
                if args.score_key is None or row["score_key"] == args.score_key]
    if not selected:
        raise SystemExit(f"score not found: {args.score_key}")
    for row in selected:
        runtime_item = row.get("runtime_item") or {}
        if row.get("reference") is not None or runtime_item.get("reference_plan") is not None:
            raise ValueError(f"private reference found in public eval input: {row.get('sample_id')}")
    by_score: dict[str, list[dict]] = {}
    source_meta: dict[str, dict] = {}
    for row in selected:
        item = deepcopy(row["runtime_item"])
        by_score.setdefault(str(row["score_key"]), []).append(item)
        source_meta[str(row["sample_id"])] = row
    for phrases in by_score.values():
        phrases.sort(key=lambda item: int(item["input"]["event_range"]["start"]))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if args.preflight_only:
        probe = next(iter(by_score.values()))[0]
        probe["baseline_plan"] = blank_plan_from_item(probe)
        for stage, basic in (("fingering_agent", True), ("guqinization", False)):
            tokenizer.apply_chat_template(
                [{"role": "system", "content": public_system_for(stage, basic=basic)},
                 {"role": "user", "content": render_public_prompt(probe, stage)}],
                tools=openai_tools(public_tools_for(stage, basic=basic)),
                add_generation_prompt=True,
                enable_thinking=not args.disable_thinking,
            )
        print(json.dumps({"preflight": "ok", "scores": len(by_score),
                          "phrases": len(selected)}, ensure_ascii=False))
        return 0
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="auto", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        ),
    )
    model = load_adapter_checked(model, args.adapter)
    generation_eos_ids = qwen35_eos_token_ids(tokenizer)

    def generate(messages: list[dict], tools: list[dict], sampled: bool) -> tuple[str, bool, int]:
        encoded = tokenizer.apply_chat_template(
            messages, tools=openai_tools(tools), add_generation_prompt=True,
            enable_thinking=not args.disable_thinking,
            return_tensors="pt", return_dict=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        options = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": sampled,
            "repetition_penalty": 1.04,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": generation_eos_ids,
        }
        if sampled:
            options["temperature"] = 0.25
        with torch.inference_mode():
            output = model.generate(**encoded, **options)
        new_tokens = output[0, encoded["input_ids"].shape[-1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
        return (
            trim_qwen35_assistant_turn(decoded),
            len(new_tokens) >= args.max_new_tokens,
            int(len(new_tokens)),
        )

    def run_stage(item: dict, stage: str, historical: dict) -> dict:
        basic = stage == "fingering_agent"
        tools = public_tools_for(stage, basic=basic)
        last_trace = []
        for attempt in range(args.attempts):
            runtime = RealToolRuntime(item, historical, basic_fingering=basic)
            messages = [
                {"role": "system", "content": public_system_for(stage, basic=basic)},
                {"role": "user", "content": render_public_prompt(item, stage)},
            ]
            trace = []
            for round_number in range(1, args.max_rounds + 1):
                raw, truncated, generated_tokens = generate(messages, tools, sampled=attempt > 0)
                calls = parse_calls(raw)
                print(json.dumps({
                    "event": "eval_round",
                    "sample_id": item.get("trajectory_id"),
                    "stage": stage,
                    "attempt": attempt + 1,
                    "round": round_number,
                    "generated_tokens": generated_tokens,
                    "tool_calls": [call.get("name") for call in calls],
                    "truncated": truncated,
                }, ensure_ascii=False), flush=True)
                round_trace = {"round": round_number, "raw_output": raw,
                               "truncated": truncated, "tool_calls": calls,
                               "tool_results": []}
                trace.append(round_trace)
                if not calls:
                    if stage == "guqinization" and not truncated:
                        return {"ok": True, "no_op": True,
                                "plan": item["baseline_plan"], "trace": trace,
                                "final_reply": "逐音审阅完成，当前段无需修改。"}
                    # This mirrors the teacher runner: an unfinished Base is
                    # told to continue rather than being silently accepted.
                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "tool_results": [{
                                "ok": False,
                                "error": "当前段仍有待填写的演奏音，不能直接结束；请提交需要修改的 jianzi_rows。",
                            }],
                            "instruction": "根据真实反馈继续；需要工具时继续调用工具。",
                        }, ensure_ascii=False),
                    })
                    continue
                assistant_calls = [{
                    "id": f"call_{round_number}_{number}", "type": "function",
                    # Qwen3.5's native template iterates arguments as a mapping;
                    # unlike OpenAI wire JSON, do not serialize it to a string
                    # when feeding the next local inference round.
                    "function": {"name": call["name"], "arguments": call.get("arguments") or {}},
                } for number, call in enumerate(calls, 1)]
                messages.append({"role": "assistant", "content": visible_reasoning(raw),
                                 "tool_calls": assistant_calls})
                accepted = None
                for assistant_call, call in zip(assistant_calls, calls):
                    result = runtime.invoke(call["name"], call.get("arguments") or {})
                    round_trace["tool_results"].append({
                        "name": call["name"],
                        "arguments": call.get("arguments") or {},
                        "result": result,
                    })
                    messages.append({"role": "tool", "tool_call_id": assistant_call["id"],
                                     "name": call["name"], "content": json.dumps(result, ensure_ascii=False)})
                    if call["name"] == "edit_plan" and result.get("ok") and result.get("result", {}).get("valid"):
                        complete = validate_jianzi_only(
                            item, runtime.accumulated_patches,
                            toward_reference=False,
                            require_complete=basic,
                        )[0]
                        if complete:
                            accepted = runtime.calls[-1]["result"]["result"]["preview_actions"]
                if accepted is not None:
                    return {"ok": True, "no_op": False,
                            "plan": {"actions": accepted}, "trace": trace,
                            "final_reply": (
                                "工具预览已通过，当前段基础减字填写完成。"
                                if basic else
                                "工具预览已通过，当前段减字润色完成。"
                            )}
            last_trace = trace
        return {"ok": False, "trace": last_trace}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            if prior.get("schema_version") != EVAL_SCHEMA_VERSION:
                raise ValueError(
                    f"resume output uses incompatible schema {prior.get('schema_version')!r}; "
                    "use a new --output path so pre-fix generations are not reused"
                )
            if (
                prior.get("protocol_valid")
                and isinstance((prior.get("guqinizer") or {}).get("plan"), dict)
            ):
                completed[str(prior["sample_id"])] = prior

    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as output:
        for score_key in sorted(by_score):
            previous = None
            historical = {}
            for item in by_score[score_key]:
                sample_id = str(item["trajectory_id"])
                handoff = item["input"].setdefault("phrase_handoff", {})
                handoff.pop("previous_phrase", None)
                if previous:
                    handoff["previous_phrase"] = {
                        "phrase_id": previous["phrase_id"],
                        "status": "confirmed_readonly",
                        "notes": previous["input"]["notes_without_jianzi"],
                        # This is the model's own preceding Guqinizer output,
                        # never the sealed/reference annotation.
                        "actions": previous["reference_plan"]["actions"],
                    }
                item["harmonic_region_at_start"] = harmonic_region_at_phrase_start(
                    item, historical
                )

                prior = completed.get(sample_id)
                if prior:
                    final_plan = deepcopy(prior["guqinizer"]["plan"])
                    item["baseline_plan"] = final_plan
                    item["reference_plan"] = final_plan
                    previous = item
                    historical[(score_key, item["phrase_id"])] = item
                    continue

                item["baseline_plan"] = blank_plan_from_item(item)
                base = run_stage(item, "fingering_agent", historical)
                record = {
                    "schema_version": EVAL_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "split": item.get("split"),
                    "score_key": score_key,
                    "phrase_id": item["phrase_id"],
                    "input_sha256": source_meta.get(sample_id, {}).get("input_sha256"),
                    "base": base,
                }
                if base["ok"]:
                    item["baseline_plan"] = base["plan"]
                    record["guqinizer"] = run_stage(item, "guqinization", historical)
                guqinizer = record.get("guqinizer") or {"ok": False}
                record["protocol_valid"] = bool(base.get("ok") and guqinizer.get("ok"))
                if record["protocol_valid"]:
                    final_plan = guqinizer["plan"]
                    record["jianzi_rows"] = [
                        [int(action["source_index"]), str(action.get("jianzi_text") or "")]
                        for action in final_plan.get("actions", [])
                    ]
                    record["parse_error"] = None
                else:
                    record["jianzi_rows"] = []
                    record["parse_error"] = (
                        "base_stage_failed" if not base.get("ok")
                        else "guqinizer_stage_failed"
                    )
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                if not record["protocol_valid"]:
                    # Later phrases must not be evaluated with missing or gold
                    # previous context. Continue with the next independent score.
                    break
                item["reference_plan"] = final_plan
                previous = item
                historical[(score_key, item["phrase_id"])] = item
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
