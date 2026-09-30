#!/usr/bin/env python3
"""Evaluate held-out scores with the production public agent loop.

``--workflow two_stage`` runs Base -> Guqinizer -> next phrase.  ``--workflow
single_stage`` runs the direct-final agent used by the single-stage teacher
corpus, from a blank plan -> next phrase.  No annotation/reference plan is
loaded in either workflow; the model's finalized plan is the only read-only
history passed to the following phrase.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the evaluator usable on bare servers
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 1.2: public evaluation inputs now carry continuous event_index values.
# Older prediction files must not be resumed against that different protocol.
EVAL_SCHEMA_VERSION = "agent-eval-prediction-1.3"


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
            # vLLM's OpenAI-compatible response separates ordinary assistant
            # content from tool calls.  Preserve that neutral terminology in
            # evaluation traces; ``decision_summary`` is reserved for the
            # teacher-trajectory protocol below.
            if isinstance(value.get("assistant_content"), str):
                return value["assistant_content"].strip()
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
    parser.add_argument(
        "--vllm-url",
        help="optional vLLM OpenAI-compatible base URL, e.g. http://127.0.0.1:8101/v1",
    )
    parser.add_argument(
        "--vllm-model", default="guqin-sft",
        help="served model/LoRA alias when --vllm-url is set",
    )
    parser.add_argument("--vllm-timeout-seconds", type=float, default=1800)
    parser.add_argument(
        "--vllm-max-model-len", type=int, default=16384,
        help="vLLM server context limit; caps each request's output budget dynamically",
    )
    parser.add_argument(
        "--adapter",
        help="optional LoRA adapter; omit to evaluate the unadapted base model",
    )
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--generation-batch-size", type=int, default=1,
        help=("batch ready generation turns from independent scores. "
              "Each score remains phrase-sequential; default 1 preserves the "
              "original serial loop."),
    )
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="render Qwen3.5 generation prompts with enable_thinking=False",
    )
    parser.add_argument(
        "--workflow", choices=("two_stage", "single_stage"), default="two_stage",
        help=("agent workflow: two_stage is Base→Guqinizer; single_stage is the "
              "direct-final public prompt and tool surface used to construct the "
              "single-stage training corpus"),
    )
    # Compatibility for existing launch commands.  Keep the canonical
    # user-facing switch above so a run's workflow is explicit in its command.
    parser.add_argument(
        "--single-stage", dest="workflow", action="store_const", const="single_stage",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--constrain-walk-hui", action="store_true",
        help="constrain Guqinizer walk endpoints (edit_plan.jianzi_rows) to the"
             " Base stage's hui or pitch-correct positions via token masking",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true",
                        help="disable score/phrase progress bars")
    parser.add_argument("--stop-after-scores", type=int,
                        help="gracefully stop after N fully processed scores; useful for first-score verification")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.single_stage = args.workflow == "single_stage"
    if args.generation_batch_size < 1:
        raise SystemExit("--generation-batch-size must be positive")
    if args.generation_batch_size > 1 and args.constrain_walk_hui:
        raise SystemExit(
            "cross-score batching is not yet compatible with --constrain-walk-hui; "
            "use batch size 1 or disable the constraint"
        )
    if args.vllm_url and args.constrain_walk_hui:
        raise SystemExit(
            "--constrain-walk-hui requires an in-process logits processor and is "
            "not available through the vLLM HTTP backend"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from ABC_J.scripts.generate_teacher_tool_trajectories import (
        blank_plan_from_item, canonical_jianzi_text, public_system_for,
        public_tools_for, render_public_prompt,
    )
    from agents.ToolRuntime import (
        RealToolRuntime, harmonic_region_at_phrase_start,
        public_pitch_warning_source_indices, validate_jianzi_only,
    )
    from agents.abc_to_jianzipu.trajectory_replay import replay_patches
    from scripts.adapter_loading import load_adapter_checked
    from scripts.constrained_decoding.guqinizer_walk_constraint import (
        WalkHuiConstraintProcessor, build_id_texts, build_walk_constraints,
    )
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
        stages = (("single_stage", False),) if args.single_stage else (
            ("fingering_agent", True), ("guqinization", False)
        )
        for stage, basic in stages:
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
    model = None
    if not args.vllm_url:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto", quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            ),
        )
        if args.adapter:
            model = load_adapter_checked(model, args.adapter)
    generation_eos_ids = qwen35_eos_token_ids(tokenizer)
    id_texts_cache: dict[int, str] | None = None

    def generate_vllm(messages: list[dict], tools: list[dict], sampled: bool):
        """Submit one complete tool turn to the shared continuous-batching server."""
        import time
        import urllib.error
        import urllib.request

        # The local template accepts tool arguments as mappings, whereas the
        # OpenAI-compatible HTTP API requires their JSON-string representation.
        # Keep the local conversation untouched and convert only the wire copy.
        wire_messages = deepcopy(messages)
        for wire_message in wire_messages:
            for tool_call in wire_message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                if not isinstance(function.get("arguments"), str):
                    function["arguments"] = json.dumps(
                        function.get("arguments") or {}, ensure_ascii=False
                    )
        wire_tools = openai_tools(tools)
        # Token counting uses Qwen's native template, which requires tool
        # arguments to remain mappings.  Only the OpenAI wire payload uses
        # the JSON-string form above.
        rendered = tokenizer.apply_chat_template(
            messages, tools=wire_tools, add_generation_prompt=True,
            enable_thinking=not args.disable_thinking, return_tensors="pt",
        )
        # Qwen's remote tokenizer returns a BatchEncoding here, while other
        # tokenizer implementations return a tensor directly.
        input_ids = getattr(rendered, "input_ids", rendered)
        prompt_tokens = int(input_ids.shape[-1])
        available_output = args.vllm_max_model_len - prompt_tokens
        if available_output < 1:
            raise RuntimeError(
                f"vLLM prompt already exceeds context: {prompt_tokens} >= "
                f"{args.vllm_max_model_len}"
            )
        request_max_tokens = min(args.max_new_tokens, available_output)
        payload = {
            "model": args.vllm_model,
            "messages": wire_messages,
            "tools": wire_tools,
            "tool_choice": "auto",
            "max_tokens": request_max_tokens,
            "temperature": 0.25 if sampled else 0.0,
            "repetition_penalty": 1.04,
            "chat_template_kwargs": {
                "enable_thinking": not args.disable_thinking,
            },
        }
        if prompt_tokens > 8192:
            context_dir = args.output.parent / "context_over_8192"
            context_dir.mkdir(parents=True, exist_ok=True)
            context_path = context_dir / f"request_{int(time.time() * 1000)}.json"
            context_path.write_text(json.dumps({
                "kind": "prompt_over_8192",
                "prompt_tokens": prompt_tokens,
                "requested_max_tokens": request_max_tokens,
                "model_context_limit": args.vllm_max_model_len,
                "request": payload,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        request = urllib.request.Request(
            args.vllm_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error = None
        for retry in range(5):
            try:
                with urllib.request.urlopen(
                    request, timeout=args.vllm_timeout_seconds
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                if error.code == 400:
                    debug_dir = args.output.parent / (args.output.stem + "_debug")
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = debug_dir / f"request_{int(time.time() * 1000)}.json"
                    debug_path.write_text(json.dumps({
                        "error": body,
                        "prompt_tokens": prompt_tokens,
                        "requested_max_tokens": request_max_tokens,
                        "request": payload,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                last_error = RuntimeError(
                    f"vLLM HTTP {error.code}: {body[:1200]}"
                )
                if error.code < 500 and error.code != 429:
                    raise last_error from error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            if retry < 4:
                time.sleep(min(8.0, 0.5 * (2 ** retry)))
        else:
            raise RuntimeError(f"vLLM request failed after 5 attempts: {last_error}")

        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {result}")
        choice = choices[0]
        message = choice.get("message") or {}
        calls = []
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"vLLM returned invalid tool arguments: {arguments[:500]}"
                    ) from error
            calls.append({"name": function.get("name"), "arguments": arguments})
        content = message.get("content") or ""
        if calls:
            # Keep the evaluator trace parseable while preserving the OpenAI
            # distinction between ordinary assistant content and tool calls.
            # ``decision_summary`` belongs only to teacher trajectories.
            raw = json.dumps({
                "assistant_content": content,
                "tool_calls": calls,
            }, ensure_ascii=False)
        else:
            raw = content
        usage = result.get("usage") or {}
        generated_tokens = int(usage.get("completion_tokens") or 0)
        return raw, choice.get("finish_reason") == "length", generated_tokens, None

    def generate(
        messages: list[dict], tools: list[dict], sampled: bool,
        constraint_table=None,
    ):
        nonlocal id_texts_cache
        if args.vllm_url:
            return generate_vllm(messages, tools, sampled)
        encoded = tokenizer.apply_chat_template(
            messages, tools=openai_tools(tools), add_generation_prompt=True,
            enable_thinking=not args.disable_thinking,
            return_tensors="pt", return_dict=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        processor = None
        if constraint_table is not None:
            if id_texts_cache is None:
                id_texts_cache = build_id_texts(tokenizer)
            processor = WalkHuiConstraintProcessor(
                id_texts_cache, constraint_table,
                int(encoded["input_ids"].shape[-1]),
            )
        options = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": sampled,
            "repetition_penalty": 1.04,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": generation_eos_ids,
        }
        if processor is not None:
            options["logits_processor"] = [processor]
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
            processor,
        )

    def generate_batch(requests: list[dict]) -> list[tuple[str, bool, int, None] | dict]:
        """Generate one ready turn per independent score in a single batch.

        A request owns its complete agent state, so only the model forward pass
        is shared. Tool execution and the subsequent state transition remain
        strictly per request.  Walk constraints are intentionally rejected at
        argument parsing because their logits processor has a single sequence
        state machine rather than a batch-indexed one.
        """
        if not requests:
            return []
        if args.vllm_url:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                futures = [executor.submit(
                    generate_vllm, request["messages"], request["tools"],
                    bool(request["sampled"]),
                ) for request in requests]
                results = []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        # One malformed model tool call must not tear down an
                        # entire worker shard.  The state machine feeds this
                        # back as a normal corrective user observation.
                        results.append({"generation_error": (
                            f"{type(exc).__name__}: {exc}"
                        )})
                return results
        sampled = {bool(request["sampled"]) for request in requests}
        if len(sampled) != 1:
            raise ValueError("batch must contain requests with the same sampling mode")
        encoded_rows = []
        for request in requests:
            encoded_rows.append(tokenizer.apply_chat_template(
                request["messages"], tools=openai_tools(request["tools"]),
                add_generation_prompt=True, enable_thinking=not args.disable_thinking,
                return_tensors="pt", return_dict=True,
            ))
        max_length = max(int(row["input_ids"].shape[-1]) for row in encoded_rows)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        input_ids = torch.full(
            (len(encoded_rows), max_length), int(pad_id), dtype=torch.long,
        )
        attention_mask = torch.zeros((len(encoded_rows), max_length), dtype=torch.long)
        for index, row in enumerate(encoded_rows):
            ids = row["input_ids"][0]
            mask = row.get("attention_mask")
            length = int(ids.shape[-1])
            input_ids[index, max_length - length:] = ids
            if mask is None:
                attention_mask[index, max_length - length:] = 1
            else:
                attention_mask[index, max_length - length:] = mask[0]
        options = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": sampled.pop(),
            "repetition_penalty": 1.04,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": generation_eos_ids,
        }
        if options["do_sample"]:
            options["temperature"] = 0.25
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                **options,
            )
        eos_ids = {int(value) for value in generation_eos_ids}
        results = []
        for row in output:
            generated = row[max_length:]
            values = [int(value) for value in generated.tolist()]
            actual = len(values)
            for position, value in enumerate(values):
                if value in eos_ids:
                    actual = position + 1
                    break
            new_tokens = generated[:actual]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=False)
            results.append((
                trim_qwen35_assistant_turn(decoded),
                actual >= args.max_new_tokens,
                actual,
                None,
            ))
        return results

    def run_stage_steps(item: dict, stage: str, historical: dict):
        """Yield one model-generation request at a time for a stage."""
        basic = stage == "fingering_agent"
        tools = public_tools_for(stage, basic=basic)
        constrain = args.constrain_walk_hui and not basic
        if stage in {"guqinization", "single_stage"}:
            item["public_pitch_warning_source_indices"] = sorted(
                public_pitch_warning_source_indices(item, historical=historical)
            )
        else:
            item.pop("public_pitch_warning_source_indices", None)
        # Mirror the teacher runner's pitch gate: a jianzi_pitch_mismatch
        # warning blocks acceptance until the model has actually re-edited
        # that event in a LATER round (a same-turn rewrite that still warns
        # does not count).  The model decides when to stop — acceptance also
        # happens when it stops calling tools with a clean last preview.
        source_to_event = {
            int(note["index"]): int(note["event_index"])
            for note in item["input"].get("notes_without_jianzi") or []
            if note.get("index") is not None and note.get("event_index") is not None
        }
        last_trace = []
        for attempt in range(args.attempts):
            runtime = RealToolRuntime(item, historical, basic_fingering=basic)
            messages = [
                {"role": "system", "content": public_system_for(stage, basic=basic)},
                {"role": "user", "content": render_public_prompt(item, stage)},
            ]
            trace = []
            # A warning asks for one deliberate follow-up edit, not an
            # optimizer-style requirement to drive the warning count to zero.
            # Once the model has made that later edit, it may decide to stop.
            warning_followup_required = False
            saw_pitch_warning = False
            last_valid: list | None = None
            for round_number in range(1, args.max_rounds + 1):
                constraint_table = None
                if constrain:
                    # The allowed sets follow the *current* plan state: Base
                    # plus every edit this stage has already applied, so the
                    # live left-hand string reflects the Guqinizer's own
                    # earlier rewrites.
                    current_replay = replay_patches(
                        item["baseline_plan"], runtime.accumulated_patches,
                        strict_before=False,
                    )
                    current_text = {
                        int(action["source_index"]): action.get("jianzi_text")
                        for action in current_replay.actions
                    }
                    constraint_table = build_walk_constraints(item, current_text)
                generation = yield {
                    "messages": messages,
                    "tools": tools,
                    "sampled": attempt > 0,
                    "constraint_table": constraint_table,
                }
                if isinstance(generation, dict) and generation.get("generation_error"):
                    error = str(generation["generation_error"])
                    round_entry = {
                        "round": round_number,
                        "raw_output": "",
                        "truncated": False,
                        "tool_calls": [],
                        "tool_results": [],
                        "generation_error": error,
                    }
                    trace.append(round_entry)
                    print(json.dumps({
                        "event": "eval_generation_error",
                        "sample_id": item.get("trajectory_id"),
                        "stage": stage,
                        "attempt": attempt + 1,
                        "round": round_number,
                        "error": error,
                    }, ensure_ascii=False), flush=True)
                    messages.append({
                        "role": "user",
                        "content": (
                            "上一轮工具调用未能执行：模型输出的工具参数不是完整合法的 JSON，"
                            f"错误为 {error}。未执行任何工具，也未修改曲谱。"
                            "请重新调用一个工具；参数必须是完整 JSON，尤其确认数组、花括号和引号均已闭合。"
                        ),
                    })
                    continue
                raw, truncated, generated_tokens, processor = generation
                calls = parse_calls(raw)
                round_entry = {
                    "round": round_number, "raw_output": raw,
                    "truncated": truncated, "tool_calls": calls,
                    "tool_results": [],
                }
                if constraint_table is not None:
                    round_entry["constraint"] = {
                        "stats": dict(processor.stats),
                        "allowed_endpoints": constraint_table.preview_allowed(),
                    }
                trace.append(round_entry)
                print(json.dumps({
                    "event": "eval_round",
                    "sample_id": item.get("trajectory_id"),
                    "stage": stage,
                    "attempt": attempt + 1,
                    "round": round_number,
                    "generated_tokens": generated_tokens,
                    "tool_calls": [call.get("name") for call in calls],
                    "truncated": truncated,
                    **({"constraint_stats": dict(processor.stats)}
                       if processor is not None else {}),
                }, ensure_ascii=False), flush=True)
                if not calls:
                    if last_valid is not None and not warning_followup_required:
                        # The model chose to stop after a valid preview.
                        return {"ok": True, "no_op": False,
                                "plan": {"actions": last_valid}, "trace": trace,
                                "final_reply": visible_reasoning(raw) or (
                                    "工具预览已通过，当前段减字填写完成。"
                                    if basic else
                                    "工具预览已通过，当前段减字润色完成。"
                                )}
                    if (stage == "guqinization" and not truncated
                            and not warning_followup_required):
                        return {"ok": True, "no_op": True,
                                "plan": item["baseline_plan"], "trace": trace,
                                "final_reply": "逐音审阅完成，当前段无需修改。"}
                    # Do not accept an empty final answer immediately after
                    # a pitch warning: ask for one edit attempt.  This is not
                    # a demand that the attempt eliminates every warning.
                    if warning_followup_required:
                        error = (
                            "上一轮出现音高警告；请至少提交一次 edit_plan 尝试后再决定"
                            "是否结束。请依据上一轮工具返回核对相关音序。"
                        )
                    else:
                        error = "当前段仍有待填写的演奏音，不能直接结束；请提交需要修改的 jianzi_rows。"
                    messages.append({
                        "role": "user",
                        "content": json.dumps({
                            "tool_results": [{
                                "ok": False, "error": error,
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
                warned_now: set[int] = set()
                repeated_submission: list[int] | None = None
                for assistant_call, call in zip(assistant_calls, calls):
                    if call["name"] == "edit_plan":
                        submitted = [
                            row for row in (call.get("arguments") or {}).get("jianzi_rows") or []
                            if isinstance(row, list) and len(row) == 2
                            and isinstance(row[0], int) and not isinstance(row[0], bool)
                        ]
                        if submitted:
                            current = {
                                int(action["source_index"]): action.get("jianzi_text")
                                for action in replay_patches(
                                    item["baseline_plan"], runtime.accumulated_patches,
                                    strict_before=False,
                                ).actions
                            }
                            unchanged_events: list[int] = []
                            for event_index, text in submitted:
                                try:
                                    source_index = runtime._event_to_source(event_index)
                                except ValueError:
                                    break
                                if canonical_jianzi_text(text) == canonical_jianzi_text(
                                    current.get(source_index)
                                ):
                                    unchanged_events.append(event_index)
                            if len(unchanged_events) == len(submitted):
                                repeated_submission = unchanged_events
                    result = runtime.invoke(call["name"], call.get("arguments") or {})
                    round_entry["tool_results"].append({
                        "name": call["name"],
                        "arguments": call.get("arguments") or {},
                        "result": result,
                    })
                    messages.append({"role": "tool", "tool_call_id": assistant_call["id"],
                                     "name": call["name"], "content": json.dumps(result, ensure_ascii=False)})
                    if call["name"] != "edit_plan":
                        continue
                    if result.get("ok") and result.get("result", {}).get("valid"):
                        complete, report = validate_jianzi_only(
                            item, runtime.accumulated_patches,
                            toward_reference=False,
                            require_complete=basic or stage == "single_stage",
                            # Keep this acceptance/pitch-gate audit in the
                            # same musical state as RealToolRuntime.edit_plan.
                            # In particular, phrase-spanning harmonic state
                            # depends on the preceding phrases.
                            historical=historical,
                        )
                        if complete:
                            accepted = runtime.calls[-1]["result"]["result"]["preview_actions"]
                            warned_now = {
                                source_to_event.get(int(warning["source_index"]), int(warning["source_index"]))
                                for warning in report.get("warnings") or []
                                if warning.get("code") == "jianzi_pitch_mismatch"
                                and warning.get("source_index") is not None
                            }
                if accepted is not None:
                    last_valid = accepted
                    if repeated_submission:
                        round_entry["repeated_submission"] = sorted(
                            repeated_submission
                        )
                        return {
                            "ok": True, "no_op": False,
                            "plan": {"actions": accepted}, "trace": trace,
                            "final_reply": (
                                "检测到重复提交未改变的音序（"
                                + "、".join(str(index) for index in sorted(repeated_submission))
                                + "），采用当前有效预览并停止交互。"
                            ),
                        }
                    # Any edit after the first warning fulfills the required
                    # repair attempt.  Later warnings remain visible in the
                    # tool result, but do not force an endless loop.
                    if warning_followup_required:
                        warning_followup_required = False
                    if warned_now and not saw_pitch_warning:
                        saw_pitch_warning = True
                        warning_followup_required = True
                    round_entry["pitch_gate"] = {
                        "warned_now": sorted(warned_now),
                        "followup_edit_required": warning_followup_required,
                    }
                    # A valid preview is only a candidate final state. Keep
                    # it and let the model decide in the next assistant turn.
                    continue
            # Rounds exhausted: keep the last valid preview rather than
            # failing the phrase (a broken stage would poison the cascade).
            if last_valid is not None:
                return {"ok": True, "no_op": False,
                        "plan": {"actions": last_valid}, "trace": trace,
                        "final_reply": "轮次用尽，采用最后一份有效预览。",
                        "pitch_followup_edit_required_at_stop": warning_followup_required}
            last_trace = trace
        return {"ok": False, "trace": last_trace}

    def run_stage(item: dict, stage: str, historical: dict) -> dict:
        """Original serial execution path, implemented via the same state machine."""
        steps = run_stage_steps(item, stage, historical)
        try:
            request = next(steps)
            while True:
                try:
                    result = generate(
                        request["messages"], request["tools"], request["sampled"],
                        request["constraint_table"],
                    )
                except Exception as exc:
                    result = {"generation_error": f"{type(exc).__name__}: {exc}"}
                request = steps.send(result)
        except StopIteration as stopped:
            return stopped.value

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
            final_stage_key = "single_stage" if args.single_stage else "guqinizer"
            if (
                prior.get("protocol_valid")
                and isinstance((prior.get(final_stage_key) or {}).get("plan"), dict)
            ):
                completed[str(prior["sample_id"])] = prior

    def run_batched_scores(score_keys: list[str]) -> int:
        """Dynamically batch ready turns while preserving each score's order."""
        progress_enabled = not args.no_progress and tqdm is not None
        score_progress = tqdm(total=len(score_keys), desc="曲谱", unit="首", file=sys.stdout,
                              dynamic_ncols=True) if progress_enabled else None
        phrase_progress = tqdm(total=sum(len(by_score[key]) for key in score_keys),
                                desc="phrase", unit="条", file=sys.stdout,
                                dynamic_ncols=True) if progress_enabled else None
        states = {
            key: {"score_key": key, "phrases": by_score[key], "position": 0,
                  "previous": None, "historical": {}, "finished": False}
            for key in score_keys
        }

        def finish(state: dict) -> None:
            if not state["finished"]:
                state["finished"] = True
                if score_progress:
                    score_progress.update(1)

        def prepare(state: dict):
            while state["position"] < len(state["phrases"]):
                item = state["phrases"][state["position"]]
                sample_id = str(item["trajectory_id"])
                handoff = item["input"].setdefault("phrase_handoff", {})
                handoff.pop("previous_phrase", None)
                if state["previous"]:
                    previous = state["previous"]
                    handoff["previous_phrase"] = {
                        "phrase_id": previous["phrase_id"], "status": "confirmed_readonly",
                        "notes": previous["input"]["notes_without_jianzi"],
                        "actions": previous["reference_plan"]["actions"],
                    }
                item["harmonic_region_at_start"] = harmonic_region_at_phrase_start(
                    item, state["historical"]
                )
                prior = completed.get(sample_id)
                if prior:
                    final_key = "single_stage" if args.single_stage else "guqinizer"
                    final_plan = deepcopy(prior[final_key]["plan"])
                    item["baseline_plan"] = final_plan
                    item["reference_plan"] = final_plan
                    state["previous"] = item
                    state["historical"][(state["score_key"], item["phrase_id"])] = item
                    state["position"] += 1
                    if phrase_progress:
                        phrase_progress.update(1)
                    continue
                item["baseline_plan"] = blank_plan_from_item(item)
                first_stage = "single_stage" if args.single_stage else "fingering_agent"
                steps = run_stage_steps(item, first_stage, state["historical"])
                return {"state": state, "item": item, "stage": first_stage,
                        "steps": steps, "request": next(steps), "base": None}
            finish(state)
            return None

        def write_record(session: dict, base: dict, guqinizer: dict | None, output) -> bool:
            state, item = session["state"], session["item"]
            single_stage = args.single_stage
            final = (guqinizer if single_stage else guqinizer) or {"ok": False}
            valid = bool(final.get("ok")) if single_stage else bool(base.get("ok") and final.get("ok"))
            record = {
                "schema_version": EVAL_SCHEMA_VERSION,
                "sample_id": str(item["trajectory_id"]), "split": item.get("split"),
                "score_key": state["score_key"], "phrase_id": item["phrase_id"],
                "input_sha256": source_meta.get(str(item["trajectory_id"]), {}).get("input_sha256"),
                "protocol_valid": valid,
            }
            if single_stage:
                record["single_stage"] = final
            else:
                record["base"] = base
                if guqinizer is not None:
                    record["guqinizer"] = guqinizer
            if valid:
                final_plan = final["plan"]
                record["jianzi_rows"] = [
                    [int(action["source_index"]), str(action.get("jianzi_text") or "")]
                    for action in final_plan.get("actions", [])
                ]
                record["parse_error"] = None
            else:
                final_plan = None
                record["jianzi_rows"] = []
                record["parse_error"] = (
                    "single_stage_failed" if single_stage else
                    ("base_stage_failed" if not base.get("ok") else "guqinizer_stage_failed")
                )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if phrase_progress:
                phrase_progress.update(1)
            if final_plan is None:
                finish(state)
                return False
            item["reference_plan"] = final_plan
            state["previous"] = item
            state["historical"][(state["score_key"], item["phrase_id"])] = item
            state["position"] += 1
            return True

        mode = "a" if args.resume else "w"
        sessions = [session for state in states.values() if (session := prepare(state))]
        with args.output.open(mode, encoding="utf-8", newline="\n") as output:
            while sessions:
                sampled = bool(sessions[0]["request"]["sampled"])
                batch = [session for session in sessions
                         if bool(session["request"]["sampled"]) == sampled][:args.generation_batch_size]
                results = generate_batch([session["request"] for session in batch])
                for session, result in zip(batch, results):
                    try:
                        session["request"] = session["steps"].send(result)
                        continue
                    except StopIteration as stopped:
                        stage_result = stopped.value
                    if session["stage"] == "fingering_agent" and stage_result.get("ok"):
                        session["base"] = stage_result
                        session["item"]["baseline_plan"] = stage_result["plan"]
                        steps = run_stage_steps(session["item"], "guqinization",
                                                session["state"]["historical"])
                        session.update({"stage": "guqinization", "steps": steps,
                                        "request": next(steps)})
                        continue
                    if session["stage"] == "single_stage":
                        keep_score = write_record(session, None, stage_result, output)
                    elif session["stage"] == "fingering_agent":
                        keep_score = write_record(session, stage_result, None, output)
                    else:
                        keep_score = write_record(session, session["base"], stage_result, output)
                    if keep_score and (next_session := prepare(session["state"])) is not None:
                        session.update(next_session)
                        continue
                    sessions.remove(session)
        if phrase_progress:
            phrase_progress.close()
        if score_progress:
            score_progress.close()
        return 0

    score_keys = sorted(by_score)
    if args.generation_batch_size > 1:
        if args.stop_after_scores is not None:
            score_keys = score_keys[:args.stop_after_scores]
        return run_batched_scores(score_keys)
    progress_enabled = not args.no_progress and tqdm is not None
    score_progress = tqdm(total=len(score_keys), desc="曲谱", unit="首", file=sys.stdout,
                          dynamic_ncols=True) if progress_enabled else None
    mode = "a" if args.resume else "w"
    processed_scores = 0
    with args.output.open(mode, encoding="utf-8", newline="\n") as output:
        for score_key in score_keys:
            previous = None
            historical = {}
            phrases = by_score[score_key]
            phrase_progress = (
                tqdm(total=len(phrases), desc=f"{score_key} phrase", unit="条",
                     file=sys.stdout, dynamic_ncols=True, leave=False)
                if progress_enabled else None
            )
            for item in phrases:
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
                    final_key = "single_stage" if args.single_stage else "guqinizer"
                    final_plan = deepcopy(prior[final_key]["plan"])
                    item["baseline_plan"] = final_plan
                    item["reference_plan"] = final_plan
                    previous = item
                    historical[(score_key, item["phrase_id"])] = item
                    if phrase_progress:
                        phrase_progress.update(1)
                    continue

                item["baseline_plan"] = blank_plan_from_item(item)
                if args.single_stage:
                    direct = run_stage(item, "single_stage", historical)
                    record = {
                        "schema_version": EVAL_SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "split": item.get("split"),
                        "score_key": score_key,
                        "phrase_id": item["phrase_id"],
                        "input_sha256": source_meta.get(sample_id, {}).get("input_sha256"),
                        "single_stage": direct,
                        "protocol_valid": bool(direct.get("ok")),
                    }
                    if record["protocol_valid"]:
                        final_plan = direct["plan"]
                        record["jianzi_rows"] = [
                            [int(action["source_index"]), str(action.get("jianzi_text") or "")]
                            for action in final_plan.get("actions", [])
                        ]
                        record["parse_error"] = None
                    else:
                        final_plan = None
                        record["jianzi_rows"] = []
                        record["parse_error"] = "single_stage_failed"
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    if phrase_progress:
                        phrase_progress.update(1)
                    if final_plan is None:
                        # Do not feed a later phrase an incomplete predecessor.
                        break
                    item["reference_plan"] = final_plan
                    previous = item
                    historical[(score_key, item["phrase_id"])] = item
                    continue

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
                if phrase_progress:
                    phrase_progress.update(1)
                if not record["protocol_valid"]:
                    # Later phrases must not be evaluated with missing or gold
                    # previous context. Continue with the next independent score.
                    break
                item["reference_plan"] = final_plan
                previous = item
                historical[(score_key, item["phrase_id"])] = item
            if phrase_progress:
                phrase_progress.close()
            processed_scores += 1
            if score_progress:
                score_progress.update(1)
            if (args.stop_after_scores is not None
                    and processed_scores >= args.stop_after_scores):
                print(json.dumps({"event": "eval_stop_after_scores",
                                  "processed_scores": processed_scores,
                                  "score_key": score_key}, ensure_ascii=False), flush=True)
                break
    if score_progress:
        score_progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
