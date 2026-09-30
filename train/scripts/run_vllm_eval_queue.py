#!/usr/bin/env python3
"""Run public agent evaluation through one vLLM queue service."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def patch_evaluator(source: Path, destination: Path) -> None:
    """Make an isolated vLLM-enabled copy; leave the production evaluator intact."""
    text = source.read_text(encoding="utf-8")
    # The production evaluator now owns the HTTP backend.  Copy it verbatim
    # instead of applying the legacy textual transformation a second time.
    if 'parser.add_argument(\n        "--vllm-url"' in text:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return
    anchor = '    parser.add_argument("--base-model", required=True)\n'
    addition = anchor + '''    parser.add_argument("--vllm-url", help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--vllm-model", default="guqin-sft")
    parser.add_argument("--vllm-timeout-seconds", type=float, default=1800)
'''
    if text.count(anchor) != 1:
        raise RuntimeError("evaluator CLI anchor changed; refusing to patch")
    text = text.replace(anchor, addition)
    anchor = "    args = parser.parse_args()\n"
    addition = anchor + '''    if args.vllm_url and args.constrain_walk_hui:
        raise SystemExit("vLLM HTTP backend cannot use the local walk-hui logits processor")
'''
    if text.count(anchor) != 1:
        raise RuntimeError("evaluator argument-validation anchor changed")
    text = text.replace(anchor, addition)

    start = text.find("    model = AutoModelForCausalLM.from_pretrained(\n")
    end = text.find("    generation_eos_ids = qwen35_eos_token_ids(tokenizer)\n", start)
    if start < 0 or end < 0 or text.count("    model = AutoModelForCausalLM.from_pretrained(\n") != 1:
        raise RuntimeError("could not locate evaluator model-loading block")
    block = text[start:end]
    if "if args.adapter:" not in block:
        raise RuntimeError("expected adapter-loading block not found")
    block = "    model = None\n    if not args.vllm_url:\n" + "".join(
        "    " + line if line.strip() else line for line in block.splitlines(keepends=True)
    )
    text = text[:start] + block + text[end:]

    anchor = "    def generate(\n        messages: list[dict], tools: list[dict], sampled: bool,\n"
    helper = '''    def generate_vllm(messages: list[dict], tools: list[dict], sampled: bool):
        import time
        import urllib.error
        import urllib.request

        # The production local chat loop keeps function arguments as mappings
        # for Qwen's native template. OpenAI-compatible HTTP messages require
        # them as JSON strings, so convert only a deep copy on the wire.
        wire_messages = deepcopy(messages)
        for wire_message in wire_messages:
            for tool_call in wire_message.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    function["arguments"] = json.dumps(arguments or {}, ensure_ascii=False)

        payload = {
            "model": args.vllm_model,
            "messages": wire_messages,
            "tools": openai_tools(tools),
            "tool_choice": "auto",
            "max_tokens": args.max_new_tokens,
            "temperature": 0.25 if sampled else 0.0,
            "repetition_penalty": 1.04,
            "chat_template_kwargs": {"enable_thinking": not args.disable_thinking},
        }
        endpoint = args.vllm_url.rstrip("/") + "/chat/completions"
        last_error = None
        for retry in range(5):
            request = urllib.request.Request(
                endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=args.vllm_timeout_seconds) as response:
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
                        "request": payload,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                last_error = RuntimeError(f"vLLM HTTP {error.code}: {body[:1200]}")
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
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append({"name": function.get("name"), "arguments": arguments})
        content = message.get("content") or message.get("reasoning_content") or ""
        # This is an evaluator trace, not a teacher trajectory.  Keep the
        # ordinary assistant text under a neutral field name.
        raw = (json.dumps({"assistant_content": content, "tool_calls": calls}, ensure_ascii=False)
               if calls else content)
        usage = result.get("usage") or {}
        generated = int(usage.get("completion_tokens") or 0)
        return raw, choice.get("finish_reason") == "length", generated, None

'''
    if text.count(anchor) != 1:
        raise RuntimeError("evaluator generation anchor changed")
    text = text.replace(anchor, helper + anchor)
    anchor = "        nonlocal id_texts_cache\n"
    addition = anchor + "        if args.vllm_url:\n            return generate_vllm(messages, tools, sampled)\n"
    if text.count(anchor) != 1:
        raise RuntimeError("evaluator generation dispatch anchor changed")
    text = text.replace(anchor, addition)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def balanced_shards(rows: list[dict], worker_count: int) -> list[list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["score_key"])].append(row)
    shards: list[list[dict]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for _, score_rows in sorted(grouped.items(), key=lambda item: -len(item[1])):
        target = min(range(worker_count), key=lambda index: loads[index])
        shards[target].extend(score_rows)
        loads[target] += len(score_rows)
    for shard in shards:
        shard.sort(key=lambda row: (str(row["score_key"]), int(
            (row.get("runtime_item") or {}).get("input", {}).get("event_range", {}).get("start", 0)
        )))
    return shards


def wait_healthy(base_url: str, process: subprocess.Popen, server_log: Path) -> None:
    deadline = time.monotonic() + 900
    health_url = base_url.removesuffix("/v1").rstrip("/") + "/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited ({process.returncode}); inspect {server_log}")
        try:
            with urllib.request.urlopen(health_url, timeout=3):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(3)
    raise TimeoutError(f"vLLM did not become healthy; inspect {server_log}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter",
        help="optional LoRA adapter; omit to evaluate the unadapted base model",
    )
    parser.add_argument("--vllm-env", default="guqin-vllm-019")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument(
        "--workflow", choices=("two_stage", "single_stage"), default="two_stage",
        help=("agent workflow: two_stage is Base→Guqinizer; single_stage uses the "
              "same direct-final public prompt and tools as single-stage SFT data"),
    )
    parser.add_argument(
        "--single-stage", dest="workflow", action="store_const", const="single_stage",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--port", type=int, default=8217)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.84)
    args = parser.parse_args()
    if args.workers < args.max_num_seqs:
        raise SystemExit("workers must be >= max-num-seqs so there are waiting clients")

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("evaluation input is empty")
    output_dir = args.output_dir.resolve()
    workers_dir = output_dir / "workers"
    workers_dir.mkdir(parents=True, exist_ok=True)
    shards = balanced_shards(rows, args.workers)
    active = [(index, shard) for index, shard in enumerate(shards) if shard]
    print(json.dumps({
        "event": "vllm_eval_start", "phrases": len(rows),
        "scores": len({str(row["score_key"]) for row in rows}),
        "workers": len(active), "max_num_seqs": args.max_num_seqs,
        "workflow": args.workflow,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }, ensure_ascii=False), flush=True)

    input_paths, output_paths, log_paths = {}, {}, {}
    for index, shard in active:
        input_path = workers_dir / f"input_{index:02d}.jsonl"
        prediction_path = workers_dir / f"predictions_{index:02d}.jsonl"
        log_path = workers_dir / f"worker_{index:02d}.log"
        input_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard),
            encoding="utf-8",
        )
        input_paths[index], output_paths[index], log_paths[index] = input_path, prediction_path, log_path

    # Keep the generated evaluator beside the production script so its ROOT
    # calculation and package imports resolve exactly as in normal runs.
    evaluator_copy = Path(__file__).with_name("eval_two_stage_score_vllm_runtime.py")
    patch_evaluator(Path(__file__).with_name("eval_two_stage_score.py"), evaluator_copy)
    conda = Path.home() / "miniconda3" / "bin" / "conda"
    server_log = output_dir / "vllm_server.log"
    served_model_name = "guqin-sft" if args.adapter else "qwen35-base"
    command = [
        str(conda), "run", "--no-capture-output", "-n", args.vllm_env,
        "vllm", "serve", args.base_model,
        "--served-model-name", served_model_name,
        "--port", str(args.port), "--dtype", "bfloat16",
        "--quantization", "bitsandbytes",
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
    ]
    if args.adapter:
        command.extend([
            "--enable-lora", "--lora-modules", f"guqin-sft={args.adapter}",
            "--max-lora-rank", "8", "--max-loras", "1",
        ])
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    server_file = server_log.open("w", encoding="utf-8")
    server = subprocess.Popen(
        command, stdout=server_file, stderr=subprocess.STDOUT, env=env,
        start_new_session=True,
    )
    base_url = f"http://127.0.0.1:{args.port}/v1"
    clients: dict[int, subprocess.Popen] = {}
    log_files = {}
    try:
        wait_healthy(base_url, server, server_log)
        with urllib.request.urlopen(base_url + "/models", timeout=10) as response:
            models = json.loads(response.read().decode("utf-8"))
        model_ids = {entry.get("id") for entry in models.get("data", [])}
        if served_model_name not in model_ids:
            raise RuntimeError(
                f"served model {served_model_name!r} missing: {sorted(model_ids)}"
            )
        print(json.dumps({
            "event": "vllm_ready", "models": sorted(model_ids),
            "client_workers": len(active), "server_log": str(server_log),
        }, ensure_ascii=False), flush=True)

        for index, shard in active:
            cmd = [
                sys.executable, str(evaluator_copy),
                "--input", str(input_paths[index]), "--output", str(output_paths[index]),
                "--base-model", args.base_model, "--vllm-url", base_url,
                "--vllm-model", served_model_name,
                "--vllm-max-model-len", str(args.max_model_len),
                "--max-new-tokens", str(args.max_new_tokens),
                "--max-rounds", str(args.max_rounds), "--attempts", str(args.attempts),
                "--resume",
            ]
            cmd.extend(["--workflow", args.workflow])
            log_file = log_paths[index].open("w", encoding="utf-8")
            log_files[index] = log_file
            clients[index] = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
            )
            print(json.dumps({
                "event": "worker_started", "worker": index,
                "scores": len({str(row["score_key"]) for row in shard}),
                "phrases": len(shard), "log": str(log_paths[index]),
            }, ensure_ascii=False), flush=True)

        failures = {}
        while clients:
            for index, process in list(clients.items()):
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    failures[index] = code
                del clients[index]
            if clients:
                time.sleep(5)
        if failures:
            raise RuntimeError(f"worker failures {failures}; inspect {workers_dir}")

        # A resumed evaluator appends a replacement record when an earlier
        # invalid attempt is retried successfully.  Keep the latest record
        # for every sample here, rather than treating that normal retry trace
        # as a queue failure.
        records_by_id: dict[str, dict] = {}
        duplicate_records_discarded = 0
        for index, _ in active:
            path = output_paths[index]
            if not path.exists():
                raise RuntimeError(f"worker output missing: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                if sample_id in records_by_id:
                    duplicate_records_discarded += 1
                records_by_id[sample_id] = record
        expected = {str(row["sample_id"]) for row in rows}
        if len(expected) != len(rows):
            raise RuntimeError("evaluation input contains duplicate sample_id values")
        unexpected = sorted(set(records_by_id) - expected)
        if unexpected:
            raise RuntimeError(
                f"worker output contains {len(unexpected)} sample_ids absent from input: "
                f"{unexpected[:10]}"
            )
        skipped = [str(row["sample_id"]) for row in rows
                   if str(row["sample_id"]) not in records_by_id]
        order = {str(row["sample_id"]): index for index, row in enumerate(rows)}
        records = sorted(records_by_id.values(), key=lambda record: order[str(record["sample_id"])])
        result_path = output_dir / "predictions.jsonl"
        result_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        valid = sum(bool(record.get("protocol_valid")) for record in records)
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(json.dumps({
            "expected": len(rows),
            "completed": len(records),
            "protocol_valid": valid,
            "skipped_after_invalid_predecessor": len(skipped),
            "skipped_sample_ids": skipped,
            "duplicate_records_discarded": duplicate_records_discarded,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "event": "vllm_eval_complete", "phrases": len(records),
            "protocol_valid": valid, "skipped": len(skipped),
            "duplicate_records_discarded": duplicate_records_discarded,
            "predictions": str(result_path), "summary": str(summary_path),
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        for process in clients.values():
            if process.poll() is None:
                process.terminate()
        for process in clients.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        for log_file in log_files.values():
            log_file.close()
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
        server_file.close()
        evaluator_copy.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
