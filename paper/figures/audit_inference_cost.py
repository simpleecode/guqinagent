"""Estimate API-equivalent cost from retained final prediction trajectories.

CPU-only, no inference/API calls. Requires tokenizers and jinja2. Tokenizer
assets live in tmp/cost-tokenizers/{qwen,glm}. Prices are public provider list
prices, not self-hosting expenses or a historical invoice. Missing server-side
cache usage is NOT treated as a measured zero: an explicit no-discount pricing
scenario is applied. Failed/discarded runs outside retained traces are excluded.
"""
import hashlib
import json
import argparse
from pathlib import Path

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXPERIMENTS = [
    ("traditional", "handbook", "react_handbook_full_test50", "glm"),
    ("traditional", "fixed_workflow", "fixed_reference_ablation_current48", "qwen"),
    ("traditional", "single_stage", "single_stage_current48", "qwen"),
    ("traditional", "two_stage_off", "warning_ablation_current48_off", "qwen"),
    ("traditional", "two_stage", "warning_ablation_current48_on", "qwen"),
    ("modern", "handbook", "react_handbook_modern", "glm"),
    ("modern", "fixed_workflow", "modern_fixed_workflow", "qwen"),
    ("modern", "single_stage", "modern_single_stage", "qwen"),
    ("modern", "two_stage_off", "modern_two_stage_off", "qwen"),
    ("modern", "two_stage", "modern_two_stage", "qwen"),
]
PRICES = {
    "qwen": {"model": "Qwen3.5-9B", "input_per_million_usd": .17,
             "output_per_million_usd": .25, "cached_input_per_million_usd": None,
             "source": "https://www.together.ai/pricing"},
    "glm": {"model": "GLM-5.3", "input_per_million_usd": 1.4,
            "output_per_million_usd": 4.4, "cached_input_per_million_usd": .26,
            "source": "https://www.together.ai/pricing",
            "manufacturer_crosscheck": "https://docs.z.ai/guides/overview/pricing"},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Counter:
    def __init__(self, family):
        folder = ROOT / "tmp/cost-tokenizers" / family
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        config_path = folder / "tokenizer_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        template_path = folder / "chat_template.jinja"
        template = template_path.read_text() if template_path.exists() else config["chat_template"]
        if isinstance(template, list):
            template = next(item["template"] for item in template if item["name"] == "default")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                           extensions=["jinja2.ext.loopcontrols"])
        def tojson(value, **kwargs):
            kwargs.setdefault("ensure_ascii", False)
            return json.dumps(value, **kwargs)
        env.filters["tojson"] = tojson
        def fail(message):
            raise ValueError(message)
        env.globals["raise_exception"] = fail
        self.template = env.from_string(template)
        self.config = config
        self.assets = {str(p.relative_to(ROOT)): sha(p) for p in folder.iterdir() if p.is_file()}

    def render(self, messages, tools, generation=True):
        return self.template.render(messages=messages, tools=tools,
            add_generation_prompt=generation, enable_thinking=False,
            bos_token=self.config.get("bos_token") or "",
            eos_token=self.config.get("eos_token") or "<|im_end|>")

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def assistant(turn, family):
    content = turn.get("raw_output") or ""
    if family == "qwen":
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and "assistant_content" in payload:
                content = payload["assistant_content"]
        except (ValueError, TypeError):
            pass
    return {"role": "assistant", "content": content,
            "tool_calls": [{"type": "function", "function": {
                "name": call["name"], "arguments": call.get("arguments") or {}}}
                for call in turn.get("tool_calls") or []]}


def tools_for_glm():
    # Read existing tool definitions, never invoke or re-execute the runtime.
    import sys
    sys.path.insert(0, str(ROOT))
    from ABC_J.scripts.generate_teacher_tool_trajectories import public_tools_for, public_system_for
    tools = [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""),
        "parameters": t.get("input_schema") or {"type": "object"}}}
        for t in public_tools_for("single_stage", basic=False)]
    # Same concatenation as handbook.compose_system, without importing GEPA
    # (this is an offline token counter, not another rollout executor).
    handbook = (ROOT / "baseline/gepa_guqin/knowledge/complex_fingering_handbook.md").read_text().strip()
    return public_system_for("single_stage", basic=False).rstrip() + "\n\n" + handbook, tools


def no_tools_followup(warning):
    error = ("上一轮出现音高警告；请至少提交一次 edit_plan 尝试后再决定是否结束。请依据上一轮工具返回核对相关音序。"
             if warning else "当前段仍有待填写的演奏音，不能直接结束；请提交需要修改的 jianzi_rows。")
    return {"role": "user", "content": json.dumps({"tool_results": [{"ok": False, "error": error}],
            "instruction": "根据真实反馈继续；需要工具时继续调用工具。"}, ensure_ascii=False)}


def audit_error_coverage(report):
    """Do not confuse preflight rejections with paid but unrecorded outputs."""
    for entry in report["arms"].values():
        preflight = response_missing = 0
        path = ROOT / entry["prediction_path"]
        assert sha(path) == entry["prediction_sha256"]
        for row in map(json.loads, path.open()):
            for stage in ("base", "guqinizer", "single_stage"):
                for turn in row.get(stage, {}).get("trace", []):
                    error = str(turn.get("generation_error") or "")
                    if not error:
                        continue
                    if "prompt already exceeds context" in error:
                        preflight += 1
                    else:
                        response_missing += 1
        assert preflight + response_missing == entry["generation_errors_without_usage"]
        entry["preflight_context_rejections"] = preflight
        entry["failed_calls_with_unavailable_response_usage"] = response_missing
    report["cost_completeness"] = (
        "Visible retained responses only; failed calls with unavailable response usage are excluded, "
        "not assigned a measured zero cost. Preflight context rejections made no model-generation request.")


def add_direct_costs(report, counter):
    """Apply the identical visible-turn estimator to saved, tool-free requests.

    Include invalid responses in workload; normalize by actual attempts, never
    treating skipped phrases as zero-cost calls or as scorable musical output.
    """
    for dataset, scope, experiment in [
        ('traditional', 'test48', 'direct_sft_current48'),
        ('modern', 'modern', 'modern_direct_sft'),
    ]:
        path = ROOT / 'train/eval_outputs/direct_sft' / scope / 'predictions.jsonl'
        original_sha = sha(path)
        config = json.loads((path.parent / 'direct_config.json').read_text())
        summary = json.loads((ROOT / 'evaluation_results' / experiment / 'summary.json').read_text())
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(predictions) == len({r['sample_id'] for r in predictions}) == summary['prediction_samples']
        assert config['tools'] is False and config['thinking'] is False
        inp = out = 0
        for row in predictions:
            direct = row['direct']
            assert direct['model_generations'] == 1 and direct['tool_calls'] == 0
            messages = direct['request_messages']
            prompt = counter.render(messages, None)
            response = {'role': 'assistant', 'content': direct['generation']['text']}
            completed = counter.render(messages + [response], None, generation=False)
            prefix = 0
            for left, right in zip(prompt, completed):
                if left != right:
                    break
                prefix += 1
            inp += counter.count(prompt)
            out += counter.count(completed[prefix:])
        prices = PRICES['qwen']
        cost = (inp * prices['input_per_million_usd'] + out * prices['output_per_million_usd']) / 1e6
        completion = summary['completion']
        report['arms'][dataset + '/direct_sft'] = {
            'experiment': experiment, 'samples': len(predictions),
            'expected_work_units': completion['expected'],
            'protocol_valid_work_units': completion['protocol_valid'],
            'retained_generation_rounds': len(predictions), 'generation_errors_without_usage': 0,
            'input_tokens_estimated': inp, 'output_tokens_estimated': out,
            'observed_cache_hit_tokens': None, 'scenario_cache_hit_tokens': 0,
            'estimated_api_equivalent_usd': cost, 'estimated_usd_per_work_unit': cost / len(predictions),
            'normalization': 'per attempted work unit, including invalid JSON; skipped suffixes are not model calls',
            'quality_scope': completion['musical_metric_scope'],
            'prediction_path': str(path.relative_to(ROOT)), 'prediction_sha256': original_sha,
        }
        assert sha(path) == original_sha
        print(json.dumps({'arm': dataset + '/direct_sft', 'usd_est': cost, 'attempts': len(predictions)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--direct-only', action='store_true', help='Extend the existing audit without recomputing other methods')
    args = parser.parse_args()
    if args.direct_only:
        report = json.loads((OUT / 'inference_cost.json').read_text())
        counter = Counter('qwen')
        assert report['prices'] == PRICES
        assert report['tokenizer_assets']['qwen'] == counter.assets
        add_direct_costs(report, counter)
        audit_error_coverage(report)
        (OUT / 'inference_cost.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        return
    counters = {family: Counter(family) for family in PRICES}
    glm_system, glm_tools = tools_for_glm()
    report = {"schema_version": "final-trace-api-cost-1", "pricing_checked_date": "2026-11-02",
              "currency": "USD", "prices": PRICES, "tokenizer_assets": {k: v.assets for k, v in counters.items()},
              "scope": "retokenized retained final trajectories; no discarded runs or training",
              "cache_scenario": "no cache discount; observed cache-hit usage unavailable",
              "usage_kind": "reconstructed token estimates, not provider billing usage",
              "glm_input_reconstruction": "saved user_prompt and tool receipts; current production-seed system plus fixed handbook and native model template",
              "omitted": "unrecorded hidden reasoning, failed calls absent from retained traces, provider-only framing and cache internals",
              "arms": {}}
    for dataset, method, experiment, family in EXPERIMENTS:
        config = json.loads((ROOT / "evaluation_results" / experiment / "config.json").read_text())
        path = Path(config["prediction_path"].replace("/Volumes/F/work/Gunqin-agent/", str(ROOT) + "/"))
        if not path.is_absolute():
            path = ROOT / path
        original_sha = sha(path)
        predictions = {r["sample_id"]: r for r in map(json.loads, path.open())}
        assert len(predictions) == (1031 if dataset == "traditional" else 91)
        counter = counters[family]
        inp = out = rounds = errors = 0
        for row in predictions.values():
            assert row.get("protocol_valid") is True
            for stage in ("base", "guqinizer", "single_stage"):
                session = row.get(stage)
                if not session:
                    continue
                history = [{"role": "system", "content": glm_system},
                           {"role": "user", "content": session.get("user_prompt", "")}]
                for turn in session.get("trace") or []:
                    if turn.get("generation_error"):
                        errors += 1
                        if family == "glm":
                            raise ValueError("Cannot safely reconstruct failed GLM calls")
                        continue
                    if family == "qwen":
                        messages = turn.get("request_messages")
                        tools = turn.get("request_tools")
                        if not messages or not tools:
                            raise ValueError(f"Missing frozen request: {experiment}/{row['sample_id']}")
                    else:
                        messages, tools = history, glm_tools
                    prompt = counter.render(messages, tools)
                    response = assistant(turn, family)
                    completed = counter.render(messages + [response], tools, generation=False)
                    # Templates can rewrite an assistant prefix when generation
                    # becomes a completed turn. Count the completed turn suffix
                    # after their common rendered prefix; never count raw wrapper JSON.
                    prefix = 0
                    for left, right in zip(prompt, completed):
                        if left != right:
                            break
                        prefix += 1
                    suffix = completed[prefix:]
                    if len(suffix) > 1000000:
                        raise ValueError("Unexpected template rewrite")
                    inp += counter.count(prompt)
                    out += counter.count(suffix)
                    rounds += 1
                    if family == "glm":
                        history.append(response)
                        for result in turn.get("tool_results") or []:
                            history.append({"role": "tool", "name": result["name"],
                                "content": json.dumps(result["result"], ensure_ascii=False)})
                        if not turn.get("tool_calls"):
                            warning = bool(turn.get("pitch_gate", {}).get("followup_edit_required"))
                            history.append(no_tools_followup(warning))
        prices = PRICES[family]
        cost = (inp * prices["input_per_million_usd"] + out * prices["output_per_million_usd"]) / 1e6
        entry = {"experiment": experiment, "samples": len(predictions), "retained_generation_rounds": rounds,
                 "generation_errors_without_usage": errors, "input_tokens_estimated": inp,
                 "output_tokens_estimated": out, "observed_cache_hit_tokens": None,
                 "scenario_cache_hit_tokens": 0, "estimated_api_equivalent_usd": cost,
                 "estimated_usd_per_work_unit": cost / len(predictions),
                 "prediction_path": str(path.relative_to(ROOT)), "prediction_sha256": original_sha}
        assert sha(path) == original_sha
        report["arms"][dataset + "/" + method] = entry
        print(json.dumps({"arm": dataset + "/" + method, "usd_est": cost, "rounds": rounds}), flush=True)
    add_direct_costs(report, counters['qwen'])
    audit_error_coverage(report)
    (OUT / "inference_cost.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    if "--audit-existing-errors" in sys.argv:
        path = OUT / "inference_cost.json"
        report = json.loads(path.read_text())
        audit_error_coverage(report)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    else:
        main()
