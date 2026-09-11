#!/usr/bin/env python3
"""Teacher-generated score-level tuning decisions with real tool execution.

Replaces the deterministic `sft_tuning_decision_train.jsonl` export as the
training-data source for the tuning task, so that every training trajectory
comes from the same audited teacher pipeline. The public prompt contains only
melody features (tonic, meter, degree histogram, range, sections); the
annotated tuning of the score stays teacher-private and is recorded solely in
the audit file as demonstration supervision.

Tools (really executed by the runner):
- get_tuning_catalog: known tunings with their seven open-string MIDI values;
- check_scale_resources: for a candidate open_midi, report 散音／泛音 coverage
  of the tonic, the dominant and the most frequent scale degrees;
- submit_tuning: audited submission; passing the audit commits the decision
  and terminates the sample (termination.kind = accepted_submit_tuning).

Hard audit on submit_tuning (derived from observed_tuning_registry.json):
seven strictly increasing values inside the observed range with adjacent gaps
inside the observed envelope. Matching the annotated tuning is NOT required.

Run --self-test for an offline end-to-end check with a scripted fake teacher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from agents.abc_to_jianzipu.inverse_baseline import AUDIT  # noqa: E402
from agents.abc_to_jianzipu.pitch_candidates import _pitch  # noqa: E402
from generate_teacher_tool_trajectories import (  # noqa: E402
    extract_decision_summary, parse_teacher_envelope, text_blocks,
)

REGISTRY_PATH = ROOT / "ABC_J/agent_training/exported/observed_tuning_registry.json"
DEFAULT_INPUT = ROOT / "ABC_J/agent_training/exported/sft_tuning_decision_train.jsonl"
DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
STANDARD_OPEN_MIDI = [48.0, 50.0, 53.0, 55.0, 57.0, 60.0, 62.0]
PITCH_CLASS_NAMES = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}

TOOLS = [
    {"name": "get_tuning_catalog", "description": "列出已知古琴调弦及其七弦散音音高。",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "check_scale_resources", "description": "核验候选调弦下主音、属音和高频音级的绝对音名、曲内 MIDI、散音／泛音资源、加权覆盖率及相对正调的改弦距离。",
     "input_schema": {"type": "object", "properties": {
         "open_midi": {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7}},
         "required": ["open_midi"]}},
    {"name": "submit_tuning", "description": "提交目录中的已知调弦；系统按 open_midi 规范化名称，通过审计即提交并结束。",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"},
         "open_midi": {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7}},
         "required": ["open_midi"]}},
]

PUBLIC_SYSTEM = """你是 ABC→古琴减字谱系统中的曲级调弦规划师。输入是一首乐曲的旋律特征。
正调是默认路线：先评估主音、属音和高频重要音在正调下的散音／泛音资源，再结合散音需求判断
是否值得改弦；只有替代调弦带来明确资源收益时才离开正调，不得从调号直接映射调弦。
常见候选调弦（按一至七弦列出实际 open_midi）：
正调 48/50/53/55/57/60/62；慢三弦 48/50/51/55/57/60/62；慢二 48/48/53/55/57/60/62；
慢一三六 46/50/51/55/57/58/62；慢一三四六 46/50/51/53/57/58/62；
紧五弦 48/50/53/55/59/60/62；紧五慢一 46/50/53/55/59/60/62；
紧二五 48/52/53/55/59/60/62；紧二五七 48/52/53/55/59/60/64。
这些只是优先核验的常见候选，不是调号到调弦的固定映射；同名调弦若有音高变体，仍以实际 open_midi 为准。
提交前应比较候选的加权散音／泛音覆盖率与相对正调的改弦距离；低频音级的一处新增命中不足以支持大幅改弦。
最终只能提交 get_tuning_catalog 中的已知调弦，不得自行拼装调弦或自由命名；名称由系统按 open_midi 规范化。
输出必须给出七弦实际 open_midi 音高；同名调弦可能有音高变体，不能只给名称。
在每次工具调用前用一句简短、可由公开输入核查的决策摘要说明理由。"""


def registry_envelope() -> dict:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    tunings = registry["tunings"]
    values = [value for tuning in tunings for value in tuning["open_midi"]]
    gaps = [round(b - a, 6) for tuning in tunings
            for a, b in zip(tuning["open_midi"], tuning["open_midi"][1:])]
    return {
        "tunings": tunings,
        "value_low": min(values) - 2.0,
        "value_high": max(values) + 2.0,
        "gap_low": min(gaps),
        "gap_high": max(gaps) + 0.5,
    }


def audit_tuning(open_midi: list, envelope: dict) -> list[dict]:
    problems = []
    if len(open_midi) != 7:
        return [{"code": "open_midi_must_have_7_values"}]
    for position, value in enumerate(open_midi, 1):
        if not isinstance(value, (int, float)):
            problems.append({"string": position, "code": "value_not_number"})
        elif not envelope["value_low"] <= float(value) <= envelope["value_high"]:
            problems.append({"string": position, "code": "value_out_of_range",
                             "value": value})
    for position, (a, b) in enumerate(zip(open_midi, open_midi[1:]), 1):
        gap = round(float(b) - float(a), 6)
        if gap < 0:
            problems.append({"strings": [position, position + 1], "code": "descending"})
        elif not envelope["gap_low"] <= gap <= envelope["gap_high"]:
            problems.append({"strings": [position, position + 1], "code": "gap_out_of_envelope",
                             "gap": gap})
    return problems


def degree_midi(tonic_text: str, degree: int, octave_shift: int = 0) -> float | None:
    base = AUDIT.parse_tonic_midi({"tonic": tonic_text})
    if base is None or degree not in DEGREE_SEMITONES:
        return None
    return float(base) + DEGREE_SEMITONES[degree] + 12 * octave_shift


def render_catalog(envelope: dict) -> str:
    lines = ["名称｜七弦散音 MIDI", "正调｜48.0 50.0 53.0 55.0 57.0 60.0 62.0（默认路线）"]
    for tuning in envelope["tunings"]:
        if tuning["name"] == "正调":
            continue
        values = " ".join(f"{float(value):g}" for value in tuning["open_midi"])
        lines.append(f'{tuning["name"]}｜{values}')
    lines.append(f'合法音高区间｜{envelope["value_low"]:g} ~ {envelope["value_high"]:g}｜'
                 f'相邻弦间距｜{envelope["gap_low"]:g} ~ {envelope["gap_high"]:g}')
    return "\n".join(lines)


def scale_resources(record: dict, open_midi: list) -> dict:
    features = record["input"]["melody_features"]
    histogram = features.get("degree_histogram") or {}
    normalized_histogram = {}
    for raw_degree, raw_count in histogram.items():
        try:
            degree, count = int(raw_degree), int(raw_count)
        except (TypeError, ValueError):
            continue
        if degree in DEGREE_SEMITONES:
            normalized_histogram[degree] = count
    if not normalized_histogram:
        raise ValueError("degree_histogram_empty_or_invalid")
    ranked = sorted(normalized_histogram.items(), key=lambda item: -item[1])
    # The contract promises tonic/dominant plus frequent degrees. Keep 1 and 5
    # even when one of them falls outside the six most frequent entries.
    focus_degrees = list(dict.fromkeys(
        [degree for degree in (1, 5) if degree in normalized_histogram]
        + [degree for degree, _count in ranked]
    ))[:6]
    focus = [(degree, normalized_histogram[degree]) for degree in focus_degrees]
    pitch_low = int(features["pitch_min_midi"])
    pitch_high = int(features["pitch_max_midi"])
    lines = ["音级｜绝对音名｜曲内目标 MIDI｜出现次数｜散音命中｜泛音命中（弦·徽）"]
    resources = []
    for degree, count in focus:
        label = {1: "主音", 5: "属音"}.get(degree, "")
        open_hits, harmonic_hits = [], []
        base_target = degree_midi(features["tonic"], degree)
        if base_target is None:
            continue
        pitch_class = int(round(base_target)) % 12
        targets = [midi for midi in range(pitch_low, pitch_high + 1)
                   if midi % 12 == pitch_class]
        for target in targets:
            for string, opened in enumerate(open_midi, 1):
                if abs(_pitch(float(opened), None, "open") - target) < 0.5:
                    open_hits.append(f"{string}弦")
                for hui in range(1, 14):
                    if abs(_pitch(float(opened), float(hui), "harmonic") - target) < 0.5:
                        harmonic_hits.append(f"{string}弦·{hui}徽")
        open_hits = list(dict.fromkeys(open_hits))
        harmonic_hits = list(dict.fromkeys(harmonic_hits))
        note_name = PITCH_CLASS_NAMES[pitch_class]
        resources.append({
            "degree": degree, "role": label or None, "note_name": note_name,
            "pitch_class": pitch_class, "target_midi_in_score_range": targets,
            "count": count, "open_hits": open_hits, "harmonic_hits": harmonic_hits,
        })
        lines.append(f"{degree}（{label}）｜{note_name}｜{'/'.join(map(str, targets)) or '无'}｜"
                     f"{count}｜{','.join(open_hits) or '无'}｜"
                     f"{'、'.join(harmonic_hits[:4]) or '无'}")
    total = sum(item["count"] for item in resources) or 1
    open_weight = sum(item["count"] for item in resources if item["open_hits"])
    harmonic_weight = sum(item["count"] for item in resources if item["harmonic_hits"])
    retune_distance = sum(abs(a - b) for a, b in zip(open_midi, STANDARD_OPEN_MIDI))
    return {
        "format": "guqin-scale-resources-1.1",
        "text": "\n".join(lines),
        "resources": resources,
        "summary": {
            "focused_note_count": total,
            "open_coverage_ratio": round(open_weight / total, 6),
            "harmonic_coverage_ratio": round(harmonic_weight / total, 6),
            "retune_semitones_vs_standard": round(retune_distance, 6),
        },
    }


def render_scale_resources(record: dict, open_midi: list) -> str:
    """Backward-compatible readable rendering used by offline checks."""
    return scale_resources(record, open_midi)["text"]


def catalog_matches(open_midi: list, envelope: dict) -> list[str]:
    matches = []
    for tuning in envelope["tunings"]:
        if all(abs(float(a) - float(b)) < 0.01
               for a, b in zip(open_midi, tuning["open_midi"])):
            matches.append(tuning["name"])
    return matches


def render_user_prompt(record: dict) -> str:
    features = record["input"]["melody_features"]
    histogram = features.get("degree_histogram") or {}
    degrees = "、".join(f"{degree}级×{count}" for degree, count
                        in sorted(histogram.items(), key=lambda item: -int(item[1])))
    sections = features.get("sections") or []
    return "\n".join([
        f"乐曲编号｜{record['score_key']}",
        f"调号主音｜{features.get('tonic')}｜拍号｜{features.get('meter')}",
        f"发声音符数｜{features.get('sounding_notes')}｜"
        f"音域｜MIDI {features.get('pitch_min_midi')}~{features.get('pitch_max_midi')}",
        f"音级分布｜{degrees}",
        f"分段数｜{len(sections)}",
        "任务｜规划七弦调弦：先查资源，再提交含七弦 open_midi 的最终决策。",
    ])


def invoke_tool(name: str, args: dict, record: dict, envelope: dict) -> tuple[dict, dict]:
    """Execute a tuning tool; returns (public_result, audit_result)."""
    if name == "get_tuning_catalog":
        result = {"format": "guqin-tuning-catalog-1.0", "text": render_catalog(envelope)}
        return result, result
    if name == "check_scale_resources":
        open_midi = [float(value) for value in args["open_midi"]]
        problems = audit_tuning(open_midi, envelope)
        if problems:
            return {"ok": False, "errors": problems}, {"ok": False, "errors": problems}
        result = scale_resources(record, open_midi)
        return result, result
    if name == "submit_tuning":
        open_midi = [float(value) for value in args["open_midi"]]
        problems = audit_tuning(open_midi, envelope)
        matches = catalog_matches(open_midi, envelope) if not problems else []
        if not problems and not matches:
            problems.append({"code": "unsupported_custom_tuning",
                             "message": "请从 get_tuning_catalog 返回的已知调弦中选择"})
        if problems:
            result = {"ok": False, "valid": False, "errors": problems}
        else:
            result = {"ok": True, "valid": True,
                      "submitted": {"name": matches[0], "open_midi": open_midi},
                      "requested_name": args.get("name")}
        audit = dict(result)
        audit["open_midi"] = open_midi
        return result, audit
    raise ValueError(f"unknown tool: {name}")


def generate_one(client, model: str, record: dict, envelope: dict,
                 max_rounds: int) -> tuple[dict, dict]:
    user_prompt = render_user_prompt(record)
    annotation = record["target"]
    private_instruction = {
        "role": PUBLIC_SYSTEM,
        "teacher_only_goal": {
            "annotated_tuning": annotation,
            "note": "标注调弦是示范／偏好监督，不是唯一正确标签；只需给出合法且音乐上合理的决策。",
        },
        "tools": TOOLS,
        "output_contract": {
            "tool_turn": {"decision_summary": "可选；一句基于公开输入或工具结果的理由",
                          "tool_calls": [{"name": "工具名", "arguments": {}}]},
            "rules": [
                "每轮只输出一个 JSON 对象，不要输出 Markdown 或 JSON 之外的文字。",
                "某次 submit_tuning 通过审计时，系统直接提交并结束。",
                "不得编造工具名或参数。",
            ],
            "forbidden": "decision_summary 不得提及 reference、teacher、标注答案或私有答案。",
        },
    }
    api_messages = [{"role": "user", "content": user_prompt}]
    public_messages = [{"role": "system", "content": PUBLIC_SYSTEM},
                       {"role": "user", "content": user_prompt}]
    tool_log: list[dict] = []
    teacher_io_trace: list[dict] = []
    accepted = None
    for round_number in range(1, max_rounds + 1):
        options = {"model": model, "max_tokens": 1200, "temperature": 0.2,
                   "system": json.dumps(private_instruction, ensure_ascii=False),
                   "messages": api_messages}
        if model == "MiniMax-M3":
            options["thinking"] = {"type": "disabled"}
        response = client.messages.create(**options)
        teacher_io_trace.append({"round": round_number, "request": options,
                                 "response": response.model_dump(exclude_none=True)})
        raw_text = text_blocks(response)
        envelope_payload = parse_teacher_envelope(raw_text)
        summary = envelope_payload["decision_summary"]
        api_messages.append({"role": "assistant", "content": raw_text})
        public_calls, public_results, executed = [], [], []
        for call_number, call in enumerate(envelope_payload["tool_calls"], 1):
            call_id = f"call_{round_number:02d}_{call_number:02d}"
            result, audit_result = invoke_tool(call["name"], dict(call["arguments"]),
                                               record, envelope)
            tool_log.append({"name": call["name"], "arguments": call["arguments"],
                             "result": audit_result})
            executed.append({"name": call["name"], "arguments": call["arguments"],
                             "result": result})
            public_calls.append({"id": call_id, "type": "function", "function": {
                "name": call["name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False)}})
            public_results.append({"role": "tool", "tool_call_id": call_id,
                                   "name": call["name"],
                                   "content": json.dumps(result, ensure_ascii=False)})
            if call["name"] == "submit_tuning" and result.get("ok"):
                accepted = result["submitted"]
        public_messages.append({"role": "assistant", "content": summary,
                                "tool_calls": public_calls})
        public_messages.extend(public_results)
        if accepted is not None:
            break
        api_messages.append({"role": "user", "content": json.dumps({
            "tool_results": executed,
            "instruction": "根据这些真实工具结果继续；需要工具时输出 tool_calls。",
        }, ensure_ascii=False)})
    if accepted is None:
        raise ValueError("teacher exceeded tool-round limit without accepted submit_tuning")
    annotation_values = annotation.get("open_midi") or []
    private = {"sample_id": f"{record['score_key']}-tuning-teacher-tools", "teacher_private": {
        "annotated_tuning": annotation,
        "accepted_tuning": accepted,
        "per_string_cents_vs_annotation": [
            round((float(a) - float(b)) * 100, 2) if len(annotation_values) == 7 else None
            for a, b in zip(accepted["open_midi"], annotation_values)],
        "tool_execution_log": tool_log,
        "teacher_io_trace": teacher_io_trace,
        "teacher_model": model,
    }}
    public = {"schema_version": "tuning-agent-messages-1.0",
              "sample_id": private["sample_id"], "task": "select_guqin_tuning",
              "tools": TOOLS, "messages": public_messages,
              "provenance": {"score_key": record["score_key"],
                             "score_family_id": record.get("score_family_id"),
                             "split": record.get("split")},
              "termination": {"kind": "accepted_submit_tuning"},
              "verification": {"tools_really_executed": True,
                               "tuning_audit_passed": True,
                               "private_leakage_passed": True}}
    serialized = json.dumps(public, ensure_ascii=False)
    for key in ("teacher_private", "annotated_tuning", "target_patches"):
        if f'"{key}"' in serialized:
            raise ValueError(f"private key {key} leaked into public record")
    return public, private


class _FakeBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeBlock(text)]

    def model_dump(self, **_):
        return {"content": [{"type": "text", "text": block.text} for block in self.content]}


class _FakeMessages:
    def __init__(self, envelopes: list[str]):
        self._envelopes = envelopes

    def create(self, **_options):
        return _FakeResponse(self._envelopes.pop(0))


class FakeClient:
    """Scripted offline teacher for --self-test."""

    def __init__(self, envelopes: list[str]):
        self.messages = _FakeMessages(envelopes)


def self_test() -> None:
    records = [json.loads(line) for line in DEFAULT_INPUT.open(encoding="utf-8")]
    record = next(item for item in records if item["score_key"] == "S0tu8BXC")
    envelope = registry_envelope()
    fake = FakeClient([
        json.dumps({"decision_summary": "先查看已知调弦目录。", "tool_calls": [
            {"name": "get_tuning_catalog", "arguments": {}}]}, ensure_ascii=False),
        json.dumps({"decision_summary": "核验正调下的主音属音散泛资源。", "tool_calls": [
            {"name": "check_scale_resources",
             "arguments": {"open_midi": [48.0, 50.0, 53.0, 55.0, 57.0, 60.0, 62.0]}}]}, ensure_ascii=False),
        json.dumps({"decision_summary": "正调资源已覆盖主音与属音，无需改弦，提交。", "tool_calls": [
            {"name": "submit_tuning",
             "arguments": {"name": "正调",
                           "open_midi": [48.0, 50.0, 53.0, 55.0, 57.0, 60.0, 62.0]}}]}, ensure_ascii=False),
    ])
    public, private = generate_one(fake, "self-test-model", record, envelope, max_rounds=6)
    assert public["termination"]["kind"] == "accepted_submit_tuning"
    assert len(public["messages"]) == 2 + 2 * 3  # system+user, then 3 assistant+tool pairs
    assert private["teacher_private"]["tool_execution_log"][0]["result"]["text"].startswith(
        "名称｜")
    resources = invoke_tool(
        "check_scale_resources",
        {"open_midi": [48.0, 50.0, 53.0, 55.0, 57.0, 60.0, 62.0]},
        record,
        envelope,
    )[0]["text"]
    assert len(resources.splitlines()) > 1, "scale resource table must contain data rows"
    assert "1（主音）" in resources, "scale resource table must contain the tonic"
    assert "5（属音）" in resources, "scale resource table must contain the dominant"
    structured = scale_resources(record, STANDARD_OPEN_MIDI)
    assert structured["format"] == "guqin-scale-resources-1.1"
    assert structured["resources"] and structured["resources"][0]["note_name"]
    assert "target_midi_in_score_range" in structured["resources"][0]
    slow_two = invoke_tool(
        "submit_tuning", {"name": "慢二", "open_midi": [48, 48, 53, 55, 57, 60, 62]},
        record, envelope,
    )[0]
    assert slow_two.get("ok") is True and slow_two["submitted"]["name"] == "慢二"
    custom = invoke_tool(
        "submit_tuning", {"name": "自定", "open_midi": [46, 50, 53, 58, 59, 60, 63]},
        record, envelope,
    )[0]
    assert custom.get("ok") is False
    assert custom["errors"][0]["code"] == "unsupported_custom_tuning"
    bad = invoke_tool("submit_tuning", {"open_midi": [62.0, 60.0] * 3 + [48.0]}, record, envelope)[0]
    assert bad.get("ok") is False and bad["errors"], "descending tuning must fail audit"
    print(json.dumps({"self_test": "ok",
                      "sample_id": public["sample_id"],
                      "messages": len(public["messages"])}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "ABC_J/agent_training/tuning_teacher")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--score-key", action="append")
    parser.add_argument("--model")
    parser.add_argument("--max-tool-rounds", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    load_dotenv(ROOT / ".env")
    from anthropic import Anthropic
    model = args.model or os.getenv("MINIMAX_MODEL", "MiniMax-M3")
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                       base_url=os.environ["ANTHROPIC_BASE_URL"].rstrip("/"),
                       timeout=120.0, max_retries=1)

    records = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    if args.score_key:
        wanted = set(args.score_key)
        records = [record for record in records if record["score_key"] in wanted]
    selected = records[:args.limit]
    envelope = registry_envelope()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted, failures = 0, []
    with (args.output_dir / "tuning_messages_train.jsonl").open(
            "w", encoding="utf-8", newline="\n") as pub, \
        (args.output_dir / "tuning_teacher_audit.jsonl").open(
            "w", encoding="utf-8", newline="\n") as priv:
        for record in selected:
            last_error = None
            for _attempt in range(1, args.max_attempts + 1):
                try:
                    public, private = generate_one(client, model, record, envelope,
                                                   args.max_tool_rounds)
                    pub.write(json.dumps(public, ensure_ascii=False) + "\n")
                    priv.write(json.dumps(private, ensure_ascii=False) + "\n")
                    pub.flush()
                    priv.flush()
                    accepted += 1
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                failures.append({"score_key": record["score_key"],
                                 "attempts": args.max_attempts,
                                 "error": f"{type(last_error).__name__}: {last_error}"})
    report = {"schema_version": "tuning-teacher-report-1.0", "model": model,
              "source_scores": len(selected), "accepted": accepted,
              "rejected": len(failures), "failures": failures}
    (args.output_dir / "tuning_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
