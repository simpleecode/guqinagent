#!/usr/bin/env python3
"""Rewrite teacher decision summaries into a private-source-free public form.

The original teacher I/O is never overwritten.  Assistant reasoning messages
are replaced in the public copy; tool arguments/results and all trajectory
fields remain byte-for-byte equivalent after JSON decoding.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# These are source markers, not a quality score.  A rewritten summary must be
# understandable without knowing that a private reference existed.
BANNED_PUBLIC_TERMS = (
    "gqs", "reference", "teacher", "target", "teacher-only", "private answer",
    "reference plan", "reference action", "标注", "答案", "教师提示", "系统提示",
    "提示词", "教师私有", "私有参考", "最终标注", "参考谱面", "参考谱", "参考建议",
    "参考中", "参考方向", "参考显示", "参考中的", "按参考", "根据参考", "与参考一致", "参考使用",
)
REDACTION_REPLACEMENTS = (
    ("参考谱面", "当前谱面"), ("参考谱", "当前谱面"), ("参考建议", "演奏判断"),
    ("参考答案", "演奏判断"), ("标注答案", "演奏判断"),
    ("与参考一致", "与前后音连贯"), ("根据参考", "根据音高与指法"),
    ("参考方向", "演奏判断"), ("参考显示", "谱面显示"),
    ("参考中的", "谱面中的"), ("参考中", "谱面中"), ("按参考", "按音高与指法"),
    ("参考使用", "采用"), ("教师提示", "演奏要求"),
    ("教师私有", "内部信息"), ("私有参考", "内部信息"),
    ("系统提示", "当前状态"),
    ("最终标注", "最终谱面"), ("参考", "谱面"), ("标注", "谱面"),
    ("答案", "判断依据"), ("GQS", "谱面"),
    ("reference", "谱面"), ("teacher-only", "演奏"),
    ("teacher", "演奏"), ("target", "目标音"), ("private answer", "内部判断"),
)
CN_DIGITS = str.maketrans("一二三四五六七八九", "123456789")


def factual_tokens(value: str) -> set[str]:
    """Return unit-bearing numeric facts; redaction must not invent new ones."""
    normalized = str(value).translate(CN_DIGITS).replace("十", "10")
    return set(re.findall(r"\d+(?:弦|徽|分|音分|MIDI)", normalized))


def _text_blocks(response) -> str:
    return "".join(
        str(getattr(block, "text", ""))
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    ).strip()


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("rewrite response contains no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("rewrite response is not an object")
    return value


def normalize_summary(value: object) -> str:
    summary = re.sub(r"\s+", " ", str(value or "")).strip()
    if not summary:
        raise ValueError("rewritten decision_summary is empty")
    lowered = summary.casefold()
    hit = next((term for term in BANNED_PUBLIC_TERMS if term.casefold() in lowered), None)
    if hit:
        raise ValueError(f"rewritten decision_summary contains private-source term: {hit}")
    return summary


def private_reference_texts(private: dict) -> set[str]:
    """Extract concrete reference strings for a second, text-level guard."""
    values: set[str] = set()
    annotation = str(private.get("teacher_private", {}).get("annotation_gqs") or "")
    for match in re.finditer(r"音｜\[\s*\d+\s*,\s*\"[^\"]*\"\s*,\s*(\"(?:\\.|[^\"\\])*\")", annotation):
        try:
            value = json.loads(match.group(1)).strip()
        except (TypeError, json.JSONDecodeError):
            continue
        if len(value) >= 3:
            values.add(value.casefold())
    return values


def validate_rewritten_summary(summary: str, private: dict, original_summary: str = "") -> str:
    summary = normalize_summary(summary)
    original_facts = factual_tokens(original_summary)
    introduced_facts = factual_tokens(summary) - original_facts
    if introduced_facts:
        raise ValueError(
            "rewritten decision_summary introduces unsupported numeric facts: "
            + ",".join(sorted(introduced_facts))
        )
    return summary


def sanitize_fallback_summary(summary: str, private: dict) -> str:
    """Last-resort lexical scrub after model retries, without adding facts."""
    value = str(summary or "")
    for source, replacement in sorted(REDACTION_REPLACEMENTS,
                                     key=lambda pair: len(pair[0]), reverse=True):
        value = re.sub(re.escape(source), replacement, value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def append_jsonl(path: Path, value: dict) -> None:
    """Persist one record immediately so an interrupted run can resume."""
    with path.open("a", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(value, ensure_ascii=False) + "\n")
        out.flush()


def read_jsonl_map(path: Path) -> dict[str, dict]:
    """Read already-written records, tolerating a truncated final line."""
    values: dict[str, dict] = {}
    if not path.exists():
        return values
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("sample_id"):
                values[str(value["sample_id"])] = value
    return values


def write_progress(path: Path, progress: dict) -> None:
    """Atomically update a small human-readable heartbeat/progress file."""
    progress = dict(progress)
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def make_client(model: str):
    from anthropic import Anthropic

    is_glm = model.casefold().startswith("glm")
    key_name = "GLM_API_KEY" if is_glm else "ANTHROPIC_API_KEY"
    url_name = "GLM_BASE_URL" if is_glm else "ANTHROPIC_BASE_URL"
    api_key, base_url = os.getenv(key_name), os.getenv(url_name)
    if not api_key or not base_url:
        raise RuntimeError(f"{key_name}/{url_name} are not configured")
    return Anthropic(api_key=api_key, base_url=base_url.rstrip("/"),
                     timeout=120.0, max_retries=1)


def rewrite_prompt(item: dict, summaries: list[dict], retry_note: str = "") -> str:
    payload = {
        "instruction": (
            "你是训练数据后处理器。把每条原始 decision_summary 重写为学生可见的公开 reasoning。"
            "保留每个修改音的序号、实际编辑内容，以及能由当前谱面、前文或工具结果支持的音乐学依据"
            "（音高、时值、前后音连接、指法可演奏性、走手方向等）。"
            "不得改变原摘要已有的弦、徽、音高、数值或减字事实，不得新增原摘要没有的具体数值；"
            "删除私有来源和隐藏答案痕迹：不得提及 GQS、标注、参考谱面、参考谱、参考方向、参考显示、教师提示、参考答案、"
            "私有参考或 target。"
            "如果原文以私有来源作为理由，请把它改写成直接的音乐学分析；不要新增输入中不存在的事实。"
            "下面每个请求只对应一条原始摘要，只返回一个 JSON 对象 {\"summary\":字符串}；不要按音符拆成多条。"
        ),
        "agent_stage": item.get("agent_stage"),
        "turn": summaries[0] if len(summaries) == 1 else summaries,
    }
    if retry_note:
        payload["retry_note"] = retry_note
    return json.dumps(payload, ensure_ascii=False)


def public_turns(item: dict) -> list[tuple[int, dict]]:
    turns = []
    messages = item.get("messages") or []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        following = messages[index + 1] if index + 1 < len(messages) else {}
        turns.append((index, {
            "original_summary": str(message.get("content") or ""),
            "tool_calls": deepcopy(message.get("tool_calls") or []),
            "tool_result": deepcopy(following.get("content"))
            if following.get("role") == "tool" else None,
        }))
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="split source rows into disjoint shards for parallel workers")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="worker index in 0..shard-count-1")
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard-count/shard-index")
    load_dotenv(ROOT / ".env")
    model = args.model or os.getenv("REASONING_REWRITE_MODEL") or os.getenv("GLM_MODEL", "glm-5.3")
    if not model.casefold().startswith("glm"):
        raise RuntimeError("Reasoning redaction is GLM-only; refusing to call a non-GLM model")
    client = make_client(model)
    public_path = args.input_dir / "messages_train.jsonl"
    private_path = args.input_dir / "teacher_trajectory_audit.jsonl"
    public_rows = [json.loads(line) for line in public_path.open(encoding="utf-8")]
    private_rows = {row["sample_id"]: row for row in
                    (json.loads(line) for line in private_path.open(encoding="utf-8"))}
    selected = public_rows if args.limit is None else public_rows[:args.limit]
    if args.shard_count > 1:
        selected = [row for position, row in enumerate(selected)
                    if position % args.shard_count == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "messages_train.jsonl"
    audit_path = args.output_dir / "reasoning_redaction_audit.jsonl"
    progress_path = args.output_dir / "reasoning_redaction_progress.json"
    existing_rows = read_jsonl_map(output_path)
    existing_audit = read_jsonl_map(audit_path)
    selected_ids = {item["sample_id"] for item in selected}
    # Write the private copy before processing.  It is deliberately separate
    # from the public stream and is never included in the model request.
    private_output = args.output_dir / "teacher_trajectory_audit.jsonl"
    if not private_output.exists():
        with private_output.open("w", encoding="utf-8", newline="\n") as out:
            for row in private_rows.values():
                if row["sample_id"] in selected_ids:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")

    previous_progress = {}
    if progress_path.exists():
        try:
            previous_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_progress = {}
    failures_by_id = {
        str(value.get("sample_id")): value
        for value in previous_progress.get("failures", [])
        if isinstance(value, dict) and value.get("sample_id")
    }
    for sample_id in existing_rows:
        failures_by_id.pop(sample_id, None)
    stats = {
        "requested": len(selected),
        "skipped_existing": len(set(existing_rows) & selected_ids),
        "newly_completed": 0,
        "api_calls": 0,
        "failed": len(failures_by_id),
    }

    def save_heartbeat(status: str, last_sample_id: str | None = None) -> None:
        completed = len(set(existing_rows) & selected_ids)
        failed = len([sample_id for sample_id in failures_by_id if sample_id in selected_ids])
        write_progress(progress_path, {
            "schema_version": "reasoning-redaction-progress-1.0",
            "status": status,
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "model": model,
            "requested": len(selected),
            "completed": completed,
            "failed": failed,
            "remaining": max(0, len(selected) - completed),
            "api_calls": stats["api_calls"],
            "last_sample_id": last_sample_id,
            "failures": [failures_by_id[k] for k in sorted(failures_by_id)
                          if k in selected_ids],
        })

    save_heartbeat("running")

    def rewrite_one(item: dict, turn: dict, private: dict) -> str:
        factual_source = turn["original_summary"] + "\n" + str(turn.get("tool_result") or "")
        # Most accepted turns are already free of private-source wording.  A
        # local validation pass can keep those summaries byte-for-byte and
        # avoid an unnecessary API call; only turns needing actual rewriting
        # are sent to the model.
        try:
            return validate_rewritten_summary(
                turn["original_summary"], private, factual_source
            )
        except ValueError:
            pass
        error = None
        for _ in range(max(1, args.max_attempts)):
            try:
                retry_note = (
                    "上一次输出未通过校验：" + error + "。"
                    "这次只返回不含任何私有来源词的 summary；"
                    "只能使用本轮原摘要中已经出现的弦、徽、分、音高或 MIDI 数值，"
                    "绝不能从其他轮、工具输出或上下文补入任何新的数值事实。"
                    if error else ""
                )
                options = {
                    "model": model, "max_tokens": 1200, "temperature": 0.2,
                    "messages": [{
                        "role": "user",
                        "content": rewrite_prompt(item, [turn], retry_note),
                    }],
                }
                if model.casefold().startswith("glm") or model == "MiniMax-M3":
                    options["thinking"] = {"type": "disabled"}
                stats["api_calls"] += 1
                payload = parse_json_object(_text_blocks(client.messages.create(**options)))
                raw = payload.get("summary")
                if raw is None:
                    # Accept the early batch shape only when it contains one
                    # item; never silently discard extra model output.
                    legacy = payload.get("summaries")
                    if isinstance(legacy, list) and len(legacy) == 1:
                        raw = legacy[0]
                if not isinstance(raw, str):
                    raise ValueError("single summary field required")
                return validate_rewritten_summary(raw, private, factual_source)
            except Exception as exc:  # retry the single turn, then fail closed
                error = f"{type(exc).__name__}: {exc}"
        raise ValueError(error or "reasoning rewrite failed")

    for item in selected:
        sample_id = item["sample_id"]
        if sample_id in existing_rows:
            continue
        private = private_rows.get(sample_id, {})
        turns = public_turns(item)
        if not turns:
            row = deepcopy(item)
            append_jsonl(output_path, row)
            existing_rows[sample_id] = row
            failures_by_id.pop(sample_id, None)
            stats["newly_completed"] += 1
            save_heartbeat("running", sample_id)
            continue
        summaries: list[str] = []
        try:
            for _, turn in turns:
                summaries.append(rewrite_one(item, turn, private))
        except Exception as exc:  # preserve the sample only in the failure report
            failures_by_id[sample_id] = {
                "sample_id": sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            stats["failed"] = len(failures_by_id)
            save_heartbeat("running", sample_id)
            continue
        rewritten = deepcopy(item)
        for (index, original), summary in zip(turns, summaries):
            rewritten["messages"][index]["content"] = summary
            audit_row = {
                "sample_id": sample_id, "message_index": index,
                "original_summary": original["original_summary"],
                "redacted_summary": summary,
            }
            append_jsonl(audit_path, audit_row)
        rewritten["generation_mode"] = str(item.get("generation_mode") or "") + "+reasoning_redacted"
        rewritten["reasoning_redaction"] = {"status": "redacted", "model": model}
        append_jsonl(output_path, rewritten)
        existing_rows[sample_id] = rewritten
        failures_by_id.pop(sample_id, None)
        stats["newly_completed"] += 1
        save_heartbeat("running", sample_id)
    report = {
        "schema_version": "reasoning-redaction-1.0", "model": model,
        "requested": len(selected), "written": len(set(existing_rows) & selected_ids),
        "redacted": len({row["sample_id"] for row in read_jsonl_map(audit_path).values()}),
        "failed": len([sample_id for sample_id in failures_by_id if sample_id in selected_ids]),
        "failures": [failures_by_id[k] for k in sorted(failures_by_id)
                      if k in selected_ids],
        "original_private_audit_preserved": True,
    }
    (args.output_dir / "reasoning_redaction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_heartbeat("completed")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if not failures_by_id else 1


if __name__ == "__main__":
    raise SystemExit(main())
