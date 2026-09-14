#!/usr/bin/env python3
"""Build leakage-safe evaluation pairs from the current text-only trajectories.

Unlike the legacy exporter, this does not require verified structured patches.
The sealed target is the literal ``jianzi_text`` for every row.  It applies the
same mapped-source quality rule to held-out scores and drops fully blank phrases.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OMITTED_PLACEHOLDER = "无（由于是再作部分，省略）"
SCORE_BLACKLIST_PATH = ROOT / "ABC_J/results/score_blacklist.json"


def load_blacklisted_score_keys() -> set[str]:
    """Read the shared score blacklist used by selection and extraction."""
    payload = json.loads(SCORE_BLACKLIST_PATH.read_text(encoding="utf-8"))
    values = payload.get("score_keys") or []
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"invalid score_keys in {SCORE_BLACKLIST_PATH}")
    return set(values)


def render_rows(notes: list[dict], actions: list[dict], *, readonly: bool) -> str:
    by_index = {int(action["source_index"]): action for action in actions}
    lines = ["序号｜简谱｜ABC｜时值｜谱面减字"]
    for note in notes:
        index = int(note["index"])
        abc = str(note.get("abc") or "-")
        jianpu = str(note.get("jianpu") or note.get("jianpu_alt") or "休止")
        if note.get("jianpu_alt") and abc.startswith("["):
            jianpu += " " + str(note["jianpu_alt"])
        action = by_index.get(index)
        if action is not None and action.get("jianzi_text") is not None:
            text = str(action.get("jianzi_text") or "")
            surface = "" if text == "" else f"[{text}]"
        elif abc == "|" or note.get("duration") == "小节线":
            surface = "｜"
        elif abc.startswith("z") or jianpu.startswith("0"):
            surface = "[—]"
        elif abc.startswith("-") or "延音" in jianpu:
            surface = "[续音]"
        else:
            surface = "[空]" if readonly else "[减字待填写]"
        if readonly and note.get("notation_omitted"):
            surface = f"[{OMITTED_PLACEHOLDER}]"
        lines.append(f"{index}｜{jianpu}｜{abc}｜{note.get('duration') or '-'}｜{surface}")
    if readonly and any(note.get("notation_omitted") for note in notes):
        lines.append(f"注｜标注〔{OMITTED_PLACEHOLDER}〕的音为再作省略音：仍需完整指法，最终减字谱不显示其减字。")
    return "\n".join(lines)


def render_public_prompt(item: dict) -> str:
    payload = item["input"]
    metadata = payload.get("metadata") or {}
    tuning = payload.get("normalized_tuning") or {}
    handoff = payload.get("phrase_handoff") or {}
    phrase_id = handoff.get("phrase_id") or item.get("phrase_id")
    previous = handoff.get("previous_phrase")
    lines = [
        f"谱名｜{metadata.get('title') or metadata.get('score_title') or item.get('score_key', '未命名')}",
        f"调弦｜{tuning.get('name') or '未知'}｜{tuning.get('open_midi') or []}",
        f"当前段｜{phrase_id}",
        f"泛音区间｜当前段开始时{'是' if item.get('harmonic_region_at_start') else '否'}",
        "泛音区间提示｜常规写法：进入泛音区间时在减字开头添加“泛起”；结束时可在当前减字末尾添加“泛止”，也常在随后的延音行单独填写“泛止”，不要强行合并。",
        "",
    ]
    if previous:
        lines.extend([
            f"【只读前一段 {previous.get('phrase_id')}｜{previous.get('status', '已确认')}】",
            render_rows(previous.get("notes") or [], previous.get("actions") or [], readonly=True),
            "注｜前一段已在此完整提供，直接使用；仅需更早段时才先 list_context 再 expand_context。",
            "",
        ])
    lines.extend([
        f"【当前段 {phrase_id}｜待编辑】",
        render_rows(
            handoff.get("current_phrase") or payload["notes_without_jianzi"],
            item.get("baseline_plan", {}).get("actions") or [],
            readonly=False,
        ),
    ])
    return "\n".join(lines)


def load_audit_module():
    path = ROOT / "ABC_J" / "scripts" / "filter_training_data.py"
    spec = importlib.util.spec_from_file_location("mapped_quality", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def public_runtime_item(item: dict) -> dict:
    """Retain executable public state without any current/previous gold actions."""
    handoff = deepcopy(item["input"].get("phrase_handoff") or {})
    for key in ("previous_phrase", "exit_state_target", "context_source"):
        handoff.pop(key, None)
    return {
        "trajectory_id": item["trajectory_id"],
        "score_key": item["score_key"],
        "phrase_id": item["phrase_id"],
        "split": item["split"],
        "input": {
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "normalized_tuning": deepcopy(item["input"].get("normalized_tuning") or {}),
            "event_range": deepcopy(item["input"]["event_range"]),
            "notes_without_jianzi": deepcopy(item["input"]["notes_without_jianzi"]),
            "phrase_handoff": handoff,
        },
        "baseline_plan": deepcopy(item["baseline_plan"]),
    }


def clip_item(item: dict, cutoff: int | None) -> dict:
    if cutoff is None:
        return item
    result = deepcopy(item)
    notes = [n for n in result["input"]["notes_without_jianzi"] if int(n["index"]) <= cutoff]
    actions = [a for a in result["reference_plan"]["actions"] if int(a["source_index"]) <= cutoff]
    result["input"]["notes_without_jianzi"] = notes
    result["reference_plan"]["actions"] = actions
    handoff = result["input"].get("phrase_handoff") or {}
    handoff["current_phrase"] = [n for n in handoff.get("current_phrase", notes) if int(n["index"]) <= cutoff]
    end = max((int(n["index"]) for n in notes), default=cutoff) + 1
    result["input"]["event_range"]["end_exclusive"] = end
    if handoff.get("event_range"):
        handoff["event_range"]["end_exclusive"] = end
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=ROOT / "ABC_J/agent_training/inferred_v10_repeat_semantics_prompt_final",
    )
    parser.add_argument(
        "--sealed-output-dir", type=Path,
        default=ROOT / "ABC_J/agent_training/evaluation_text_protocol_v2",
    )
    parser.add_argument(
        "--public-output-dir", type=Path,
        default=ROOT / "train/eval_inputs_v2_text_protocol",
    )
    args = parser.parse_args()
    args.sealed_output_dir.mkdir(parents=True, exist_ok=True)
    args.public_output_dir.mkdir(parents=True, exist_ok=True)
    audit = load_audit_module()
    blacklisted_scores = load_blacklisted_score_keys()
    report = {
        "schema_version": "agent-text-eval-build-2.0",
        "blacklisted_score_keys": sorted(blacklisted_scores),
        "splits": {},
    }

    for split in ("validation", "test"):
        source = args.input_dir / f"inferred_trajectories_{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        quality = {key: audit.mapped_score_record(audit.mapped_path(key)) for key in sorted({r["score_key"] for r in rows})}
        sealed_path = args.sealed_output_dir / f"evaluation_pairs_{split}.jsonl"
        public_path = args.public_output_dir / f"{split}.jsonl"
        counts = Counter(source_phrases=len(rows))
        with sealed_path.open("w", encoding="utf-8", newline="\n") as sealed, public_path.open("w", encoding="utf-8", newline="\n") as public:
            for original in rows:
                if original["score_key"] in blacklisted_scores:
                    counts["dropped_blacklisted_score"] += 1
                    continue
                rule = quality[original["score_key"]]
                if rule["classification"] == "drop_score":
                    counts["dropped_low_coverage_score"] += 1
                    continue
                start = int(original["input"]["event_range"]["start"])
                cutoff = rule["last_nonempty_index"] if rule["classification"] == "trim_trailing" else None
                if cutoff is not None and start > int(cutoff):
                    counts["dropped_trailing_blank_phrase"] += 1
                    continue
                item = clip_item(original, int(cutoff) if cutoff is not None else None)
                actions = item["reference_plan"]["actions"]
                if not any(str(a.get("jianzi_text") or "").strip() for a in actions):
                    counts["dropped_fully_blank_phrase"] += 1
                    continue
                prompt = render_public_prompt(item)
                public_payload = {
                    "schema_version": "agent-eval-input-2.0",
                    "sample_id": item["trajectory_id"],
                    "split": split,
                    "score_key": item["score_key"],
                    "phrase_id": item["phrase_id"],
                    "public_prompt": prompt,
                    "runtime_item": public_runtime_item(item),
                }
                public_payload["input_sha256"] = fingerprint(public_payload)
                sealed_payload = {
                    "schema_version": "agent-eval-pair-2.0",
                    **public_payload,
                    "reference": {"actions": actions},
                    "training_allowed": False,
                    "source_trajectory_version": item.get("trajectory_version"),
                }
                public.write(json.dumps(public_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                sealed.write(json.dumps(sealed_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["included"] += 1
                if cutoff is not None and int(original["input"]["event_range"]["end_exclusive"]) - 1 > int(cutoff):
                    counts["clipped_crossing_phrase"] += 1
        report["splits"][split] = dict(counts)
    (args.sealed_output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
