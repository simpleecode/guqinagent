#!/usr/bin/env python3
"""统计训练阶段减字的可解析音高与音高匹配情况。

本脚本复用 ``scripts/audit_jianpu_jianzi_pitch.py`` 的解析算法，仅在送入旧
解析器前把明确的“七弦”等弦号表示归一化为“7弦”，不把
“有数字/有弦”这类文本启发式当作音高证据。输入是一份原始 inferred
trajectory JSONL 和最终教师审计 JSONL；对每条 Fingering/Guqinizer
accepted_plan 逐音重建审计，输出阶段级统计和可追溯明细。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PITCH_AUDIT_PATH = ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"
spec = importlib.util.spec_from_file_location("pitch_audit", PITCH_AUDIT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load pitch audit: {PITCH_AUDIT_PATH}")
PITCH_AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PITCH_AUDIT)


STAGES = ("fingering", "guqinizer")
ZH_STRING_DIGITS = {
    "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7",
}
FINGER_ABBREVIATIONS = {"大": "大指", "名": "名指", "中": "中指", "食": "食指", "跪": "跪指"}
HUI_DIGITS = "一二三四五六七八九十"
RIGHT_HAND_TOKENS = "抹挑勾剔擘托打摘撮"


def normalize_jianzi_for_pitch(text: str) -> str:
    """Normalize spelling only; leave the legacy pitch algorithm unchanged.

    The corpus commonly writes ``七弦`` while the old parser's string token is
    deliberately narrow (``7弦``).  Converting the one unambiguous Chinese
    numeral immediately before ``弦`` prevents a representation mismatch from
    being reported as an unparseable pitch.  Hui numerals and all technique
    text are left untouched and are still interpreted by the old parser.
    """
    normalized = re.sub(
        r"([一二三四五六七])弦",
        lambda match: ZH_STRING_DIGITS[match.group(1)] + "弦",
        text,
    )
    # Common compact model spellings omit the final 弦 and/or abbreviate the
    # left-hand finger.  These two patterns are unambiguous: the final Chinese
    # numeral is a string because it follows a right-hand token (or a hui plus
    # right-hand token), not a free-standing position number.
    normalized = re.sub(
        rf"^散(?P<right>[{RIGHT_HAND_TOKENS}])(?P<string>[一二三四五六七])$",
        lambda match: (
            "散" + match.group("right")
            + ZH_STRING_DIGITS[match.group("string")] + "弦"
        ),
        normalized,
    )
    normalized = re.sub(
        rf"^(?P<finger>[大名中食跪])(?P<hui>[{HUI_DIGITS}]+)"
        rf"(?P<right>[{RIGHT_HAND_TOKENS}])(?P<string>[一二三四五六七])$",
        lambda match: (
            FINGER_ABBREVIATIONS[match.group("finger")]
            + match.group("hui") + "徽" + match.group("right")
            + ZH_STRING_DIGITS[match.group("string")] + "弦"
        ),
        normalized,
    )
    return normalized


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def stage_from_sample_id(sample_id: str) -> str | None:
    lowered = sample_id.lower()
    if "-fingering_" in lowered or lowered.endswith("-fingering"):
        return "fingering"
    if "-guqinization" in lowered or "-guqinizer" in lowered:
        return "guqinizer"
    return None


def base_trajectory_id(sample_id: str) -> str:
    for marker in ("-fingering_agent", "-guqinization", "-guqinizer"):
        if marker in sample_id:
            return sample_id.split(marker, 1)[0]
    return sample_id


def build_pitch_audit(source: dict[str, Any], accepted_plan: dict[str, Any],
                      tolerance_cents: float) -> dict[str, Any]:
    """Run the existing pitch audit against a plan's text-only final state."""
    phrase = (source.get("input") or {}).get("phrase_handoff") or {}
    notes = phrase.get("current_phrase") or (source.get("input") or {}).get(
        "notes_without_jianzi"
    ) or []
    action_by_index = {
        int(action["source_index"]): action
        for action in (accepted_plan or {}).get("actions", [])
        if isinstance(action, dict) and action.get("source_index") is not None
    }
    audit_notes = []
    for note in notes:
        index = int(note["index"])
        action = action_by_index.get(index) or {}
        text = action.get("jianzi_text")
        # None is the structured representation of a not-yet-filled row; the
        # old auditor treats both it and the empty string as non-parseable.
        raw_text = "" if text is None else str(text)
        audit_notes.append({
            "index": index,
            "jianpu": note.get("jianpu"),
            "jianpu_alt": note.get("jianpu_alt"),
            "jianzi": normalize_jianzi_for_pitch(raw_text),
            "jianzi_original": raw_text,
            "lyric": note.get("lyric", ""),
        })
    data = {
        "metadata": (source.get("input") or {}).get("metadata") or {},
        "notes": audit_notes,
    }
    report = PITCH_AUDIT.audit(data, tolerance_cents)
    for row, note in zip(report["details"], audit_notes):
        row["jianzi_original"] = note["jianzi_original"]
    return report


def classify(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    nonempty_rows = [
        row for row in report["details"] if str(row.get("jianzi", "")).strip()
    ]
    parseable = [
        row for row in report["details"]
        if row.get("status") in {"matched", "mismatched"}
    ]
    matched = [row for row in parseable if row.get("status") == "matched"]
    skip_reasons = Counter(
        str(row.get("reason", "unknown"))
        for row in report["details"]
        if row.get("status") == "skipped"
    )
    return {
        "parseable_pitch_rows": len(parseable),
        "nonempty_jianzi_rows": len(nonempty_rows),
        "all_jianzi_empty": not nonempty_rows,
        "matched_pitch_rows": len(matched),
        "mismatched_pitch_rows": sum(
            row.get("status") == "mismatched" for row in parseable
        ),
        "unparseable_reason_counts": dict(skip_reasons),
        "no_parseable_pitch": not parseable,
        "parseable_and_at_least_one_match": bool(parseable and matched),
        # A phrase used as supervised pitch data should not be admitted merely
        # because one accidental row happens to agree.  This stricter class is
        # deliberately kept alongside the historical one so old reports stay
        # interpretable.
        "parseable_and_at_least_two_matches_or_half_matched": bool(
            parseable
            and (len(matched) >= 2 or len(matched) * 2 >= len(parseable))
        ),
        "audit_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="inferred trajectories JSONL (repeat to add a fallback source)")
    parser.add_argument("--audit", type=Path, required=True,
                        help="teacher_trajectory_audit JSONL")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="write JSON report")
    parser.add_argument("--tolerance-cents", type=float, default=50.0)
    args = parser.parse_args()

    sources: dict[str, dict[str, Any]] = {}
    for source_path in args.source:
        for source in read_jsonl(source_path):
            trajectory_id = source.get("trajectory_id")
            if trajectory_id:
                # The first source is authoritative; later sources fill only
                # legacy/rerun IDs absent from the current source.
                sources.setdefault(str(trajectory_id), source)

    stage_counts = Counter()
    classifications = Counter()
    details: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in read_jsonl(args.audit):
        sample_id = str(row.get("sample_id", ""))
        stage = stage_from_sample_id(sample_id)
        if stage not in STAGES:
            continue
        trajectory_id = base_trajectory_id(sample_id)
        source = sources.get(trajectory_id)
        if source is None:
            skipped.append({"sample_id": sample_id, "reason": "source_not_found"})
            continue
        accepted_plan = (row.get("teacher_private") or {}).get("accepted_plan")
        if not isinstance(accepted_plan, dict):
            skipped.append({"sample_id": sample_id, "reason": "accepted_plan_missing"})
            continue
        report = build_pitch_audit(source, accepted_plan, args.tolerance_cents)
        result = classify(report)
        stage_counts[stage] += 1
        if result["no_parseable_pitch"]:
            classifications[(stage, "no_parseable_pitch")] += 1
        if result["parseable_and_at_least_one_match"]:
            classifications[(stage, "parseable_and_at_least_one_match")] += 1
        if result["parseable_and_at_least_two_matches_or_half_matched"]:
            classifications[(stage, "parseable_and_at_least_two_matches_or_half_matched")] += 1
        if result["parseable_pitch_rows"] and not result["matched_pitch_rows"]:
            classifications[(stage, "parseable_but_no_match")] += 1
        details.append({
            "sample_id": sample_id,
            "trajectory_id": trajectory_id,
            "stage": stage,
            "classification": {
                key: value for key, value in result.items()
                if key != "audit_summary"
            },
            "audit_summary": result["audit_summary"],
        })

    by_stage: dict[str, Any] = {}
    for stage in STAGES:
        rows = [item for item in details if item["stage"] == stage]
        stage_reasons = Counter()
        stage_phase_reasons = Counter()
        for item in rows:
            stage_reasons.update(item["classification"]["unparseable_reason_counts"])
            if item["classification"]["no_parseable_pitch"]:
                stage_phase_reasons.update(
                    item["classification"]["unparseable_reason_counts"].keys()
                )
        by_stage[stage] = {
            "total_phases": len(rows),
            "all_jianzi_empty_count": sum(
                item["classification"]["all_jianzi_empty"] for item in rows
            ),
            "nonempty_jianzi_rows": sum(
                item["classification"]["nonempty_jianzi_rows"] for item in rows
            ),
            "no_parseable_pitch_count": sum(
                item["classification"]["no_parseable_pitch"] for item in rows
            ),
            "parseable_and_at_least_one_match_count": sum(
                item["classification"]["parseable_and_at_least_one_match"]
                for item in rows
            ),
            "parseable_and_at_least_two_matches_or_half_matched_count": sum(
                item["classification"]["parseable_and_at_least_two_matches_or_half_matched"]
                for item in rows
            ),
            "parseable_but_no_match_count": sum(
                item["classification"]["parseable_pitch_rows"] > 0
                and not item["classification"]["matched_pitch_rows"]
                for item in rows
            ),
            "unparseable_reason_counts": dict(stage_reasons),
            "no_parseable_phase_reason_presence": dict(stage_phase_reasons),
        }

    overall_reasons = Counter()
    overall_phase_reasons = Counter()
    for item in details:
        overall_reasons.update(
            item["classification"]["unparseable_reason_counts"]
        )
        if item["classification"]["no_parseable_pitch"]:
            overall_phase_reasons.update(
                item["classification"]["unparseable_reason_counts"].keys()
            )

    result = {
        "schema_version": "training-phase-pitch-parse-stats-1.0",
        "algorithm": {
            "source": "scripts/audit_jianpu_jianzi_pitch.py",
            "normalization": "仅将紧邻‘弦’的一至七中文数字归一化为阿拉伯数字；不改变减字语义",
            "tolerance_cents": args.tolerance_cents,
            "definition": {
                "no_parseable_pitch": "该阶段所有减字经旧审计解析后都没有可比较的 MIDI 音高",
                "parseable_and_at_least_one_match": "该阶段至少有一个可解析减字，且至少一音与简谱 MIDI 在容差内匹配",
                "parseable_and_at_least_two_matches_or_half_matched": "该阶段至少有两个可解析减字匹配，或匹配音数不少于全部可解析音的一半",
            },
        },
        "input": {
            "source": [str(path) for path in args.source],
            "audit": str(args.audit),
            "source_trajectory_count": len(sources),
            "audit_phase_count": sum(stage_counts.values()),
            "skipped_count": len(skipped),
        },
        "summary": {
            "total_phases": len(details),
            "all_jianzi_empty_count": sum(
                item["classification"]["all_jianzi_empty"] for item in details
            ),
            "nonempty_jianzi_rows": sum(
                item["classification"]["nonempty_jianzi_rows"] for item in details
            ),
            "no_parseable_pitch_count": sum(
                item["classification"]["no_parseable_pitch"] for item in details
            ),
            "parseable_and_at_least_one_match_count": sum(
                item["classification"]["parseable_and_at_least_one_match"]
                for item in details
            ),
            "parseable_and_at_least_two_matches_or_half_matched_count": sum(
                item["classification"]["parseable_and_at_least_two_matches_or_half_matched"]
                for item in details
            ),
            "parseable_but_no_match_count": sum(
                item["classification"]["parseable_pitch_rows"] > 0
                and not item["classification"]["matched_pitch_rows"]
                for item in details
            ),
            "unparseable_reason_counts": dict(overall_reasons),
            "no_parseable_phase_reason_presence": dict(overall_phase_reasons),
        },
        "by_stage": by_stage,
        "details": details,
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "summary": result["summary"],
        "by_stage": result["by_stage"],
        "skipped": len(skipped),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
