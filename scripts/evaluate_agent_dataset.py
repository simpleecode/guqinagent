#!/usr/bin/env python3
"""Evaluate every reference score listed in the leakage-group manifest.

This is a corpus baseline, not an aesthetic oracle.  It reports hard pitch /
coverage metrics and descriptive, context-aware proxies for guqin technique.
Later generated plans can be compared against these reference distributions.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "ABC_J" / "results" / "dataset_split_groups.csv"
DEFAULT_OUTPUT = ROOT / "ABC_J" / "results" / "agent_metric_baseline.json"
DEFAULT_CSV_OUTPUT = ROOT / "ABC_J" / "results" / "agent_metric_baseline.csv"
AUDIT_PATH = ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"

RIGHT_HAND = ("挑", "勾", "抹", "剔", "托", "擘", "打", "摘", "撮", "泼", "剌")
LEFT_FINGER = ("大指", "食指", "中指", "名指", "跪指")
TECHNIQUES = (
    "绰", "注", "吟", "猱", "撞", "逗", "往来", "掐起", "带起", "抓起", "爪起", "滔起",
    "进复", "退复", "进", "退", "复", "泛起", "泛止", "撮", "泼", "剌",
)
SUSTAINED_TECHNIQUES = {"吟", "猱", "往来"}
SLIDE_TECHNIQUES = {"绰", "注", "撞", "逗", "进复", "退复", "进", "退", "复", "滔起"}
DURATION_RANK = {
    "六十四分": 1, "三十二分": 2, "十六分": 3, "八分": 4,
    "四分": 5, "二分": 6, "全音": 7,
}


def _load_audit():
    spec = importlib.util.spec_from_file_location("agent_metric_pitch_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pitch audit: {AUDIT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit()


def _safe_div(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _tokens(text: str, choices: tuple[str, ...]) -> list[str]:
    # Longest first prevents 进复 from also being counted as 进 and 复.
    remaining = text
    found: list[str] = []
    for token in sorted(choices, key=len, reverse=True):
        count = remaining.count(token)
        found.extend([token] * count)
        remaining = remaining.replace(token, "")
    return found


def _is_sounding(note: dict, tonic_midi: float) -> bool:
    return any(
        AUDIT.parse_jianpu(note.get(field), tonic_midi) is not None
        for field in ("jianpu", "jianpu_alt")
    )


def _duration_rank(note: dict) -> int:
    text = str(note.get("duration", ""))
    return next((rank for label, rank in DURATION_RANK.items() if label in text), 0)


def _mode(text: str) -> str | None:
    if "泛" in text:
        return "harmonic"
    if "散" in text:
        return "open"
    if any(token in text for token in LEFT_FINGER) or "徽" in text:
        return "stopped"
    return None


def _technique_context(note: dict, technique: str, sounding: bool) -> bool | None:
    """Return a conservative context proxy, or None when not automatable."""
    if technique in SUSTAINED_TECHNIQUES:
        return sounding and _duration_rank(note) >= DURATION_RANK["八分"]
    if technique in SLIDE_TECHNIQUES:
        return sounding
    if technique in {"掐起", "带起", "抓起", "爪起", "撮", "泼", "剌"}:
        return sounding
    if technique in {"泛起", "泛止"}:
        return True
    return None


def evaluate_score(row: dict[str, str], tolerance_cents: float) -> dict:
    directory = Path(row["final_data_path"])
    readable_path = directory / "jianpu_jianzi_readable.json"
    data = json.loads(readable_path.read_text(encoding="utf-8"))
    metadata = data.get("metadata") or {}
    tonic_midi = AUDIT.parse_tonic_midi(metadata)
    notes = data.get("notes", [])
    sounding_notes = [note for note in notes if _is_sounding(note, tonic_midi)]
    nonempty_actions = sum(bool(str(note.get("jianzi", "")).strip()) for note in sounding_notes)
    pitch_unavailable_reason = None
    try:
        audit = AUDIT.audit(data, tolerance_cents)
        pitch = audit["summary"]
    except ValueError as exc:
        if "open_strings must contain 7 strings" not in str(exc):
            raise
        pitch_unavailable_reason = "missing_tuning_open_strings"
        pitch = {
            "compared_notes": 0, "matched": 0, "match_rate": None,
            "mean_absolute_cents": None,
        }

    technique_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    right_counts: Counter[str] = Counter()
    left_counts: Counter[str] = Counter()
    context_valid = 0
    context_evaluable = 0
    for note in notes:
        text = str(note.get("jianzi", ""))
        sounding = _is_sounding(note, tonic_midi)
        techniques = _tokens(text, TECHNIQUES)
        technique_counts.update(techniques)
        right_counts.update(_tokens(text, RIGHT_HAND))
        left_counts.update(_tokens(text, LEFT_FINGER))
        mode = _mode(text)
        if mode and sounding:
            mode_counts[mode] += 1
        for technique in techniques:
            valid = _technique_context(note, technique, sounding)
            if valid is not None:
                context_evaluable += 1
                context_valid += int(valid)

    total_techniques = sum(technique_counts.values())
    sounding_count = len(sounding_notes)
    jianzipu = str(data.get("jianzipu", ""))
    bracket_valid = jianzipu.count("[") == jianzipu.count("]")
    sections = metadata.get("sections") or []
    return {
        "leakage_group_id": row["leakage_group_id"],
        "score_key": row["score_key"],
        "score_id": row.get("score_id", ""),
        "title": metadata.get("score_title") or row.get("candidate_title", ""),
        "rounds": row.get("rounds", ""),
        "tonic": metadata.get("tonic", row.get("tonic", "")),
        "notes_length_manifest": int(row["notes_length"]) if row.get("notes_length") else None,
        "hard": {
            "pitch_status": "available" if pitch_unavailable_reason is None else "unavailable",
            "pitch_unavailable_reason": pitch_unavailable_reason,
            "target_sounding_notes": sounding_count,
            "pitch_compared_notes": pitch["compared_notes"],
            "pitch_matched_notes": pitch["matched"],
            "pitch_accuracy_at_50c": pitch["match_rate"],
            "pitch_coverage": (
                _safe_div(pitch["compared_notes"], sounding_count)
                if pitch_unavailable_reason is None else None
            ),
            "pitch_joint_success": (
                _safe_div(pitch["matched"], sounding_count)
                if pitch_unavailable_reason is None else None
            ),
            "pitch_mae_cents": pitch["mean_absolute_cents"],
            "action_text_coverage": _safe_div(nonempty_actions, sounding_count),
            "bracket_valid": bracket_valid,
            "section_count": len(sections),
        },
        "yun_proxy": {
            "technique_total": total_techniques,
            "techniques_per_100_sounding_notes": round(
                100 * total_techniques / sounding_count, 6
            ) if sounding_count else None,
            "technique_context_accuracy": _safe_div(context_valid, context_evaluable),
            "technique_context_coverage": _safe_div(context_evaluable, total_techniques),
            "technique_counts": dict(sorted(technique_counts.items())),
            "mode_counts": dict(sorted(mode_counts.items())),
            "right_hand_counts": dict(sorted(right_counts.items())),
            "left_finger_counts": dict(sorted(left_counts.items())),
        },
        "source": str(readable_path),
    }


def _distribution(rows: list[dict], path: tuple[str, str]) -> dict:
    values = [row[path[0]][path[1]] for row in rows if row[path[0]][path[1]] is not None]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "q10": round(percentile(0.10), 6),
        "q25": round(percentile(0.25), 6),
        "q75": round(percentile(0.75), 6),
        "q90": round(percentile(0.90), 6),
    }


def evaluate_manifest(manifest: Path, tolerance_cents: float) -> dict:
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    per_score: list[dict] = []
    errors: list[dict] = []
    for index, row in enumerate(manifest_rows, 1):
        try:
            per_score.append(evaluate_score(row, tolerance_cents))
        except Exception as exc:  # keep the manifest auditable and continue
            errors.append({"score_key": row.get("score_key"), "error": str(exc)})
        if index % 25 == 0:
            print(f"evaluated {index}/{len(manifest_rows)}", flush=True)

    pitch_rows = [item for item in per_score if item["hard"]["pitch_status"] == "available"]
    total_target = sum(item["hard"]["target_sounding_notes"] for item in pitch_rows)
    total_compared = sum(item["hard"]["pitch_compared_notes"] for item in per_score)
    total_matched = sum(item["hard"]["pitch_matched_notes"] for item in per_score)
    total_techniques = sum(item["yun_proxy"]["technique_total"] for item in per_score)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "manifest_rows": len(manifest_rows),
        "leakage_groups": len({row["leakage_group_id"] for row in manifest_rows}),
        "evaluated_scores": len(per_score),
        "tolerance_cents": tolerance_cents,
        "errors": errors,
        "aggregate": {
            "pitch_available_scores": len(pitch_rows),
            "pitch_unavailable_scores": len(per_score) - len(pitch_rows),
            "target_sounding_notes": total_target,
            "pitch_compared_notes": total_compared,
            "pitch_matched_notes": total_matched,
            "micro_pitch_accuracy_at_50c": _safe_div(total_matched, total_compared),
            "micro_pitch_coverage": _safe_div(total_compared, total_target),
            "micro_pitch_joint_success": _safe_div(total_matched, total_target),
            "technique_total": total_techniques,
            "techniques_per_100_sounding_notes": round(
                100 * total_techniques /
                sum(item["hard"]["target_sounding_notes"] for item in per_score), 6
            ) if per_score else None,
            "distributions": {
                "pitch_accuracy_at_50c": _distribution(per_score, ("hard", "pitch_accuracy_at_50c")),
                "pitch_coverage": _distribution(per_score, ("hard", "pitch_coverage")),
                "pitch_joint_success": _distribution(per_score, ("hard", "pitch_joint_success")),
                "action_text_coverage": _distribution(per_score, ("hard", "action_text_coverage")),
                "techniques_per_100_sounding_notes": _distribution(
                    per_score, ("yun_proxy", "techniques_per_100_sounding_notes")
                ),
                "technique_context_accuracy": _distribution(
                    per_score, ("yun_proxy", "technique_context_accuracy")
                ),
            },
        },
        "per_score": per_score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--tolerance-cents", type=float, default=50.0)
    args = parser.parse_args()
    report = evaluate_manifest(args.manifest, args.tolerance_cents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = []
    for item in report["per_score"]:
        csv_rows.append({
            "leakage_group_id": item["leakage_group_id"],
            "score_key": item["score_key"],
            "title": item["title"],
            "rounds": item["rounds"],
            "tonic": item["tonic"],
            "target_sounding_notes": item["hard"]["target_sounding_notes"],
            "pitch_status": item["hard"]["pitch_status"],
            "pitch_accuracy_at_50c": item["hard"]["pitch_accuracy_at_50c"],
            "pitch_coverage": item["hard"]["pitch_coverage"],
            "pitch_joint_success": item["hard"]["pitch_joint_success"],
            "pitch_mae_cents": item["hard"]["pitch_mae_cents"],
            "action_text_coverage": item["hard"]["action_text_coverage"],
            "section_count": item["hard"]["section_count"],
            "techniques_per_100_notes": item["yun_proxy"]["techniques_per_100_sounding_notes"],
            "technique_context_accuracy": item["yun_proxy"]["technique_context_accuracy"],
            "source": item["source"],
        })
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({
        "output": str(args.output),
        "csv_output": str(args.csv_output),
        "manifest_rows": report["manifest_rows"],
        "evaluated_scores": report["evaluated_scores"],
        "errors": len(report["errors"]),
        **report["aggregate"],
    }, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
