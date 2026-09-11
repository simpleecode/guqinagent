#!/usr/bin/env python3
"""导出音高审计两类 phase 对应的完整双阶段轨迹。

默认按 phrase 成对筛选：只有当同一 phrase 的 Fingering 和 Guqinizer
都属于“无可解析音高”或“至少一音匹配”时，才保留该 phrase 的两个阶段。
这样不会产生只有单阶段的半条轨迹，也不会把“可解析但完全不匹配”的阶段混入。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


STAGES = ("fingering", "guqinizer")
ROOT = Path(__file__).resolve().parents[2]
_PITCH_STATS_MODULE = None


def canonical_reference_class(item: dict[str, Any]) -> dict[str, Any]:
    """Classify a canonical inferred phrase using the existing pitch audit."""
    audit_path = ROOT / "scripts" / "audit_training_phase_pitch_parse_stats.py"
    global _PITCH_STATS_MODULE
    if _PITCH_STATS_MODULE is None:
        spec = importlib.util.spec_from_file_location("pitch_stats_for_export", audit_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {audit_path}")
        _PITCH_STATS_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PITCH_STATS_MODULE)
    module = _PITCH_STATS_MODULE
    report = module.build_pitch_audit(item, item.get("reference_plan") or {}, 50.0)
    return module.classify(report)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def stage_eligible(item: dict[str, Any]) -> bool:
    # ``classify()`` returns the flags at the top level in the new inferred
    # protocol; older stats rows may wrap them under ``classification``.
    cls = item.get("classification") or item
    return bool(
        cls.get("no_parseable_pitch")
        or cls.get("parseable_and_at_least_one_match")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=Path,
                        help="完整公开轨迹 messages_train.jsonl（旧版筛选模式）")
    parser.add_argument("--stats", type=Path,
                        help="training_phase_pitch_parse_stats.json（旧版筛选模式）")
    parser.add_argument("--inferred-dir", type=Path,
                        help="新的 inferred_gqs_v12 目录；直接按参考减字计算 phrase 音高资格")
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "test"),
        default=("train",),
        help="新协议模式要筛选的数据 split；默认仅 train，避免教师训练轨迹混入验证/测试曲谱",
    )
    parser.add_argument("--require-match", action="store_true",
                        help="只保留至少一个可解析且匹配的 phrase，不包含全不可解析类")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # OrderedDict preserves source order and makes the ID list stable.
    phase_stats: dict[str, dict[str, dict[str, Any]]] = OrderedDict()
    canonical_rows: dict[str, dict[str, Any]] = OrderedDict()
    if args.inferred_dir:
        for split in args.splits:
            path = args.inferred_dir / f"inferred_trajectories_{split}.jsonl"
            if not path.exists():
                continue
            for item in read_jsonl(path):
                trajectory_id = str(item.get("trajectory_id") or "")
                if not trajectory_id:
                    continue
                result = canonical_reference_class(item)
                canonical_rows[trajectory_id] = item
                phase_stats[trajectory_id] = {
                    "fingering": result,
                    "guqinizer": result,
                }
    else:
        if not args.messages or not args.stats:
            raise ValueError("旧版模式必须同时提供 --messages 和 --stats；新协议请使用 --inferred-dir")
        with args.stats.open(encoding="utf-8") as handle:
            stats_payload = json.load(handle)
        stats_items = stats_payload.get("details", [])
        if not isinstance(stats_items, list):
            raise ValueError("stats JSON must contain a details array")
        for item in stats_items:
            trajectory_id = str(item.get("trajectory_id", ""))
            stage = str(item.get("stage", ""))
            if trajectory_id and stage in STAGES:
                phase_stats.setdefault(trajectory_id, {})[stage] = item
    selected_phrases = {
        trajectory_id
        for trajectory_id, stages in phase_stats.items()
        if all(stage in stages and (
            stages[stage].get("parseable_and_at_least_one_match")
            if args.require_match else stage_eligible(stages[stage])
        ) for stage in STAGES)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_messages = args.output_dir / "messages_train.jsonl"
    kept = 0
    total = 0
    kept_by_stage = {stage: 0 for stage in STAGES}
    kept_ids: list[str] = []
    seen_ids: set[str] = set()
    if args.messages:
        with output_messages.open("w", encoding="utf-8", newline="\n") as out:
            for row in read_jsonl(args.messages):
                total += 1
                provenance = row.get("provenance") or {}
                trajectory_id = str(provenance.get("source_trajectory_id") or "")
                if trajectory_id not in selected_phrases:
                    continue
                out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
                agent_stage = str(row.get("agent_stage", "")).lower()
                if "fingering" in agent_stage:
                    kept_by_stage["fingering"] += 1
                elif "guqin" in agent_stage:
                    kept_by_stage["guqinizer"] += 1
                if trajectory_id not in seen_ids:
                    seen_ids.add(trajectory_id)
                    kept_ids.append(trajectory_id)
    else:
        # Rerun input is canonical one-row-per-phrase data; no model-generated
        # messages exist yet, so emit phrase IDs only and let the generator
        # produce both stages from these IDs.
        for trajectory_id in selected_phrases:
            kept_ids.append(trajectory_id)
        kept = 0
        total = len(canonical_rows)

    ids_path = args.output_dir / "pitch_eligible_phrase_ids.txt"
    ids_path.write_text("".join(f"{item}\n" for item in kept_ids), encoding="utf-8")
    manifest = {
        "schema_version": "pitch-eligible-trajectory-export-1.0",
        "selection": {
            "included_phase_classes": (["parseable_and_at_least_one_match"]
                                        if args.require_match else [
                                            "no_parseable_pitch",
                                            "parseable_and_at_least_one_match",
                                        ]),
            "pairing": "both_fingering_and_guqinizer_must_be_eligible",
            "excluded_class": "parseable_but_no_match",
        },
        "inputs": {"messages": str(args.messages) if args.messages else None,
                    "stats": str(args.stats) if args.stats else None,
                    "inferred_dir": str(args.inferred_dir) if args.inferred_dir else None,
                    "splits": list(args.splits) if args.inferred_dir else None},
        "counts": {
            "source_message_rows": total,
            "exported_message_rows": kept,
            "source_phrase_count": len(phase_stats),
            "exported_phrase_count": len(kept_ids),
            "exported_fingering_count": kept_by_stage["fingering"],
            "exported_guqinizer_count": kept_by_stage["guqinizer"],
        },
        "files": {
            "messages": output_messages.name if args.messages else None,
            "phrase_ids": ids_path.name,
        },
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
