#!/usr/bin/env python3
"""Apply mapped-jianzi quality filtering to existing inferred trajectories.

Most old phrases are retained byte-for-byte. Dropped scores and phrases fully
inside a bad trailing tail are omitted; only phrases crossing a trim boundary
are replaced by the corresponding phrase from the quality-filtered rebuild.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "ABC_J/agent_training/source_quality_audit_v1/quality_report.json"
DEFAULT_OLD = ROOT / "ABC_J/agent_training/inferred_v3"
DEFAULT_REBUILT = ROOT / "ABC_J/agent_training/inferred_quality_filtered_v1"
DEFAULT_OUTPUT = ROOT / "ABC_J/agent_training/inferred_quality_filtered_minimal_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def classify(item: dict, drops: set[str], cutoffs: dict[str, int]) -> str:
    key = item.get("score_key")
    if key in drops:
        return "drop_score"
    if key not in cutoffs:
        return "keep"
    cutoff = cutoffs[key]
    event_range = item.get("input", {}).get("event_range", {})
    start = event_range.get("start")
    end = event_range.get("end_exclusive")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"missing event_range for {key}/{item.get('phrase_id')}")
    if start > cutoff:
        return "drop_trailing_phrase"
    if end - 1 > cutoff:
        return "replace_crossing_phrase"
    return "keep"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--rebuilt-dir", type=Path, default=DEFAULT_REBUILT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    drops = {r["score_key"] for r in audit["records"] if r["classification"] == "drop_score"}
    cutoffs = {
        r["score_key"]: int(r["last_nonempty_index"])
        for r in audit["records"]
        if r["classification"] == "trim_trailing" and r["last_nonempty_index"] is not None
    }
    args.output_dir.mkdir(parents=True)
    report: dict = {"dropped_score_keys": sorted(drops), "trim_cutoffs": cutoffs, "splits": {}}
    for split in ("train", "validation", "test"):
        old_path = args.old_dir / f"inferred_trajectories_{split}.jsonl"
        rebuilt_path = args.rebuilt_dir / f"inferred_trajectories_{split}.jsonl"
        old_rows = read_jsonl(old_path)
        rebuilt_rows = read_jsonl(rebuilt_path)
        crossing_ids: set[tuple[str, str]] = set()
        kept: list[dict] = []
        counters = {"old": len(old_rows), "kept_old": 0, "keep": 0, "drop_score": 0, "drop_trailing_phrase": 0, "replace_crossing_phrase": 0, "inserted_rebuilt": 0}
        for item in old_rows:
            decision = classify(item, drops, cutoffs)
            counters[decision] += 1
            if decision == "replace_crossing_phrase":
                crossing_ids.add((item.get("score_key", ""), item.get("phrase_id", "")))
            elif decision == "keep":
                kept.append(item)
                counters["kept_old"] += 1
        replacements = [item for item in rebuilt_rows if (item.get("score_key", ""), item.get("phrase_id", "")) in crossing_ids]
        if len(replacements) != len(crossing_ids):
            found = {(item.get("score_key", ""), item.get("phrase_id", "")) for item in replacements}
            missing = sorted(crossing_ids - found)
            raise ValueError(f"missing rebuilt crossing phrases in {split}: {missing}")
        kept.extend(replacements)
        kept.sort(key=lambda item: (item.get("score_key", ""), item.get("phrase_id", "")))
        counters["inserted_rebuilt"] = len(replacements)
        write_jsonl(args.output_dir / old_path.name, kept)
        report["splits"][split] = counters

    # The minimal merge should have the same trajectory set as the full rebuild.
    old_ids = {(item["trajectory_id"]): item for split in ("train", "validation", "test") for item in read_jsonl(args.output_dir / f"inferred_trajectories_{split}.jsonl")}
    rebuilt_ids = {(item["trajectory_id"]): item for split in ("train", "validation", "test") for item in read_jsonl(args.rebuilt_dir / f"inferred_trajectories_{split}.jsonl")}
    report["same_trajectory_ids_as_full_rebuild"] = set(old_ids) == set(rebuilt_ids)
    report["minimal_trajectory_count"] = len(old_ids)
    if not report["same_trajectory_ids_as_full_rebuild"]:
        report["id_difference"] = {"missing": sorted(set(rebuilt_ids) - set(old_ids)), "extra": sorted(set(old_ids) - set(rebuilt_ids))}
    summary_source = args.rebuilt_dir / "inferred_trajectories_summary.json"
    if summary_source.exists():
        shutil.copy2(summary_source, args.output_dir / summary_source.name)
    (args.output_dir / "minimal_merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
