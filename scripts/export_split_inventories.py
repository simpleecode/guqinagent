#!/usr/bin/env python3
"""Export simple ID inventories for split verification.

Writes one-ID-per-line lists plus a consistency report into
ABC_J/results/split_inventories/:

- all_score_ids.txt                     241 score keys
- pitch_pass_40_score_ids.txt           scores with pitch_accuracy_at_50c > 0.40
- pitch_fail_40_score_ids.txt           the complement (for quick diffing)
- {train,validation,test}_score_ids.txt score-level split (agent_dataset_splits.csv)
- {train,validation,test}_phrase_ids.txt phrase-level split (inferred_v3)

The report cross-checks: leakage groups never cross splits; the split column
in dataset_split_groups.csv agrees with agent_dataset_splits.csv; per-score
pitch rates agree between agent_metric_baseline.csv and
dataset_split_groups.csv; every inferred_v3 phrase's split equals its score's
split; the pitch filter's coverage inside each split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "ABC_J/results"
SPLITS = ("train", "validation", "test")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_list(path: Path, items: list[str]) -> None:
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inferred-dir", type=Path,
                        default=ROOT / "ABC_J/agent_training/inferred_v3")
    parser.add_argument("--pitch-threshold", type=float, default=0.40)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "split_inventories")
    args = parser.parse_args()

    groups = read_csv(RESULTS / "dataset_split_groups.csv")
    splits_rows = read_csv(RESULTS / "agent_dataset_splits.csv")
    metrics = read_csv(RESULTS / "agent_metric_baseline.csv")

    all_scores = [row["score_key"] for row in groups]
    split_of = {row["score_key"]: row["split"] for row in splits_rows}
    group_of = {row["score_key"]: row["leakage_group_id"] for row in groups}

    metric_rate = {row["score_key"]: float(row["pitch_accuracy_at_50c"])
                   for row in metrics if row.get("pitch_accuracy_at_50c")}
    groups_rate = {row["score_key"]: float(row["pitch_match_rate"]) for row in groups
                   if row.get("pitch_match_rate") not in (None, "")}

    # dataset_split_groups.csv is the authoritative manifest (experiment plan
    # §4); agent_metric_baseline.csv predates the after-fix pitch audit and
    # carries degenerate 0.0 accuracies for a dozen scores, so it is fallback.
    rate_of = dict(groups_rate)
    for key, value in metric_rate.items():
        rate_of.setdefault(key, value)
    pass_ids = sorted(key for key, rate in rate_of.items()
                      if rate > args.pitch_threshold)
    fail_ids = sorted(set(all_scores) - set(pass_ids))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_list(args.output_dir / "all_score_ids.txt", sorted(all_scores))
    write_list(args.output_dir / "pitch_pass_40_score_ids.txt", pass_ids)
    write_list(args.output_dir / "pitch_fail_40_score_ids.txt", fail_ids)

    phrase_ids = {split: [] for split in SPLITS}
    for split in SPLITS:
        path = args.inferred_dir / f"inferred_trajectories_{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            phrase_ids[split] = [json.loads(line)["trajectory_id"]
                                 for line in handle if line.strip()]
        write_list(args.output_dir / f"{split}_phrase_ids.txt", phrase_ids[split])

    score_ids = {}
    for split in SPLITS:
        score_ids[split] = sorted(key for key, value in split_of.items() if value == split)
        write_list(args.output_dir / f"{split}_score_ids.txt", score_ids[split])

    # ---- consistency checks ----
    problems = []

    missing_split = sorted(set(all_scores) - set(split_of))
    unknown_split = sorted(set(split_of) - set(all_scores))
    if missing_split or unknown_split:
        problems.append({"check": "score_coverage", "missing_in_splits": missing_split,
                         "unknown_in_splits": unknown_split})

    overlap = {f"{a}&{b}": sorted(set(score_ids[a]) & set(score_ids[b]))
               for a, b in (("train", "validation"), ("train", "test"),
                            ("validation", "test"))}
    if any(overlap.values()):
        problems.append({"check": "split_overlap", "detail": overlap})

    groups_csv_split = {row["score_key"]: (row.get("split") or "").strip()
                        for row in groups}
    disagree = sorted(key for key in all_scores
                      if groups_csv_split[key] and groups_csv_split[key] != split_of.get(key))
    if disagree:
        problems.append({"check": "split_column_disagreement", "scores": disagree[:20],
                         "count": len(disagree)})

    group_splits = defaultdict(set)
    for key, split in split_of.items():
        group_splits[group_of.get(key, "?")].add(split)
    crossing = {group: sorted(values) for group, values in group_splits.items()
                if len(values) > 1}
    if crossing:
        problems.append({"check": "leakage_group_crossing", "groups": crossing})

    rate_gap = sorted((key, round(metric_rate[key], 6), round(groups_rate[key], 6))
                      for key in metric_rate.keys() & groups_rate.keys()
                      if abs(metric_rate[key] - groups_rate[key]) > 1e-6)
    data_quality_notes = []
    if rate_gap:
        # Informational: metric baseline is older than the after-fix pitch
        # audit (2026-08-12); its zeros are artifacts, the manifest wins.
        data_quality_notes.append({
            "note": "pitch_rate_source_disagreement",
            "manifest_wins": True,
            "count": len(rate_gap),
            "examples": rate_gap[:12],
            "recommendation": "re-run the metric baseline after the tonic audit fix",
        })

    phrase_mismatch = []
    for split in SPLITS:
        for trajectory_id in phrase_ids[split]:
            score_key = trajectory_id.rsplit("-p", 1)[0]
            if split_of.get(score_key) != split:
                phrase_mismatch.append({"trajectory_id": trajectory_id,
                                        "score_split": split_of.get(score_key),
                                        "phrase_split": split})
    if phrase_mismatch:
        problems.append({"check": "phrase_split_mismatch", "count": len(phrase_mismatch),
                         "examples": phrase_mismatch[:10]})

    pass_set = set(pass_ids)
    report = {
        "schema_version": "split-inventories-1.0",
        "sources": {
            "groups_csv": str(RESULTS / "dataset_split_groups.csv"),
            "splits_csv": str(RESULTS / "agent_dataset_splits.csv"),
            "metrics_csv": str(RESULTS / "agent_metric_baseline.csv"),
            "phrases": str(args.inferred_dir),
        },
        "counts": {
            "scores_total": len(all_scores),
            "scores_by_split": {split: len(score_ids[split]) for split in SPLITS},
            "phrases_by_split": {split: len(phrase_ids[split]) for split in SPLITS},
            "pitch_pass_over_40": len(pass_ids),
            "pitch_fail_over_40": len(fail_ids),
        },
        "pitch_pass_by_split": {
            split: {"pass": sum(1 for key in score_ids[split] if key in pass_set),
                    "total": len(score_ids[split])} for split in SPLITS
        },
        "pitch_rate_range": {
            "min": min(rate_of.values()), "max": max(rate_of.values()),
        } if rate_of else {},
        "leakage_groups": len(set(group_of.values())),
        "consistency_problems": problems,
        "data_quality_notes": data_quality_notes,
        "all_checks_passed": not problems,
    }
    (args.output_dir / "split_inventories_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["counts"], ensure_ascii=False))
    print("pitch_pass_by_split:", json.dumps(report["pitch_pass_by_split"]))
    print("all_checks_passed:", report["all_checks_passed"])
    if problems:
        print(json.dumps(problems, ensure_ascii=False, indent=2)[:2000])
    print(f"inventories written to {args.output_dir}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
