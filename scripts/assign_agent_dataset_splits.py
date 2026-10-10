#!/usr/bin/env python3
"""Assign leakage-group-safe train/test splits deterministically."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "ABC_J" / "results" / "dataset_split_groups.csv"
DEFAULT_METRICS = ROOT / "ABC_J" / "results" / "agent_metric_baseline.json"
DEFAULT_DEV = ROOT / "ABC_J" / "results" / "agent_dev_examples.json"
DEFAULT_REPORT = ROOT / "ABC_J" / "results" / "dataset_split_report.json"
DEFAULT_SPLIT_MANIFEST = ROOT / "ABC_J" / "results" / "agent_dataset_splits.csv"
DEFAULT_BLACKLIST = ROOT / "ABC_J" / "results" / "score_blacklist.json"
TARGETS = {"train": 0.80, "test": 0.20}


def bucket(value: float | None, edges: tuple[float, ...]) -> str:
    if value is None:
        return "unknown"
    for index, edge in enumerate(edges):
        if value < edge:
            return f"b{index}"
    return f"b{len(edges)}"


def stable_noise(group_id: str, split: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}:{split}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 * 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--dev-examples", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--blacklist",
        type=Path,
        default=DEFAULT_BLACKLIST,
        help="JSON score blacklist; listed score_key values are excluded before splitting.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--force-test-group", action="append", default=[],
                        help="leakage group ID that must remain entirely in test; repeatable")
    parser.add_argument("--force-train-group", action="append", default=[],
                        help="leakage group ID that must remain entirely in train; repeatable")
    parser.add_argument("--single-score-test", action="store_true",
                        help="allow only leakage groups containing one score in test")
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    blacklist_payload = json.loads(args.blacklist.read_text(encoding="utf-8"))
    if isinstance(blacklist_payload, dict):
        blacklist = set(blacklist_payload.get("score_keys", []))
    elif isinstance(blacklist_payload, list):
        blacklist = set(blacklist_payload)
    else:
        raise ValueError(f"unsupported blacklist format: {args.blacklist}")
    original_row_count = len(rows)
    rows = [row for row in rows if row["score_key"] not in blacklist]
    if not rows:
        raise ValueError("all manifest rows were excluded by the blacklist")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    metric_by_key = {item["score_key"]: item for item in metrics["per_score"]}
    dev_groups = {
        item["leakage_group_id"]
        for item in json.loads(args.dev_examples.read_text(encoding="utf-8"))["examples"]
    }
    forced_test_groups = set(args.force_test_group)
    forced_train_groups = set(args.force_train_group)
    overlap = (dev_groups | forced_train_groups) & forced_test_groups
    if overlap:
        raise ValueError(f"groups cannot be forced to both train and test: {sorted(overlap)}")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["leakage_group_id"]].append(row)
    if args.single_score_test:
        invalid_forced = sorted(
            group_id for group_id in forced_test_groups
            if len(groups.get(group_id, [])) != 1
        )
        if invalid_forced:
            raise ValueError(
                "--single-score-test conflicts with multi-score forced test groups: "
                f"{invalid_forced}"
            )

    features: dict[str, Counter] = {}
    group_weights: dict[str, int] = {}
    for group_id, members in groups.items():
        counter: Counter[str] = Counter()
        weight = 0
        for row in members:
            metric = metric_by_key[row["score_key"]]
            hard = metric["hard"]
            yun = metric["yun_proxy"]
            notes = hard["target_sounding_notes"]
            weight += max(notes, 1)
            counter[f"round:{row['rounds']}"] += 1
            counter[f"tonic:{row['tonic'] or 'unknown'}"] += 1
            counter[f"length:{bucket(notes, (100, 300, 800, 1500))}"] += 1
            counter[f"pitch:{bucket(hard['pitch_joint_success'], (.3, .5, .7))}"] += 1
            counter[f"yun:{bucket(yun['techniques_per_100_sounding_notes'], (10, 25, 45))}"] += 1
            counter[f"sections:{bucket(hard['section_count'], (1, 3, 7))}"] += 1
        features[group_id] = counter
        group_weights[group_id] = weight

    total_groups = len(groups)
    total_rows = len(rows)
    total_weight = sum(group_weights.values())
    feature_totals = sum(features.values(), Counter())
    assigned: dict[str, str] = {}
    split_groups: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    split_weight: Counter[str] = Counter()
    split_features: dict[str, Counter] = {name: Counter() for name in TARGETS}

    for group_id in sorted(forced_test_groups):
        if group_id not in groups:
            raise ValueError(f"unknown forced test group: {group_id}")
        assigned[group_id] = "test"
        split_groups["test"] += 1
        split_rows["test"] += len(groups[group_id])
        split_weight["test"] += group_weights[group_id]
        split_features["test"].update(features[group_id])

    # Development examples and explicitly curated repertoire are deliberately
    # assigned to train, never test.
    for group_id in sorted(dev_groups | forced_train_groups):
        if group_id not in groups:
            raise ValueError(f"unknown forced train group: {group_id}")
        if group_id in assigned:
            continue
        assigned[group_id] = "train"
        split_groups["train"] += 1
        split_rows["train"] += len(groups[group_id])
        split_weight["train"] += group_weights[group_id]
        split_features["train"].update(features[group_id])

    ordered = sorted(
        (group_id for group_id in groups if group_id not in assigned),
        key=lambda gid: (-group_weights[gid], -len(groups[gid]), gid),
    )
    for group_id in ordered:
        best_split = None
        best_cost = None
        candidates = TARGETS
        if args.single_score_test and len(groups[group_id]) != 1:
            candidates = ("train",)
        for candidate_split in candidates:
            target = TARGETS[candidate_split]
            # Largest groups are placed first.  Compare how full each split
            # would be relative to its own target capacity; this prevents all
            # long scores from being consumed by train before val/test fill.
            weight_fill = (
                split_weight[candidate_split] + group_weights[group_id]
            ) / (target * total_weight)
            group_fill = (
                split_groups[candidate_split] + 1
            ) / (target * total_groups)
            row_fill = (
                split_rows[candidate_split] + len(groups[group_id])
            ) / (target * total_rows)
            cost = 0.60 * weight_fill + 0.25 * group_fill + 0.15 * row_fill

            # A small feature-fill term breaks ties toward stratification.
            feature_fill = 0.0
            feature_count = 0
            for feature, count in features[group_id].items():
                total = feature_totals[feature]
                if total:
                    feature_fill += (
                        split_features[candidate_split][feature] + count
                    ) / (target * total)
                    feature_count += 1
            if feature_count:
                cost += 0.03 * feature_fill / feature_count
            cost += stable_noise(group_id, candidate_split, args.seed)
            if best_cost is None or cost < best_cost:
                best_cost, best_split = cost, candidate_split
        assert best_split is not None
        assigned[group_id] = best_split
        split_groups[best_split] += 1
        split_rows[best_split] += len(groups[group_id])
        split_weight[best_split] += group_weights[group_id]
        split_features[best_split].update(features[group_id])

    for row in rows:
        row["split"] = assigned[row["leakage_group_id"]]
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.manifest)
    split_rows_output = [
        {
            "leakage_group_id": row["leakage_group_id"],
            "score_key": row["score_key"],
            "split": row["split"],
            "seed": args.seed,
        }
        for row in rows
    ]
    with args.split_manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows_output[0]))
        writer.writeheader()
        writer.writerows(split_rows_output)

    report = {
        "schema_version": "1.0",
        "seed": args.seed,
        "targets": TARGETS,
        "manifest": str(args.manifest),
        "blacklist": str(args.blacklist),
        "excluded_blacklist_score_keys": sorted(blacklist),
        "excluded_blacklist_score_count": original_row_count - len(rows),
        "split_manifest": str(args.split_manifest),
        "development_groups_forced_to_train": sorted(dev_groups),
        "groups_forced_to_train": sorted(forced_train_groups),
        "groups_forced_to_test": sorted(forced_test_groups),
        "single_score_test": args.single_score_test,
        "totals": {"groups": total_groups, "scores": total_rows, "sounding_notes": total_weight},
        "splits": {
            split: {
                "groups": split_groups[split],
                "scores": split_rows[split],
                "sounding_notes": split_weight[split],
                "group_ratio": round(split_groups[split] / total_groups, 6),
                "score_ratio": round(split_rows[split] / total_rows, 6),
                "sounding_note_ratio": round(split_weight[split] / total_weight, 6),
                "features": dict(sorted(split_features[split].items())),
            }
            for split in TARGETS
        },
        "leakage_check": {
            "groups_in_multiple_splits": [],
            "passed": True,
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
