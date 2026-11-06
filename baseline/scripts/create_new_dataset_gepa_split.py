#!/usr/bin/env python3
"""Freeze family-isolated splits and a 10-20 score pilot for the current dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test"
TRAIN = DATA / "inferred_trajectories_train.jsonl"
TEST = REPO / "train/eval_inputs_v2_text_protocol/test.jsonl"
TEST50 = TEST.parent / "test50/test_runtime.jsonl"
SEED = "new-data-gepa-family-20261029-v1"
PILOT_RNG_SEED = "random-whole10-20261104-v1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def title(value) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")))


def choose_groups(counts: dict[str, int], ratio: float) -> set[str]:
    target = round(sum(counts.values()) * ratio)
    reachable = {0: ()}
    for family in sorted(counts, key=rank):
        for total, groups in list(reachable.items()):
            new_total = total + counts[family]
            if new_total not in reachable:
                reachable[new_total] = groups + (family,)
    feasible = [n for n, groups in reachable.items() if 0 < len(groups) < len(counts)]
    return set(reachable[min(feasible, key=lambda n: (abs(n - target), n))])


def family_of(row: dict) -> str:
    return str(row.get("score_family_id") or row["score_key"])


def order(row: dict):
    notes = row["input"]["notes_without_jianzi"]
    return (min((int(n["index"]) for n in notes), default=-1), row["phrase_id"])


def contiguous_from_p0001(group: list[dict]) -> bool:
    ids = [r["phrase_id"] for r in group]
    return ids == [f"p{i:04d}" for i in range(1, len(ids) + 1)]


def select_scores_random(rows: list[dict], families: set[str], number: int,
                         require_contiguous: bool = False) -> list[str]:
    """固定种子随机抽取 number 首曲子（每家族最多一首），只按公开特征排序后洗牌，不看标注难度。"""
    by_score = defaultdict(list)
    for row in rows:
        if family_of(row) in families:
            by_score[str(row["score_key"])].append(row)
    for key in by_score:
        by_score[key].sort(key=order)
    if require_contiguous:  # 整曲模式：只收连续完整导出的曲子，缺口曲静默排除
        by_score = {k: g for k, g in by_score.items() if contiguous_from_p0001(g)}
    rng = random.Random(f"{SEED}:{PILOT_RNG_SEED}")
    # 按家族分组后随机：先抽家族再抽家族内一首，保证家族隔离与可复现
    by_family = defaultdict(list)
    for key in sorted(by_score):
        by_family[family_of(by_score[key][0])].append(key)
    pool = sorted(by_family)  # 家族名排序保证确定性
    rng.shuffle(pool)
    selected, used = [], set()
    for family in pool:
        if len(selected) == number:
            break
        if family in used:
            continue
        keys = sorted(by_family[family])
        selected.append(keys[rng.randrange(len(keys))])
        used.add(family)
    if len(selected) < number:
        raise ValueError(f"Only {len(selected)} distinct families available for {number} pilot scores")
    return selected


def collect_pilot_rows(rows: list[dict], key: str, prefix: int) -> list[dict]:
    group = sorted([r for r in rows if r["score_key"] == key], key=order)
    if prefix > 0:
        return group[:prefix]
    # 整曲模式：必须从真正的第一句开始且连续无缺口，防止静默截断
    if not contiguous_from_p0001(group):
        raise ValueError(f"score {key} phrases not contiguous from p0001; "
                         "whole-score mode only accepts fully exported scores")
    return group


def select_scores(rows: list[dict], families: set[str], number: int, prefix: int) -> list[str]:
    # Choose by public musical features only, never by reference difficulty or score.
    if prefix == 0:
        return select_scores_random(rows, families, number, require_contiguous=True)
    by_score = defaultdict(list)
    for row in rows:
        if family_of(row) in families:
            by_score[str(row["score_key"])].append(row)
    candidates = []
    for key, group in by_score.items():
        if len(group) < prefix:
            continue
        group.sort(key=order)
        # Eligible exports can omit phrases. The low-cost pilot must start at
        # the true first phrase and must not bridge a missing prefix silently.
        if [r["phrase_id"] for r in group[:prefix]] != [f"p{i:04d}" for i in range(1, prefix + 1)]:
            continue
        notes = [n for row in group[:prefix] for n in row["input"]["notes_without_jianzi"]]
        chord = any("[" in str(n.get("abc", "")) for n in notes)
        unusual_tuning = str(group[0]["input"].get("metadata", {}).get("tuning", {}).get("name", "")) != "正调"
        bucket = (chord, unusual_tuning, min(2, len(notes) // 50))
        candidates.append((bucket, rank(key), family_of(group[0]), key))
    buckets = defaultdict(list)
    for item in sorted(candidates):
        buckets[item[0]].append(item)
    selected, used = [], set()
    while len(selected) < number:
        progressed = False
        for bucket in sorted(buckets):
            while buckets[bucket]:
                _, _, family, key = buckets[bucket].pop(0)
                if family not in used:
                    selected.append(key)
                    used.add(family)
                    progressed = True
                    break
            if len(selected) == number:
                break
        if not progressed:
            raise ValueError(f"Only {len(selected)} distinct families available for {number} pilot scores")
    return selected


def write_frozen(path: Path, text: str, check: bool, refresh: bool = False) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            if refresh and not check:
                path.write_text(text, encoding="utf-8", newline="\n")
            else:
                raise SystemExit(f"Existing split differs; refusing to overwrite: {path}")
    elif check:
        raise SystemExit(f"Split file missing: {path}")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-scores", type=int, default=10)
    parser.add_argument("--prefix-phrases", type=int, default=0,
                        help="每首取前N句；0=整曲全部乐句（只收完整连续导出的曲子）")
    parser.add_argument("--selection", choices=("random", "bucketed"), default="random",
                        help="pilot 曲目选取方式：random=固定种子随机抽取（默认），bucketed=旧版特征分桶")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--refresh-plan", action="store_true", help="explicitly replace generated plan files before any training")
    args = parser.parse_args()
    if not 10 <= args.pilot_scores <= 20 or args.prefix_phrases < 0:
        parser.error("pilot scores must be 10-20; prefix phrases must be >= 0 (0 = whole score)")
    rows, main_test, test50 = read(TRAIN), read(TEST), read(TEST50)
    counts = Counter(family_of(row) for row in rows)
    score_families = defaultdict(set)
    for row in rows:
        score_families[row["score_key"]].add(family_of(row))
    if any(len(groups) != 1 for groups in score_families.values()):
        raise SystemExit("A score belongs to more than one family")
    validation = choose_groups(dict(counts), 0.25)
    remaining = {f: count for f, count in counts.items() if f not in validation}
    check = choose_groups(remaining, 0.20)
    optimize = set(remaining) - check
    group_roles = {"iteration_optimize": optimize, "fixed_validation": validation,
                   "iteration_check": check}
    splits = {}
    for name, families in group_roles.items():
        selected = [row for row in rows if family_of(row) in families]
        splits[name] = {
            "families": sorted(families), "family_count": len(families),
            "score_keys": sorted({row["score_key"] for row in selected}),
            "rows": len(selected), "ratio": len(selected) / len(rows),
        }
        splits[name]["score_key_count"] = len(splits[name]["score_keys"])
    train_keys = set(score_families)
    train_ids = {row["trajectory_id"] for row in rows}
    main_keys = {row["score_key"] for row in main_test}
    test50_keys = {row["score_key"] for row in test50}
    train_titles = {title(row["input"].get("metadata", {}).get("score_title")) for row in rows}
    test_titles = {title(row["runtime_item"]["input"].get("metadata", {}).get("score_title")) for row in test50}
    overlaps = sorted(train_keys & main_keys)
    titles_overlap = sorted((train_titles & test_titles) - {""})
    if train_keys & test50_keys or titles_overlap:
        raise SystemExit("test50 overlaps new training scores or normalized titles; audit before splitting")
    confirm_n = max(2, round(args.pilot_scores * 0.1875))
    pareto_n = max(2, round(args.pilot_scores * 0.1875))
    feedback_n = args.pilot_scores - confirm_n - pareto_n
    pilot = {}
    for name, source, number in (("feedback", "iteration_optimize", feedback_n),
                                 ("pareto", "fixed_validation", pareto_n),
                                 ("confirmation", "iteration_check", confirm_n)):
        keys = select_scores(rows, group_roles[source], number, args.prefix_phrases)
        selected = []
        for key in keys:
            selected.extend(collect_pilot_rows(rows, key, args.prefix_phrases))
        pilot[name] = {
            "source_split": source, "score_keys": keys,
            "families": sorted({family_of(r) for r in selected}),
            "sample_ids": [r["trajectory_id"] for r in selected],
            "score_count": len(keys), "rows": len(selected),
        }
    audit = {
        "train": {"rows": len(rows), "scores": len(train_keys), "families": len(counts)},
        "main_test": {"rows": len(main_test), "scores": len(main_keys),
                      "overlapping_train_score_keys": overlaps,
                      "overlapping_train_sample_ids": sum(r["sample_id"] in train_ids for r in main_test),
                      "status": "quarantined_for_independent_test_due_to_overlap",
                      "disjoint_score_keys_for_diagnostics_only": sorted(main_keys - train_keys)},
        "test50": {"rows": len(test50), "scores": len(test50_keys),
                   "public_sha256": sha(TEST50),
                   "reference_sha256": sha(TEST50.parent / "evaluation_pairs_test.jsonl"),
                   "overlapping_train_score_keys": [], "normalized_title_overlaps": [],
                   "status": "final_test_candidate_not_used_for_optimization",
                   "family_audit": "score keys and normalized titles checked; no family IDs exported in test50, original manifest asserts zero leakage-group overlap"},
    }
    identity = {"source_file": TRAIN.name, "source_sha256": sha(TRAIN)}
    manifest = {
        "schema_version": "new-data-gepa-split-1.0", **identity,
        "seed": SEED, "grouping": "score_family_id with score_key fallback",
        "splits": splits, "test_audit": audit,
    }
    pilot_manifest = {
        "schema_version": "new-data-gepa-pilot-1.0", **identity, "seed": SEED,
        "score_count": args.pilot_scores, "prefix_phrases": args.prefix_phrases,
        "selection": ("fixed-seed random whole scores (one per family, contiguous export only); "
                      "no label-based selection" if args.prefix_phrases == 0 else
                      "public-feature buckets + fixed hash ranking; one score per family; no label-based selection"),
        "splits": pilot, "test_sources": {"main": str(TEST), "test50": str(TEST50)},
        "test_hashes": {"main_public": sha(TEST), "test50_public": sha(TEST50),
                        "test50_reference": sha(TEST50.parent / "evaluation_pairs_test.jsonl")},
        "scorer_sha256": sha(REPO / "train/scripts/score_agent_predictions.py"),
    }
    inner = {**manifest, "schema_version": "prompt-iteration-inner-split-new-1.0"}
    outer = {**manifest, "schema_version": "prompt-optimization-family-split-new-1.0"}
    outer["splits"] = {"fixed_validation": splits["fixed_validation"], "prompt_optimization_train": {
        "families": sorted(optimize | check), "score_keys": sorted(train_keys - set(splits["fixed_validation"]["score_keys"])),
        "rows": splits["iteration_optimize"]["rows"] + splits["iteration_check"]["rows"],
    }}
    outputs = {
        "new_dataset_gepa_split_manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "prompt_iteration_subsplit_manifest.json": json.dumps(inner, ensure_ascii=False, indent=2) + "\n",
        "prompt_optimization_family_split_manifest.json": json.dumps(outer, ensure_ascii=False, indent=2) + "\n",
        "gepa_pilot_manifest.json": json.dumps(pilot_manifest, ensure_ascii=False, indent=2) + "\n",
        "fixed_validation_score_keys_25pct.txt": "\n".join(splits["fixed_validation"]["score_keys"]) + "\n",
        "prompt_optimization_train_score_keys_75pct.txt": "\n".join(outer["splits"]["prompt_optimization_train"]["score_keys"]) + "\n",
        "iteration_optimize_score_keys.txt": "\n".join(splits["iteration_optimize"]["score_keys"]) + "\n",
        "iteration_train_check_score_keys.txt": "\n".join(splits["iteration_check"]["score_keys"]) + "\n",
        "iteration_diagnostic_sample_ids.txt": "\n".join(pilot["feedback"]["sample_ids"]) + "\n",
        "iteration_check_sample_ids.txt": "\n".join(pilot["confirmation"]["sample_ids"]) + "\n",
    }
    explanation = ["# 新版 GEPA 数据划分（2026-10-29）", "", "原始 JSONL 不改写；名单按曲谱家族隔离。", "",
                   "| 部分 | 曲谱 | 家族 | 短语 | 用途 |", "|---|---:|---:|---:|---|"]
    descriptions = {"iteration_optimize": "读取标注差异、反思改词", "fixed_validation": "D_pareto，只用成绩选候选",
                    "iteration_check": "独立确认，不反馈具体错题给改词模型"}
    for name, item in splits.items():
        explanation.append(f"| {name} | {item['score_key_count']} | {item['family_count']} | {item['rows']} | {descriptions[name]} |")
    phrase_desc = f"前 {args.prefix_phrases} 段" if args.prefix_phrases else "整曲全部乐句"
    explanation += ["", f"训练源 SHA-256：`{identity['source_sha256']}`", "",
                    f"初步试验固定 {args.pilot_scores} 首，每首{phrase_desc}（{args.selection} 选取）：", ""]
    for name, item in pilot.items():
        explanation.append(f"- {name}：{item['score_count']} 首 / {item['rows']} 段；" + "、".join(item["score_keys"]))
    explanation += ["", "主 test 与训练有曲谱重叠，隔离为历史诊断来源，不用于独立判优。",
                    "test50 全部留到最终评测，不参与本划分及改词；已核对曲谱编号与标题无交集，家族ID未随test50导出，不能声称完成独立家族审计。",
                    "各轮HTML必须记录抽样、标注差异反馈、提示词改动、评分、接受/淘汰和最终选择。"]
    outputs["新版提示词训练数据划分说明.md"] = "\n".join(explanation) + "\n"
    # Preflight all outputs before writing anything.
    for name, value in outputs.items():
        path = DATA / name
        if path.exists() and path.read_text(encoding="utf-8") != value and not args.refresh_plan:
            raise SystemExit(f"Refusing to overwrite a different fixed split: {path}")
    for name, value in outputs.items():
        write_frozen(DATA / name, value, args.check, args.refresh_plan)
    print(json.dumps({"status": "verified" if args.check else "created", "splits": splits,
                      "pilot": pilot, "test_audit": audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
