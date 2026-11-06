#!/usr/bin/env python3
"""Create a reproducible, family-isolated inner split for prompt iteration.

The existing 75% prompt-optimization partition is divided into:

* iteration_optimize: labels may be inspected to diagnose prompt errors;
* iteration_check: labels are used only for scoring candidate generalization.

The original JSONL files are never modified.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "guqinagent/ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test"
SOURCE = DATA_ROOT / "inferred_trajectories_train.jsonl"
OUTER_MANIFEST = DATA_ROOT / "prompt_optimization_family_split_manifest.json"
SEED = "prompt-iteration-inner-family-split-20261027-v1"
CHECK_RATIO = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def choose_check_families(counts: dict[str, int], target_rows: int) -> set[str]:
    """Deterministic subset-sum selection closest to the requested row count."""
    ordered = sorted(counts, key=stable_rank)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for family in ordered:
        weight = counts[family]
        additions = {
            total + weight: chosen + (family,)
            for total, chosen in list(reachable.items())
            if total + weight not in reachable
        }
        reachable.update(additions)
    best = min(
        (total for total, chosen in reachable.items() if chosen and len(chosen) < len(ordered)),
        key=lambda total: (abs(total - target_rows), total),
    )
    return set(reachable[best])


def representative_ids(rows: list[dict], families: set[str], limit: int, label: str) -> list[str]:
    candidates = []
    seen_families = set()
    for row in rows:
        family = str(row.get("score_family_id") or row.get("score_key"))
        sample_id = str(row.get("trajectory_id") or "")
        if family not in families or not sample_id.endswith("-p0001"):
            continue
        notes = (row.get("input") or {}).get("notes_without_jianzi") or []
        attacks = sum(
            note.get("event_index") is not None
            and not str(note.get("abc") or "").startswith(("-", "z", "|"))
            for note in notes
        )
        has_chord = any("[" in str(note.get("abc") or "") for note in notes)
        if 6 <= attacks <= 24 and (row.get("reference_plan") or {}).get("actions"):
            candidates.append((has_chord, attacks // 5, stable_rank(f"{label}:{sample_id}"), family, sample_id))
    buckets: dict[tuple[bool, int], list[tuple]] = defaultdict(list)
    for item in candidates:
        buckets[(item[0], item[1])].append(item)
    for values in buckets.values():
        values.sort(key=lambda item: item[2])
    selected = []
    deferred = []
    bucket_names = sorted(buckets, key=str)
    while len(selected) < limit and any(buckets.values()):
        for bucket in bucket_names:
            while buckets[bucket] and buckets[bucket][0][3] in seen_families:
                deferred.append(buckets[bucket].pop(0))
            if buckets[bucket] and len(selected) < limit:
                item = buckets[bucket].pop(0)
                selected.append(item[4])
                seen_families.add(item[3])
    # The confirmation partition can contain fewer eligible families than the
    # requested number of samples.  Fill the remainder with different score
    # keys from those already isolated families; cross-split family isolation
    # remains intact.
    if len(selected) < limit:
        deferred.extend(item for values in buckets.values() for item in values)
        for item in sorted(deferred, key=lambda value: value[2]):
            if item[4] not in selected:
                selected.append(item[4])
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise RuntimeError(f"Could only select {len(selected)} of {limit} representatives for {label}")
    return selected


def main() -> int:
    if (DATA_ROOT / "new_dataset_gepa_split_manifest.json").exists():
        raise SystemExit("新版划分已冻结；请运行 create_new_dataset_gepa_split.py --check 校验。")
    outer = json.loads(OUTER_MANIFEST.read_text(encoding="utf-8"))
    allowed_scores = set(outer["splits"]["prompt_optimization_train"]["score_keys"])
    rows = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("score_key")) in allowed_scores:
                rows.append(row)

    family_counts: Counter[str] = Counter(
        str(row.get("score_family_id") or row.get("score_key")) for row in rows
    )
    target = round(len(rows) * CHECK_RATIO)
    check_families = choose_check_families(dict(family_counts), target)
    optimize_families = set(family_counts) - check_families

    score_families: dict[str, set[str]] = defaultdict(set)
    split_rows = Counter()
    for row in rows:
        family = str(row.get("score_family_id") or row.get("score_key"))
        score_families[family].add(str(row.get("score_key")))
        split_rows["iteration_check" if family in check_families else "iteration_optimize"] += 1

    optimize_scores = sorted({key for family in optimize_families for key in score_families[family]})
    check_scores = sorted({key for family in check_families for key in score_families[family]})
    diagnostic_ids = representative_ids(rows, optimize_families, 10, "diagnostic")
    check_ids = representative_ids(rows, check_families, 16, "check")

    manifest = {
        "schema_version": "prompt-iteration-inner-split-1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": SOURCE.name,
        "source_sha256": sha256(SOURCE),
        "outer_manifest": OUTER_MANIFEST.name,
        "outer_manifest_sha256": sha256(OUTER_MANIFEST),
        "seed": SEED,
        "grouping": "score_family_id; score_key fallback",
        "scope": "Only the existing 75% prompt_optimization_train partition",
        "splits": {
            "iteration_optimize": {
                "families": sorted(optimize_families),
                "score_keys": optimize_scores,
                "family_count": len(optimize_families),
                "score_key_count": len(optimize_scores),
                "rows": split_rows["iteration_optimize"],
                "representative_sample_ids": diagnostic_ids,
            },
            "iteration_check": {
                "families": sorted(check_families),
                "score_keys": check_scores,
                "family_count": len(check_families),
                "score_key_count": len(check_scores),
                "rows": split_rows["iteration_check"],
                "representative_sample_ids": check_ids,
            },
        },
    }

    outputs = {
        DATA_ROOT / "iteration_optimize_score_keys.txt": "\n".join(optimize_scores) + "\n",
        DATA_ROOT / "iteration_train_check_score_keys.txt": "\n".join(check_scores) + "\n",
        DATA_ROOT / "iteration_diagnostic_sample_ids.txt": "\n".join(diagnostic_ids) + "\n",
        DATA_ROOT / "iteration_check_sample_ids.txt": "\n".join(check_ids) + "\n",
        DATA_ROOT / "prompt_iteration_subsplit_manifest.json": json.dumps(
            manifest, ensure_ascii=False, indent=2
        ) + "\n",
    }
    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8", newline="\n")

    explanation = f"""# 提示词迭代集内部划分说明

本划分只处理原先 75% 的 `prompt_optimization_train`，不修改原始 JSONL。

- `iteration_optimize_score_keys.txt`：{len(optimize_scores)} 个 score_key，{split_rows['iteration_optimize']} 条；允许查看标注以归纳错误、设计提示词。
- `iteration_train_check_score_keys.txt`：{len(check_scores)} 个 score_key，{split_rows['iteration_check']} 条；只用于评分确认，不把逐题答案反馈给提示词生成。
- `iteration_diagnostic_sample_ids.txt`：第一轮 V9/V27 双基线诊断的 10 条低成本代表样本。
- `iteration_check_sample_ids.txt`：候选通过快速筛选后使用的 16 条内部确认样本。
- `prompt_iteration_subsplit_manifest.json`：完整家族、样本、随机种子和文件哈希。

两部分按 `score_family_id` 隔离；与原先 25% 固定验证集的边界保持不变。原始 test 不参与本划分。
"""
    (DATA_ROOT / "提示词迭代子集划分说明.md").write_text(
        explanation, encoding="utf-8", newline="\n"
    )
    print(json.dumps({
        "optimize_rows": split_rows["iteration_optimize"],
        "check_rows": split_rows["iteration_check"],
        "optimize_families": len(optimize_families),
        "check_families": len(check_families),
        "diagnostic_samples": len(diagnostic_ids),
        "check_samples": len(check_ids),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
