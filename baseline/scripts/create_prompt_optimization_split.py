#!/usr/bin/env python3
"""Create an answer-leakage-safe 75/25 score-family split for prompt work.

The source training JSONL is never rewritten.  This script only writes a
manifest, two score-key lists, and a Chinese README beside the source dataset.
All score keys belonging to one score_family_id stay in the same split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "ABC_J/agent_training/two_stage_audited/pitch_eligible_train_test"
SEED = "prompt-validation-family-split-20261026-v1"


def hash_rank(family: str) -> str:
    return hashlib.sha256(f"{SEED}:{family}".encode()).hexdigest()


def choose_validation(families: dict[str, dict], target: int) -> set[str]:
    """Subset-sum family groups to the nearest possible target row count."""
    ordered = sorted(families, key=hash_rank)
    maximum = target + max(item["rows"] for item in families.values())
    # For each row count retain the deterministic lexicographically earliest
    # family-rank sequence.  This keeps output repeatable without using labels.
    states: dict[int, tuple[str, ...]] = {0: ()}
    for family in ordered:
        size = families[family]["rows"]
        rank = hash_rank(family)
        for total, chosen in list(states.items()):
            candidate_total = total + size
            if candidate_total > maximum:
                continue
            candidate = tuple(sorted(chosen + (family,), key=hash_rank))
            previous = states.get(candidate_total)
            if previous is None or tuple(map(hash_rank, candidate)) < tuple(map(hash_rank, previous)):
                states[candidate_total] = candidate
    best_total = min(states, key=lambda total: (abs(total - target), total > target, tuple(map(hash_rank, states[total]))))
    return set(states[best_total])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    args = parser.parse_args()
    if (args.dataset_dir / "new_dataset_gepa_split_manifest.json").exists():
        raise SystemExit("新版划分已冻结；请运行 create_new_dataset_gepa_split.py --check 校验，不要用旧划分器覆盖。")
    if not 0 < args.validation_ratio < 1:
        raise SystemExit("--validation-ratio must be between 0 and 1")
    source = args.dataset_dir / "inferred_trajectories_train.jsonl"
    families: dict[str, dict] = defaultdict(lambda: {"rows": 0, "score_keys": set()})
    digest = hashlib.sha256()
    total_rows = 0
    with source.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            row = json.loads(raw.decode("utf-8"))
            family = str(row.get("score_family_id") or row["score_key"])
            families[family]["rows"] += 1
            families[family]["score_keys"].add(str(row["score_key"]))
            total_rows += 1
    target = round(total_rows * args.validation_ratio)
    validation_families = choose_validation(families, target)
    validation_keys = sorted(key for family in validation_families for key in families[family]["score_keys"])
    train_keys = sorted(key for family in families if family not in validation_families for key in families[family]["score_keys"])
    validation_rows = sum(families[family]["rows"] for family in validation_families)
    manifest = {
        "schema_version": "prompt-optimization-family-split-1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source.name,
        "source_sha256": digest.hexdigest(),
        "seed": SEED,
        "grouping": "score_family_id; rows without it fall back to score_key",
        "target_validation_ratio": args.validation_ratio,
        "splits": {
            "fixed_validation": {"families": sorted(validation_families), "score_keys": validation_keys,
                                 "family_count": len(validation_families), "score_key_count": len(validation_keys),
                                 "rows": validation_rows, "ratio": validation_rows / total_rows},
            "prompt_optimization_train": {"families": sorted(set(families) - validation_families), "score_keys": train_keys,
                                            "family_count": len(families) - len(validation_families), "score_key_count": len(train_keys),
                                            "rows": total_rows - validation_rows, "ratio": (total_rows - validation_rows) / total_rows},
        },
    }
    args.dataset_dir.joinpath("fixed_validation_score_keys_25pct.txt").write_text("\n".join(validation_keys) + "\n", encoding="utf-8")
    args.dataset_dir.joinpath("prompt_optimization_train_score_keys_75pct.txt").write_text("\n".join(train_keys) + "\n", encoding="utf-8")
    args.dataset_dir.joinpath("prompt_optimization_family_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.dataset_dir.joinpath("提示词训练数据划分说明.md").write_text(
        f"""# 提示词训练数据划分说明

本说明只划分 `inferred_trajectories_train.jsonl`；原始 JSONL 文件没有被修改。

| 部分 | 文件 | 用途 | 数量 |
|---|---|---|---:|
| 固定验证集（约 25%） | `fixed_validation_score_keys_25pct.txt` | 原版、V5–V32 及后续候选必须在相同曲目上重新生成并统一评分；可用于版本选择，但不能称为最终 test。 | {validation_rows} 条短语／{len(validation_keys)} 首曲／{len(validation_families)} 个曲族（{validation_rows / total_rows:.2%}） |
| 训练／迭代集（约 75%） | `prompt_optimization_train_score_keys_75pct.txt` | 找错误、生成提示词、收集训练轨迹。 | {total_rows - validation_rows} 条短语／{len(train_keys)} 首曲／{len(families) - len(validation_families)} 个曲族（{(total_rows - validation_rows) / total_rows:.2%}） |

## 划分规则

- 按 `score_family_id` 整组划分；没有家族 ID 时才按 `score_key`。同源曲不会跨训练集与验证集，避免泄漏。
- 固定随机种子：`{SEED}`；源文件 SHA-256：`{digest.hexdigest()}`。
- 后续迭代不得改变这两份列表。候选与基线必须在固定验证集使用完全相同的模型、温度、最大轮数、重试次数和师兄评分器。
- 原始 `inferred_trajectories_test.jsonl`（1,379 条／43 首曲）保持为密封最终测试集；最终评测后若还要继续改提示词，应重新划分一套未查看的最终测试集。
""", encoding="utf-8")
    print(json.dumps({"validation_rows": validation_rows, "validation_score_keys": len(validation_keys),
                      "train_rows": total_rows - validation_rows, "train_score_keys": len(train_keys),
                      "validation_families": len(validation_families)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
