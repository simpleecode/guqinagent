"""Materialize fixed, family-disjoint feedback/Pareto/confirmation pilot data."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from baseline.gepa_guqin.compat import materialize_actions
from baseline.scripts.prompt_data_sources import TRAIN


def prepare_pilot(manifest_path: Path, run_dir: Path, identity: dict):
    plan = json.loads(manifest_path.read_text(encoding="utf-8"))
    if plan["source_sha256"] != identity["source_sha256"]:
        raise SystemExit("Pilot plan source SHA-256 does not match current training data")
    with TRAIN.open(encoding="utf-8") as handle:
        sources = {r["trajectory_id"]: r for r in (json.loads(line) for line in handle if line.strip())}
    roles = {}
    histories = {}
    families_by_role = {}
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for role in ("feedback", "pareto", "confirmation"):
        ids = plan["splits"][role]["sample_ids"]
        if len(ids) != len(set(ids)) or any(sid not in sources for sid in ids):
            raise SystemExit(f"Missing/duplicate {role} sample IDs")
        selected = [sources[sid] for sid in ids]
        families_by_role[role] = {r.get("score_family_id") or r["score_key"] for r in selected}
        examples, history = [], {}
        for row in selected:
            clean = deepcopy(row)
            clean["input"].setdefault("phrase_handoff", {}).pop("previous_phrase", None)
            clean["reference_plan"] = {"actions": materialize_actions(row["reference_plan"]["actions"])}
            history[(str(row["score_key"]), str(row["phrase_id"]))] = clean
            examples.append({
                "sample_id": row["trajectory_id"], "score_key": row["score_key"],
                "phrase_id": row["phrase_id"], "split": role,
                "runtime_item": {"input": deepcopy(clean["input"]), "baseline_plan": deepcopy(row["baseline_plan"])},
                "reference": {"actions": clean["reference_plan"]["actions"]},
            })
        by_score = defaultdict(list)
        for row in selected:
            by_score[row["score_key"]].append(row["phrase_id"])
        for key, phrases in by_score.items():
            if sorted(phrases) != [f"p{i:04d}" for i in range(1, len(phrases) + 1)]:
                raise SystemExit(f"Pilot {key} does not have a complete contiguous prefix")
        roles[role], histories[role] = examples, history
    names = list(families_by_role)
    for i, role in enumerate(names):
        for other in names[i + 1:]:
            if families_by_role[role] & families_by_role[other]:
                raise SystemExit("Pilot families overlap across data roles")
    for role, examples in roles.items():
        with (data_dir / f"examples_{role}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    historical = {**histories["feedback"], **histories["pareto"]}
    with (data_dir / "historical.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in historical.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split = {
        "mode": "family_isolated_multiscore", "source": str(TRAIN),
        "warning": ("按曲谱家族隔离 feedback、pareto 和 confirmation；各曲使用连续导出。"
                    if plan.get("prefix_phrases") == 0 else
                    "初步跨曲谱试验，每首只取固定前缀；不代表全曲或最终test成绩。"),
        "score_key": f"{plan['score_count']}首曲谱",
        "pilot_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "feedback_sample_ids": plan["splits"]["feedback"]["sample_ids"],
        "pareto_sample_ids": plan["splits"]["pareto"]["sample_ids"],
        "confirmation_sample_ids": plan["splits"]["confirmation"]["sample_ids"],
        "families": {role: sorted(families) for role, families in families_by_role.items()},
        "pilot": plan["splits"],
        "test_hashes": plan["test_hashes"],
    }
    (data_dir / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return roles["feedback"], roles["pareto"], historical, split
