"""Stratified pilot sampling over inferred train phrases.

Selects a deterministic, coverage-oriented subset of train phrases for the
teacher-trajectory pilot batch. Stratification dimensions mirror the pilot
questions in DOCS/agent_training_handoff.md section 9:

- context role: score-first / section-first / continuation
- attack-note length and technique density: tercile bins, spread by
  least-filled subcell so short/long and sparse/dense phrases are all
  represented
- diversity constraints: at most 2 phrases per score, prefer untouched
  score families and rarely used tunings

Only phrases that would actually be processed by
generate_teacher_tool_trajectories.py --basic-intermediate are considered
(non-empty blank plan), so no sampled ID can be silently skipped.

Outputs (into --output-dir):
- pilot_phrases.jsonl : one feature record per selected phrase
- pilot_ids.txt       : one trajectory_id per line, for --trajectory-id
- pilot_report.json   : quotas, coverage counts and filter statistics
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTEXT_ROLE_QUOTAS = {"score_first": 6, "section_first": 10, "continuation": 24}
MAX_PER_SCORE = 2


def phrase_features(row: dict) -> dict | None:
    """Extract stratification features; None if the row must be excluded."""
    reference = row.get("reference_plan", {}).get("actions", [])
    attack_actions = [a for a in reference if a.get("attack")]
    n_attack = len(attack_actions)
    if n_attack == 0:
        attack_actions = [a for a in row["baseline_plan"]["actions"] if a.get("attack")]
        n_attack = len(attack_actions)
        if n_attack == 0:
            return None
    technique_count = sum(len(a.get("techniques") or []) for a in attack_actions)
    handoff = row["input"].get("phrase_handoff", {})
    section = handoff.get("section", {}) or {}
    if row["phrase_id"] == "p0001":
        role = "score_first"
    elif section.get("boundary_from_previous"):
        role = "section_first"
    else:
        role = "continuation"
    tuning = row["input"].get("normalized_tuning", {})
    return {
        "trajectory_id": row["trajectory_id"],
        "score_key": row["score_key"],
        "score_family_id": row["score_family_id"],
        "phrase_id": row["phrase_id"],
        "score_title": row["input"]["metadata"]["score_title"],
        "context_role": role,
        "n_attack": n_attack,
        "n_events": len(row["input"].get("notes_without_jianzi", [])),
        "n_baseline_actions": len(row["baseline_plan"].get("actions", [])),
        "technique_count": technique_count,
        "technique_density": round(technique_count / n_attack * 100, 2),
        "tuning_name": tuning.get("name"),
        "tuning_fingerprint": tuning.get("fingerprint"),
    }


def tercile_bins(features: list[dict], key: str) -> tuple[list[int], list[str]]:
    """Return bin index per row and bin edge labels for a numeric feature."""
    values = sorted(f[key] for f in features)
    q1 = values[len(values) // 3]
    q2 = values[2 * len(values) // 3]

    def bin_of(value: int | float) -> str:
        if value <= q1:
            return "low"
        if value <= q2:
            return "mid"
        return "high"

    # Bins are computed over the full pool so that pilot selections stay
    # comparable across runs and seeds.
    labels = [bin_of(f[key]) for f in features]
    edges = [f"<= {q1}", f"<= {q2}", f"> {q2}"]
    return labels, edges


def select_pilot(features: list[dict], bins: dict[str, list[str]], size: int,
                 rng: random.Random) -> tuple[list[dict], dict]:
    """Quota-based greedy selection with diversity constraints."""
    by_cell = defaultdict(list)
    for feature, length_bin, density_bin in zip(
            features, bins["n_attack"], bins["technique_density"]):
        entry = dict(feature, length_bin=length_bin, density_bin=density_bin)
        by_cell[entry["context_role"]].append(entry)
    for entries in by_cell.values():
        rng.shuffle(entries)

    total_quota = sum(CONTEXT_ROLE_QUOTAS.values())
    if size != total_quota:
        scale = size / total_quota
        quotas = {role: max(1, round(q * scale)) for role, q in CONTEXT_ROLE_QUOTAS.items()}
    else:
        quotas = dict(CONTEXT_ROLE_QUOTAS)

    selected: list[dict] = []
    used_scores: Counter = Counter()
    used_families: Counter = Counter()
    used_tunings: Counter = Counter()
    shortfall: dict[str, int] = {}

    def take_entry(entry: dict) -> None:
        selected.append(entry)
        used_scores[entry["score_key"]] += 1
        used_families[entry["score_family_id"]] += 1
        used_tunings[entry["tuning_name"]] += 1

    def pick_from(entries: list[dict], families_locked: bool) -> dict | None:
        # Least-used tuning first, then untouched family, then order.
        def rank(entry: dict) -> tuple | None:
            family_ok = used_families[entry["score_family_id"]] == 0
            if families_locked and not family_ok:
                return None
            if used_scores[entry["score_key"]] >= MAX_PER_SCORE:
                return None
            return (used_tunings[entry["tuning_name"]],
                    0 if family_ok else 1)

        best, best_key = None, None
        for entry in entries:
            key = rank(entry)
            if key is not None and (best_key is None or key < best_key):
                best, best_key = entry, key
        return best

    def fill_cell(role: str, entries: list[dict], quota: int) -> None:
        """Fill one context-role cell, always drawing next from the
        least-filled length/density subcell so bin coverage stays balanced."""
        if not entries:
            shortfall[role] = quota
            return
        grouped = defaultdict(list)
        for entry in entries:
            grouped[f"{entry['length_bin']}/{entry['density_bin']}"].append(entry)
        fill: Counter = Counter()
        taken = 0
        for families_locked in (True, False):
            available = [key for key, subcell in grouped.items() if subcell]
            while taken < quota and available:
                min_fill = min(fill[key] for key in available)
                tied = sorted(key for key in available if fill[key] == min_fill)
                key = rng.choice(tied)
                candidate = pick_from(grouped[key], families_locked)
                if candidate is None:
                    available.remove(key)
                    continue
                grouped[key].remove(candidate)
                fill[key] += 1
                take_entry(candidate)
                taken += 1
        if taken < quota:
            shortfall[role] = quota - taken

    for role, quota in sorted(quotas.items()):
        fill_cell(role, by_cell.get(role, []), quota)

    # Redistribute shortfall to continuation, by far the largest pool.
    for role, missing in sorted(shortfall.items()):
        if role == "continuation":
            continue
        fill_cell("continuation", by_cell.get("continuation", []), missing)

    selected.sort(key=lambda e: e["trajectory_id"])
    stats = {
        "role_quota": dict(quotas),
        "shortfall_before_redistribution": dict(sorted(shortfall.items())),
        "scores_used": len(used_scores),
        "families_used": len(used_families),
        "max_phrases_per_score": max(used_scores.values()) if used_scores else 0,
        "tuning_usage": dict(used_tunings),
    }
    return selected, stats


def coverage(selected: list[dict]) -> dict:
    return {
        "n_selected": len(selected),
        "context_roles": dict(Counter(e["context_role"] for e in selected)),
        "length_bins": dict(Counter(e["length_bin"] for e in selected)),
        "density_bins": dict(Counter(e["density_bin"] for e in selected)),
        "tuning_names": dict(Counter(e["tuning_name"] for e in selected)),
        "distinct_scores": len({e["score_key"] for e in selected}),
        "distinct_families": len({e["score_family_id"] for e in selected}),
        "attack_notes_range": [min(e["n_attack"] for e in selected),
                               max(e["n_attack"] for e in selected)] if selected else [],
        "technique_density_range": [min(e["technique_density"] for e in selected),
                                    max(e["technique_density"] for e in selected)] if selected else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "ABC_J/agent_training/inferred_v2/inferred_trajectories_train.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "ABC_J/agent_training/pilot_sampling")
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    features, excluded = [], Counter()
    for row in rows:
        feature = phrase_features(row)
        if feature is None:
            excluded["no_attack_or_baseline"] += 1
        else:
            features.append(feature)
    if not features:
        raise SystemExit("no usable phrases after filters")

    length_bins, length_edges = tercile_bins(features, "n_attack")
    density_bins, density_edges = tercile_bins(features, "technique_density")
    bins = {"n_attack": length_bins, "technique_density": density_bins}

    rng = random.Random(args.seed)
    selected, selection_stats = select_pilot(features, bins, args.size, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.output_dir.joinpath("pilot_phrases.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for entry in selected:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    args.output_dir.joinpath("pilot_ids.txt").write_text(
        "\n".join(e["trajectory_id"] for e in selected) + "\n", encoding="utf-8")

    report = {
        "input": str(args.input),
        "seed": args.seed,
        "pool": {"total_rows": len(rows), "usable": len(features),
                 "excluded": dict(excluded)},
        "bin_edges": {"n_attack": length_edges, "technique_density": density_edges},
        "selection": selection_stats,
        "coverage": coverage(selected),
        "ids": [e["trajectory_id"] for e in selected],
    }
    args.output_dir.joinpath("pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cov = report["coverage"]
    print(f"selected {cov['n_selected']} phrases from {len(features)} usable")
    print(f"scores: {cov['distinct_scores']}  families: {cov['distinct_families']}  "
          f"max/score: {selection_stats['max_phrases_per_score']}")
    print("context roles:", cov["context_roles"])
    print("length bins:", cov["length_bins"], " edges:", length_edges)
    print("density bins:", cov["density_bins"], " edges:", density_edges)
    print("tunings:", cov["tuning_names"])
    print(f"attack notes range: {cov['attack_notes_range']}  "
          f"technique density range: {cov['technique_density_range']}")
    print(f"outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
