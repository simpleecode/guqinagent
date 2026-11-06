"""Map native GEPA candidate indices to fresh proposal-series names."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def candidate_versions(series: str, rounds: list[dict]) -> dict[int, str]:
    versions = {0: f"{series}0"}
    for entry in rounds:
        version = f"{series}{int(entry['i']) + 1}"
        for index in entry.get("new_program_indices") or []:
            versions[int(index)] = version
    return versions


def export_series(root: Path, series: str, candidates: list[dict], rounds: list[dict], events: list[dict]):
    versions = candidate_versions(series, rounds)
    output = root / "prompts"
    output.mkdir(parents=True, exist_ok=True)
    registry = [{"version": f"{series}0", "candidate_idx": 0, "parent": None,
                 "status": "original_seed", "sha256": hashlib.sha256(candidates[0]['single_stage'].encode()).hexdigest()}]
    (output / f"{series}0.txt").write_text(candidates[0]["single_stage"] + "\n", encoding="utf-8")
    by_round = {int(entry["i"]) + 1: entry for entry in rounds}
    for event in events:
        if event.get("event") != "gepa_proposal":
            continue
        text = event.get("new_instructions", {}).get("single_stage")
        if not text:
            continue
        number = int(event["iteration"])
        version = f"{series}{number}"
        entry = by_round.get(number, {})
        accepted = entry.get("new_program_indices") or []
        (output / f"{version}.txt").write_text(text + "\n", encoding="utf-8")
        registry.append({"version": version, "candidate_idx": accepted[0] if accepted else None,
                         "parent": versions.get(int(entry.get("selected_program_candidate", 0)), f"{series}0"),
                         "status": ("pending_evaluation" if not entry else
                                    "retained_exploration_candidate" if accepted else "rejected_archived"),
                         "sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "minibatch_before": entry.get("subsample_scores"),
                         "minibatch_after": entry.get("new_subsample_scores")})
    (root / "version_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return versions
