"""Select accepted Guqinizer phrases flagged by the current missing-attack rule.

The accepted plan, rather than intermediate tool receipts, determines the
selection.  This script never changes the canonical teacher data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.ToolRuntime.runtime import missing_attack_warnings


BASE = ROOT / "ABC_J/agent_training/two_stage_audited"
SOURCE = BASE / "pitch_eligible_train_test/inferred_trajectories_train.jsonl"
AUDIT = BASE / "teacher_train_final/teacher_trajectory_audit.jsonl"
OUT = BASE / ".missing_attack_rebuild"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if (OUT / "regenerated_raw").exists():
        raise SystemExit("Generation has already started; refusing to change its selection")
    source = [json.loads(line) for line in SOURCE.open(encoding="utf-8") if line.strip()]
    plans = {}
    for line in AUDIT.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["sample_id"].endswith("-guqinization-teacher-tools"):
            phrase_id = row["sample_id"].removesuffix("-guqinization-teacher-tools")
            if phrase_id in plans:
                raise ValueError(f"Duplicate Guqinizer audit: {phrase_id}")
            plans[phrase_id] = row["teacher_private"]["accepted_plan"]
    source_ids = {row["trajectory_id"] for row in source}
    if len(source_ids) != len(source) or source_ids != set(plans):
        raise ValueError("Source and final Guqinizer IDs are not identical")

    cases = {}
    counts = Counter()
    for item in source:
        phrase_id = item["trajectory_id"]
        actions = {
            int(action["source_index"]): action
            for action in plans[phrase_id]["actions"]
        }
        warnings = missing_attack_warnings(item, actions)
        if warnings:
            cases[phrase_id] = warnings
            counts["events"] += len(warnings)
    ids = sorted(cases)
    counts["phrases"] = len(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "missing-attack-rebuild-1.0",
        "source_sha256": sha256(SOURCE),
        "audit_sha256": sha256(AUDIT),
        "selected_phrases": len(ids),
        "warning_events": counts["events"],
        "selected_ids": ids,
        "cases": cases,
    }
    (OUT / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "trajectory_ids.txt").write_text(
        "".join(phrase_id + "\n" for phrase_id in ids), encoding="utf-8"
    )
    print(json.dumps({"phrases": len(ids), "events": counts["events"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
