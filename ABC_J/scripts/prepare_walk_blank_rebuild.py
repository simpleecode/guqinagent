"""Select source phrases with a changed-pitch blank after standalone 绰上/注下.

Selection is based on the extracted reference and does not rewrite it. The
teacher must decide whether each blank is intentional coverage or missing data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.ToolRuntime.runtime import illegal_blank_after_walk_warnings

BASE = ROOT / "ABC_J/agent_training/two_stage_audited"
SOURCE = BASE / "pitch_eligible_train_test/inferred_trajectories_train.jsonl"
OUT = BASE / ".walk_blank_rebuild"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if (OUT / "regenerated_raw").exists():
        raise SystemExit("Generation already started; refusing to change selection")
    cases: dict[str, list[dict]] = {}
    source_ids: set[str] = set()
    for line in SOURCE.open(encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        phrase_id = str(item["trajectory_id"])
        if phrase_id in source_ids:
            raise ValueError(f"Duplicate source ID: {phrase_id}")
        source_ids.add(phrase_id)
        actions = {
            int(action["source_index"]): {
                "jianzi_text": action.get("jianzi_text")
                if action.get("jianzi_text") is not None
                else action.get("text") or ""
            }
            for action in item["reference_plan"]["actions"]
        }
        warnings = illegal_blank_after_walk_warnings(item, actions)
        if warnings:
            references = {
                int(action["source_index"]): action
                for action in item["reference_plan"]["actions"]
            }
            for warning in warnings:
                reference = references[warning["source_index"]]
                if (reference.get("confidence_class") != "unusable"
                        or "empty_reference_for_sounding_event" not in
                        reference.get("diagnostics", [])):
                    raise ValueError(f"Unexpected reference class: {phrase_id} {warning}")
            cases[phrase_id] = warnings
    ids = sorted(cases)
    report = {
        "schema_version": "walk-blank-rebuild-1.0",
        "source_sha256": sha256(SOURCE),
        "source_phrases": len(source_ids),
        "selected_phrases": len(ids),
        "warning_events": sum(map(len, cases.values())),
        "selected_ids": ids,
        "cases": cases,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "trajectory_ids.txt").write_text(
        "".join(phrase_id + "\n" for phrase_id in ids), encoding="utf-8"
    )
    print(json.dumps({"phrases": len(ids), "events": report["warning_events"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
