"""Select two-stage teacher phrases that actually wrote comma-separated 撮.

Inspect executed edit_plan calls and accepted plans, not reasoning or prompts.
Keep the complete score source for replay but regenerate only selected IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "ABC_J/agent_training/two_stage_audited"
SOURCE = BASE / "pitch_eligible_train_test/inferred_trajectories_train.jsonl"
AUDIT = BASE / "teacher_train_final/teacher_trajectory_audit.jsonl"
OUT = BASE / ".cuo_separator_rebuild"
CUO = re.compile(r"(?:小|大|反)?撮\s*[（(]([^）)]*、[^）)]*)[）)]")
STRING = re.compile(r"[一二三四五六七1-7]弦")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def comma_cuo(text: object) -> bool:
    if not isinstance(text, str):
        return False
    for match in CUO.finditer(text):
        parts = [part.strip() for part in match.group(1).split("、")]
        if len(parts) == 2 and all(len(STRING.findall(part)) == 1 for part in parts):
            return True
    return False


def main() -> None:
    assert comma_cuo("撮（大指九徽三弦、六弦散音）")
    assert not comma_cuo("撮（大指九徽三弦按音＋六弦散音）")
    assert not (OUT / "regenerated_raw").exists(), "Regeneration already started"
    source_ids = {
        row["trajectory_id"] for line in SOURCE.open(encoding="utf-8")
        if (row := json.loads(line))
    }
    selected: dict[str, list[dict]] = defaultdict(list)
    by_stage: dict[str, set[str]] = defaultdict(set)
    for line in AUDIT.open(encoding="utf-8"):
        row = json.loads(line)
        sample_id = row["sample_id"]
        trajectory_id, stage = sample_id.removesuffix("-teacher-tools").rsplit("-", 1)
        assert trajectory_id in source_ids, trajectory_id

        def record(index: object, text: object, location: str) -> None:
            if not comma_cuo(text):
                return
            selected[trajectory_id].append({
                "stage": stage, "index": index, "location": location, "text": text,
            })
            by_stage[stage].add(trajectory_id)

        for action in row["teacher_private"]["accepted_plan"]["actions"]:
            record(action.get("source_index"),
                   action.get("jianzi_text") or action.get("text"), "accepted_plan")
        for log in row["teacher_private"]["tool_execution_log"]:
            if log.get("name") != "edit_plan":
                continue
            entries = (log.get("arguments") or {}).get("jianzi_rows") or []
            if isinstance(entries, dict):
                entries = list(entries.items())
            for entry in entries:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    record(entry[0], entry[1], "edit_plan")
                elif isinstance(entry, dict):
                    for index, text in entry.items():
                        record(index, text, "edit_plan")

    assert selected, "No comma-separated 撮 found"
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "cuo-separator-rebuild-1.0",
        "source_sha256": digest(SOURCE),
        "audit_sha256": digest(AUDIT),
        "selected_phrases": len(selected),
        "by_stage": {stage: len(ids) for stage, ids in sorted(by_stage.items())},
        "selected_ids": sorted(selected),
        "cases": dict(sorted(selected.items())),
    }
    (OUT / "selection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "trajectory_ids.txt").write_text(
        "".join(trajectory_id + "\n" for trajectory_id in sorted(selected)),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "selected_ids"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
