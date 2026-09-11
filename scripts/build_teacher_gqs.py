#!/usr/bin/env python3
"""Materialize one lossless, whole-score teacher.gqs per dataset score."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.teacher_gqs import FIELDS, parse_teacher_gqs, render_teacher_gqs
from agents.abc_to_jianzipu.compound_gestures import find_compound_gesture


def assign_jianzi_text_targets(notes: list[dict]) -> None:
    """Keep null for unknown surfaces and empty text for compound-covered rows."""
    for note in notes:
        raw = str(note.get("jianzi") or "").strip()
        note["jianzi_text"] = raw or None
    for position, note in enumerate(notes):
        gesture = find_compound_gesture(note.get("jianzi"))
        if not gesture or "声" not in gesture:
            continue
        for following in notes[position + 1:]:
            if str(following.get("jianzi") or "").strip():
                break
            jianpu = str(following.get("jianpu") or "")
            abc = str(following.get("abc") or "")
            if (jianpu in {"|", "0（休止）", "－（延音）"}
                    or abc.startswith(("|", "z", "-"))
                    or (following.get("section") or {}).get("start")):
                break
            following["jianzi_text"] = ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "ABC_J/results/dataset_split_groups.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "ABC_J/agent_training/gqs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    written = skipped = 0
    index = []
    for row in rows:
        source = Path(row["final_data_path"]) / "jianpu_jianzi_readable.json"
        if not source.exists():
            skipped += 1
            continue
        data = json.loads(source.read_text(encoding="utf-8"))
        assign_jianzi_text_targets(data.get("notes") or [])
        text = render_teacher_gqs(data)
        # A mandatory round-trip check makes GQS a real canonical intermediate.
        normalized = {
            "metadata": data.get("metadata") or {},
            "notes": [
                {field: note.get(field) for field in FIELDS}
                for note in data.get("notes") or []
            ],
        }
        if parse_teacher_gqs(text) != normalized:
            raise ValueError(f"GQS round-trip mismatch: {row['score_key']}")
        target_dir = args.output_dir / row["score_key"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "teacher.gqs"
        target.write_text(text, encoding="utf-8", newline="\n")
        index.append({"score_key": row["score_key"], "path": str(target),
                      "notes": len(data.get("notes") or [])})
        written += 1
    report = {"schema_version": "teacher-gqs-build-1.0", "written": written,
              "skipped": skipped, "scores": index}
    (args.output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": written, "skipped": skipped}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
