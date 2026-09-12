#!/usr/bin/env python3
"""Find teacher trajectories whose accepted jianzi puts string before hui/right hand."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


STAGE_MARKERS = ("fingering_agent", "guqinization")
BAD_ORDER = re.compile(
    r"(?:大指|名指|中指|食指)?[一二三四五六七1-7]弦"
    r"[^，。；｜]*?(?:[一二三四五六七八九十百0-9]+徽|徽)"
    r"[^，。；｜]*?(?:抹|挑|勾|剔|擘|托|打|摘)"
)


def stage_and_trajectory(sample_id: str) -> tuple[str, str]:
    for stage in STAGE_MARKERS:
        marker = f"-{stage}-"
        if marker in sample_id:
            return stage, sample_id.split(marker, 1)[0]
    raise ValueError(f"unknown sample id: {sample_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    hits: list[dict] = []
    bad_by_stage = {stage: set() for stage in STAGE_MARKERS}
    with args.audit.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            stage, trajectory_id = stage_and_trajectory(sample_id)
            actions = ((row.get("teacher_private") or {}).get("accepted_plan") or {}).get("actions") or []
            for action in actions:
                text = str(action.get("jianzi_text") or "")
                if not BAD_ORDER.search(text):
                    continue
                bad_by_stage[stage].add(trajectory_id)
                hits.append({
                    "trajectory_id": trajectory_id,
                    "stage": stage,
                    "source_index": action.get("source_index"),
                    "jianzi_text": text,
                })

    rerun_both = bad_by_stage["fingering_agent"]
    rerun_guqin_only = bad_by_stage["guqinization"] - rerun_both
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rerun_both_stages.txt").write_text(
        "".join(f"{value}\n" for value in sorted(rerun_both)), encoding="utf-8"
    )
    (args.output_dir / "rerun_guqinizer_only.txt").write_text(
        "".join(f"{value}\n" for value in sorted(rerun_guqin_only)), encoding="utf-8"
    )
    with (args.output_dir / "hits.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for hit in hits:
            out.write(json.dumps(hit, ensure_ascii=False) + "\n")
    report = {
        "matched_action_count": len(hits),
        "matched_actions_by_stage": dict(Counter(hit["stage"] for hit in hits)),
        "affected_fingering_phrases": len(rerun_both),
        "affected_guqinizer_phrases": len(bad_by_stage["guqinization"]),
        "rerun_both_stages": len(rerun_both),
        "rerun_guqinizer_only": len(rerun_guqin_only),
        "total_affected_phrases": len(rerun_both | bad_by_stage["guqinization"]),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
