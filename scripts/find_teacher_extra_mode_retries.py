#!/usr/bin/env python3
"""Find accepted teacher plans with exactly one extra sound mode vs annotation.

The comparison is semantic, not a raw string count: accepted ``jianzi_text``
is re-parsed with the same reference parser used by trajectory construction.
Candidates have exactly one extra open/stopped/harmonic mode and no missing
mode, corresponding to one extra 散音／按音／泛音 respectively.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from agents.abc_to_jianzipu.reference_parser import AUDIT, parse_reference_actions


STAGE_SUFFIX = re.compile(r"-(?:fingering_agent|guqinization)-teacher-tools$")
MODE_NAME = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}


def read_jsonl(path: Path):
    for line in path.open(encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def mode_counts(actions: list[dict]) -> Counter[str]:
    return Counter(
        str(action.get("mode")) for action in actions
        if action.get("attack") and action.get("mode") in MODE_NAME
    )


def reparse_accepted(inferred: dict, accepted_actions: list[dict]) -> Counter[str]:
    by_index = {int(action["source_index"]): str(action.get("jianzi_text") or "")
                for action in accepted_actions if action.get("source_index") is not None}
    notes = []
    for source in (inferred.get("input") or {}).get("notes_without_jianzi") or []:
        note = dict(source)
        if note.get("index") is not None:
            note["jianzi"] = by_index.get(int(note["index"]), "")
        notes.append(note)
    data = {"metadata": (inferred.get("input") or {}).get("metadata") or {}, "notes": notes}
    report = AUDIT.audit(data, 50.0)
    return mode_counts([action.to_dict() for action in parse_reference_actions(data, report)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-audit", type=Path, required=True)
    parser.add_argument("--inferred", type=Path, required=True)
    parser.add_argument("--allowed-phrase-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    allowed = {line.strip() for line in args.allowed_phrase_ids.open(encoding="utf-8") if line.strip()}
    inferred = {str(row["trajectory_id"]): row for row in read_jsonl(args.inferred)}
    candidates = []
    for row in read_jsonl(args.teacher_audit):
        sample_id = str(row.get("sample_id") or "")
        phrase_id = STAGE_SUFFIX.sub("", sample_id)
        if phrase_id not in allowed or phrase_id not in inferred:
            continue
        private = row.get("teacher_private") or {}
        accepted = (private.get("accepted_plan") or {}).get("actions") or []
        if not accepted:
            continue
        reference = mode_counts(((inferred[phrase_id].get("reference_plan") or {}).get("actions") or []))
        predicted = reparse_accepted(inferred[phrase_id], accepted)
        extra = predicted - reference
        missing = reference - predicted
        if sum(extra.values()) == 1 and not missing:
            mode = next(iter(extra))
            candidates.append({
                "sample_id": sample_id,
                "phrase_id": phrase_id,
                "stage": str(row.get("sample_id") or "").split("-")[-3],
                "extra_mode": MODE_NAME[mode],
                "reference_modes": dict(reference),
                "accepted_modes": dict(predicted),
            })
    phrase_ids = {item["phrase_id"] for item in candidates}
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "extra_mode_retry_details.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "extra_mode_retry_phrase_ids.txt").write_text(
        "".join(f"{item}\n" for item in sorted(phrase_ids)), encoding="utf-8"
    )
    report = {"candidate_stage_count": len(candidates), "candidate_phrase_count": len(phrase_ids),
              "by_extra_mode": dict(Counter(item["extra_mode"] for item in candidates))}
    (args.output_dir / "extra_mode_retry_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
