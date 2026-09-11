#!/usr/bin/env python3
"""Prepare stage-specific retry ID lists for incomplete teacher generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--trajectory-id-file", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = read_jsonl(args.input)
    wanted = {line.strip() for line in args.trajectory_id_file.read_text(encoding="utf-8").splitlines()
              if line.strip()}
    source_ids = [str(row["trajectory_id"]) for row in source
                  if str(row["trajectory_id"]) in wanted]
    stage_by_id: dict[str, set[str]] = {}
    for row in read_jsonl(args.base_dir / "messages_train.jsonl"):
        sample_id = str(row.get("sample_id") or "")
        for stage in ("fingering_agent", "guqinization"):
            suffix = f"-{stage}-teacher-tools"
            if sample_id.endswith(suffix):
                stage_by_id.setdefault(sample_id[:-len(suffix)], set()).add(stage)
                break

    missing_fingering = [tid for tid in source_ids
                         if "fingering_agent" not in stage_by_id.get(tid, set())]
    missing_guqinization = [tid for tid in source_ids
                            if "guqinization" not in stage_by_id.get(tid, set())
                            and "fingering_agent" in stage_by_id.get(tid, set())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "missing_fingering_ids.txt").write_text(
        "".join(f"{tid}\n" for tid in missing_fingering), encoding="utf-8")
    (args.output_dir / "missing_guqinization_ids.txt").write_text(
        "".join(f"{tid}\n" for tid in missing_guqinization), encoding="utf-8")
    report = {
        "source_phrases": len(source_ids),
        "base_public_rows": sum(len(v) for v in stage_by_id.values()),
        "base_phrase_ids": len(stage_by_id),
        "missing_fingering": len(missing_fingering),
        "missing_guqinization_with_fingering": len(missing_guqinization),
        "missing_total_stage_rows": len(missing_fingering) + len(missing_guqinization),
    }
    (args.output_dir / "missing_stage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
