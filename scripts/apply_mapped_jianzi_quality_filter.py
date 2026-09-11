#!/usr/bin/env python3
"""Build a non-destructive, quality-filtered GQS source set.

The audit is based on mapped markdown files, while inference consumes GQS.
This script therefore copies the complete GQS tree, omits score-level drops,
and removes only trailing audio rows for the selected trim candidates.
Original sources and existing trajectories are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.teacher_gqs import parse_teacher_gqs, render_teacher_gqs


DEFAULT_AUDIT = ROOT / "ABC_J/agent_training/source_quality_audit_v1/quality_report.json"
DEFAULT_SOURCE = ROOT / "ABC_J/agent_training/gqs"
DEFAULT_OUTPUT = ROOT / "ABC_J/agent_training/gqs_quality_filtered_v1"
DEFAULT_REPORT_DIR = ROOT / "ABC_J/agent_training/source_quality_audit_v1"
GQS_AUDIO_RE = re.compile(r"^音｜(.+)$")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def filter_gqs(path: Path, output: Path, cutoff: int) -> int:
    """Copy one GQS file, omitting audio rows whose source index is past cutoff."""
    text = path.read_text(encoding="utf-8-sig")
    if text.startswith("GQS｜teacher-gqs-1.2"):
        data = parse_teacher_gqs(text)
        original = data.get("notes") or []
        data["notes"] = [note for note in original
                          if note.get("index") is None or int(note["index"]) <= cutoff]
        output.write_text(render_teacher_gqs(data), encoding="utf-8", newline="\n")
        return len(original) - len(data["notes"])
    removed = 0
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    for line in lines:
        match = GQS_AUDIO_RE.match(line.rstrip("\r\n"))
        if not match:
            kept.append(line)
            continue
        try:
            row = json.loads(match.group(1))
            index = int(row[0])
        except (ValueError, TypeError, IndexError, json.JSONDecodeError):
            kept.append(line)
            continue
        if index > cutoff:
            removed += 1
        else:
            kept.append(line)
    output.write_text("".join(kept), encoding="utf-8")
    return removed


def copy_tree(source: Path, output: Path, drops: set[str], cutoffs: dict[str, int]) -> dict:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}; choose another path or remove it explicitly")
    output.mkdir(parents=True)
    copied_scores = 0
    omitted_scores: list[str] = []
    removed_rows: dict[str, int] = {}
    for score_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        key = score_dir.name
        if key in drops:
            omitted_scores.append(key)
            continue
        target_dir = output / key
        shutil.copytree(score_dir, target_dir)
        copied_scores += 1
        if key in cutoffs:
            gqs = target_dir / "teacher.gqs"
            if not gqs.exists():
                raise FileNotFoundError(f"trim candidate has no teacher.gqs: {gqs}")
            removed_rows[key] = filter_gqs(gqs, gqs, cutoffs[key])
    return {
        "source_score_dirs": len([p for p in source.iterdir() if p.is_dir()]),
        "copied_score_dirs": copied_scores,
        "omitted_score_keys": omitted_scores,
        "trimmed_gqs_rows": removed_rows,
    }


def filter_manifest(source: Path, output: Path, drops: set[str]) -> int:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader if row.get("score_key") not in drops]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--source-gqs", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-gqs", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--dataset-manifest", type=Path, default=ROOT / "ABC_J/results/dataset_split_groups.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "ABC_J/results/agent_dataset_splits.csv")
    args = parser.parse_args()

    audit = read_json(args.audit)
    records = audit["records"]
    drops = {r["score_key"] for r in records if r["classification"] == "drop_score"}
    cutoffs = {
        r["score_key"]: int(r["last_nonempty_index"])
        for r in records
        if r["classification"] == "trim_trailing" and r["last_nonempty_index"] is not None
    }
    copy_summary = copy_tree(args.source_gqs, args.output_gqs, drops, cutoffs)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    filtered_dataset = args.report_dir / "dataset_split_groups_filtered.csv"
    filtered_split = args.report_dir / "agent_dataset_splits_filtered.csv"
    dataset_rows = filter_manifest(args.dataset_manifest, filtered_dataset, drops)
    split_rows = filter_manifest(args.split_manifest, filtered_split, drops)
    report = {
        "audit": str(args.audit),
        "source_gqs": str(args.source_gqs),
        "output_gqs": str(args.output_gqs),
        "drop_score_keys": sorted(drops),
        "trim_cutoffs": cutoffs,
        "copy": copy_summary,
        "filtered_manifests": {
            "dataset_split_groups": {"path": str(filtered_dataset), "rows": dataset_rows},
            "agent_dataset_splits": {"path": str(filtered_split), "rows": split_rows},
        },
    }
    report_path = args.report_dir / "filter_application_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **copy_summary, "dataset_rows": dataset_rows, "split_rows": split_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
