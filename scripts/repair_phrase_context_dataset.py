#!/usr/bin/env python3
"""Audit and repair accepted trajectories generated without prior-phrase context.

This utility deliberately stages regenerated records separately.  It never
modifies a batch in place: ``audit`` writes an explicit repair manifest,
``prepare-intermediates`` builds the Guqinizer baseline for that manifest, and
``merge`` makes a new, atomically validated batch directory.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


STAGES = ("fingering_agent", "guqinization")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sample_parts(sample_id: str) -> tuple[str, str] | None:
    suffix = "-teacher-tools"
    if not sample_id.endswith(suffix):
        return None
    stem = sample_id[:-len(suffix)]
    for stage in STAGES:
        marker = "-" + stage
        if stem.endswith(marker):
            return stem[:-len(marker)], stage
    return None


def prompt_has_previous(row: dict) -> bool:
    messages = row.get("messages") or []
    first_user = next((message for message in messages if message.get("role") == "user"), {})
    # The rendered heading contains its phrase identifier inside the brackets,
    # e.g. ``【只读前一段 p0051｜confirmed_readonly】``.
    return "【只读前一段 " in str(first_user.get("content") or "")


def source_has_previous(row: dict) -> bool:
    return bool((row.get("input") or {}).get("phrase_handoff", {}).get("previous_phrase"))


def command_audit(args: argparse.Namespace) -> None:
    source = {row["trajectory_id"]: row for row in read_rows(args.input)}
    public = read_rows(args.raw_dir / "messages_train.jsonl")
    affected: dict[str, list[str]] = {stage: [] for stage in STAGES}
    details: list[dict] = []
    for row in public:
        parsed = sample_parts(str(row.get("sample_id") or ""))
        if not parsed:
            continue
        trajectory_id, stage = parsed
        item = source.get(trajectory_id)
        if item and source_has_previous(item) and not prompt_has_previous(row):
            affected[stage].append(trajectory_id)
            details.append({"sample_id": row["sample_id"], "trajectory_id": trajectory_id, "stage": stage})
    affected = {stage: sorted(set(ids)) for stage, ids in affected.items()}
    both = sorted(set(affected["fingering_agent"]) & set(affected["guqinization"]))
    manifest = {
        "schema_version": "phrase-context-repair-1.0",
        "input": str(args.input), "raw_dir": str(args.raw_dir),
        "affected_by_stage": affected,
        "rerun_fingering": affected["fingering_agent"],
        "rerun_guqinization": affected["guqinization"],
        "guqinization_after_fresh_fingering": both,
        "accepted_sample_ids": sorted(item["sample_id"] for item in details),
        "counts": {"fingering_agent": len(affected["fingering_agent"]),
                   "guqinization": len(affected["guqinization"]),
                   "trajectories": len(set().union(*map(set, affected.values()))),
                   "samples": len(details)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "repair_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, ids in (("fingering_ids.txt", affected["fingering_agent"]),
                      ("guqinization_ids.txt", affected["guqinization"]),
                      ("guqinization_after_fresh_fingering_ids.txt", both)):
        (args.output_dir / name).write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


def command_prepare_intermediates(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fresh_ids = set(manifest["guqinization_after_fresh_fingering"])
    wanted = set(manifest["rerun_guqinization"])
    original = {row["trajectory_id"]: row for row in read_rows(args.original_intermediates)}
    fresh = {row["trajectory_id"]: row for row in read_rows(args.fresh_intermediates)}
    missing = sorted((wanted - fresh_ids - set(original)) | (fresh_ids - set(fresh)))
    if missing:
        raise SystemExit("missing required fingering intermediate: " + ", ".join(missing))
    rows = [(fresh if trajectory_id in fresh_ids else original)[trajectory_id]
            for trajectory_id in sorted(wanted)]
    write_rows(args.output, rows)
    print(json.dumps({"intermediates": len(rows), "fresh": len(fresh_ids)}, ensure_ascii=False))


def command_merge(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    affected = set(manifest["accepted_sample_ids"])
    base = read_rows(args.base / args.file_name)
    replacement = read_rows(args.replacement / args.file_name)
    replacement_by_id = {row["sample_id"]: row for row in replacement}
    excluded: set[str] = set()
    for path in args.exclude_sample_id_file or []:
        excluded.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    replacement_by_id = {sample_id: row for sample_id, row in replacement_by_id.items()
                         if sample_id not in excluded}
    missing = sorted(affected - set(replacement_by_id))
    if missing and not args.drop_unreplaced:
        raise SystemExit("replacement lacks accepted sample IDs: " + ", ".join(missing))
    base_ids = {row["sample_id"] for row in base}
    new_ids = set(replacement_by_id) - base_ids
    if new_ids and not args.insert_missing_after_sample_id:
        raise SystemExit("replacement contains sample IDs absent from base; specify insertion point: "
                         + ", ".join(sorted(new_ids)))
    new_rows = [replacement_by_id[sample_id] for sample_id in replacement_by_id
                if sample_id in new_ids]
    result: list[dict] = []
    inserted = False
    for row in base:
        if row["sample_id"] not in set(missing):
            result.append(replacement_by_id.get(row["sample_id"], row))
        if row["sample_id"] == args.insert_missing_after_sample_id:
            result.extend(new_rows)
            inserted = True
    if new_rows and not inserted:
        raise SystemExit("requested insertion sample ID is not in base: "
                         + str(args.insert_missing_after_sample_id))
    unexpected = set(replacement_by_id) - affected
    if unexpected:
        raise SystemExit("replacement includes unexpected IDs: " + ", ".join(sorted(unexpected)))
    expected_rows = len(base) - len(missing) + len(new_rows)
    if len(result) != expected_rows or len({row["sample_id"] for row in result}) != len(result):
        raise SystemExit("merge would create duplicate sample IDs or an unexpected row count")
    write_rows(args.output / args.file_name, result)
    print(json.dumps({"replaced": len(affected) - len(missing), "inserted": len(new_rows), "dropped": len(missing),
                      "rows": len(result), "base_ids": len(base_ids)}, ensure_ascii=False))


def command_combine(args: argparse.Namespace) -> None:
    """Combine staged stages into the single input layout expected by redaction."""
    names = ("messages_train.jsonl", "teacher_trajectory_audit.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        combined: dict[str, dict] = {}
        for directory in args.input_dir:
            path = directory / name
            if not path.exists():
                continue
            for row in read_rows(path):
                sample_id = str(row.get("sample_id") or "")
                if not sample_id or sample_id in combined:
                    raise SystemExit(f"duplicate/missing sample_id while combining {name}: {sample_id}")
                combined[sample_id] = row
        write_rows(args.output / name, list(combined.values()))
    print(json.dumps({"samples": len(combined)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--raw-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.set_defaults(func=command_audit)
    prep = sub.add_parser("prepare-intermediates")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--original-intermediates", type=Path, required=True)
    prep.add_argument("--fresh-intermediates", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(func=command_prepare_intermediates)
    merge = sub.add_parser("merge")
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--base", type=Path, required=True)
    merge.add_argument("--replacement", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--file-name", default="messages_train.jsonl")
    merge.add_argument("--exclude-sample-id-file", type=Path, action="append",
                       help="accepted repair IDs to drop because their regenerated stage is non-trainable")
    merge.add_argument("--insert-missing-after-sample-id",
                       help="insert regenerated sample IDs absent from base after this stable sample ID")
    merge.add_argument("--drop-unreplaced", action="store_true",
                       help="drop affected source samples when the regenerated pipeline correctly emits no stage")
    merge.set_defaults(func=command_merge)
    combine = sub.add_parser("combine")
    combine.add_argument("--input-dir", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(func=command_combine)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
