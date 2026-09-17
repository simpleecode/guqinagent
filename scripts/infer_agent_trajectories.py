#!/usr/bin/env python3
"""Infer canonical, auditable agent trajectories from annotated scores."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import hashlib
from dataclasses import replace
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.inverse_baseline import build_phrase_baseline
from agents.abc_to_jianzipu.inverse_diff import infer_minimal_patches
from agents.abc_to_jianzipu.reference_parser import parse_reference_actions
from agents.abc_to_jianzipu.context_handoff import attach_inferred_handoffs
from agents.abc_to_jianzipu.phrase_splitter import split_phrase_ranges
from agents.abc_to_jianzipu.repeat_materializer import (
    OMITTED_PLACEHOLDER,
    materialize_repeats,
)
from agents.abc_to_jianzipu.teacher_gqs import parse_teacher_gqs


MANIFEST = ROOT / "ABC_J" / "results" / "dataset_split_groups.csv"
SPLIT_MANIFEST = ROOT / "ABC_J" / "results" / "agent_dataset_splits.csv"
OUTPUT_DIR = ROOT / "ABC_J" / "agent_training" / "inferred"
AUDIT_PATH = ROOT / "scripts" / "audit_jianpu_jianzi_pitch.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("infer_trajectory_audit", AUDIT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit()


def phrase_ranges(notes: list[dict], tonic_midi: float, max_sounding: int):
    return split_phrase_ranges(
        notes, max_sounding=max_sounding,
        is_sounding=lambda note: any(
            AUDIT.parse_jianpu(note.get(field), tonic_midi) is not None
            for field in ("jianpu", "jianpu_alt")
        ),
    )


def assign_event_indices(notes: list[dict]) -> None:
    """Assign continuous agent-facing ordinals, excluding structural bars."""
    event_index = 0
    for note in notes:
        is_bar = (str(note.get("jianpu") or "").strip() == "|"
                  or str(note.get("abc") or "").strip() == "|"
                  or note.get("duration") == "小节线")
        if is_bar:
            note["event_index"] = None
        else:
            note["event_index"] = event_index
            event_index += 1


def infer_score(row: dict, max_sounding: int, *, gqs_dir: Path | None = None,
                materialize_repeats_enabled: bool = False) -> list[dict]:
    if gqs_dir is not None:
        source = gqs_dir / row["score_key"] / "teacher.gqs"
        if not source.exists():
            raise FileNotFoundError(f"missing canonical GQS: {source}")
        data = parse_teacher_gqs(source.read_text(encoding="utf-8"))
    else:
        source = Path(row["final_data_path"]) / "jianpu_jianzi_readable.json"
        data = json.loads(source.read_text(encoding="utf-8"))
    assign_event_indices(data["notes"])
    repeat_report = None
    if materialize_repeats_enabled:
        tonic_midi = AUDIT.parse_tonic_midi(data["metadata"])
        data["notes"], repeat_report = materialize_repeats(
            data["notes"],
            is_sounding=lambda note: any(
                AUDIT.parse_jianpu(note.get(field), tonic_midi) is not None
                for field in ("jianpu", "jianpu_alt")
            ),
        )
    audit = AUDIT.audit(data, 50.0)
    notes_by_index = {
        int(note["index"]): note for note in data["notes"]
        if note.get("index") is not None
    }
    omitted_indices = {
        index for index, note in notes_by_index.items()
        if note.get("notation_omitted")
    }
    all_references = {
        action.source_index: action for action in parse_reference_actions(data, audit)
    }
    # Repeat materialisation marks the existing sounding rows whose printed
    # glyph is omitted.  These rows are still reference targets at the text
    # layer: the teacher must see the omission explicitly and Guqinizer must be
    # able to clear a Base glyph.  Do not reuse the parser's inherited empty
    # continuation semantics; use a dedicated visible placeholder.  A marker
    # row such as ``从ㄱ再作`` keeps its own marker text, while only its
    # following blank rows receive the omission placeholder.
    for index in omitted_indices:
        action = all_references.get(index)
        note = notes_by_index.get(index) or {}
        raw_text = str(note.get("jianzi") or "").strip()
        is_bar = (str(note.get("jianpu") or "").strip() == "|"
                  or str(note.get("abc") or "").strip() == "|")
        if action is None:
            continue
        if raw_text or is_bar:
            continue
        all_references[index] = replace(
            action,
            text=OMITTED_PLACEHOLDER,
            jianzi_text=OMITTED_PLACEHOLDER,
            confidence_class="verified",
            confidence=1.0,
            evidence=list(action.evidence) + ["repeat_omission_placeholder"],
        )
    tonic_midi = AUDIT.parse_tonic_midi(data["metadata"])
    open_midi = AUDIT.parse_open_midi(data["metadata"])
    tuning_payload = json.dumps(open_midi, separators=(",", ":"))
    tuning_fingerprint = hashlib.sha256(tuning_payload.encode()).hexdigest()[:16]
    trajectories = []
    for phrase_number, (start, end) in enumerate(
        phrase_ranges(data["notes"], tonic_midi, max_sounding), 1
    ):
        source_indices = {
            int(note["index"]) for note in data["notes"][start:end]
        }
        references = [
            all_references[index] for index in sorted(source_indices)
            if index in all_references
        ]
        baseline = build_phrase_baseline(data, start, end)
        inferred = infer_minimal_patches(baseline, references)
        class_counts = Counter(action.confidence_class for action in references)
        supervised_patch_ids = [
            patch["patch_id"] for patch in inferred["patches"]
            if patch["confidence_class"] == "verified"
        ]
        steps = [{"type": "observation", "content": {
            "baseline": baseline,
            "reference_confidence": dict(Counter(
                action.confidence_class for action in references
            )),
        }}]
        for patch in inferred["patches"]:
            for tool in patch["required_tools"]:
                steps.append({"type": "inferred_tool_call", **tool})
            steps.append({"type": "decision", "patch": patch})
        steps.append({"type": "audit", "content": {
            "reference_pitch": {
                "matched": sum(
                    action.confidence_class == "verified" for action in references
                ),
                "conflicting": sum(
                    action.confidence_class == "conflicting" for action in references
                ),
            },
            "unresolved_differences": inferred["unresolved_differences"],
        }})
        trajectories.append({
            "trajectory_version": "inverse-1.3",
            "trajectory_id": f"{row['score_key']}-p{phrase_number:04d}",
            "score_family_id": row["leakage_group_id"],
            "score_key": row["score_key"],
            "phrase_id": f"p{phrase_number:04d}",
            "split": row["split"],
            "input": {
                "metadata": data["metadata"],
                "normalized_tuning": {
                    "name": str(data["metadata"].get("tuning", {}).get("name", "")),
                    "open_midi": open_midi,
                    "fingerprint": tuning_fingerprint,
                },
                "event_range": {"start": start, "end_exclusive": end},
                "notes_without_jianzi": [
                    {key: note.get(key) for key in (
                        "index", "abc", "jianpu", "jianpu_alt", "duration", "lyric",
                        "section", "notation_omitted", "event_index",
                    )}
                    for note in data["notes"][start:end]
                ],
            },
            "baseline_plan": baseline,
            "reference_plan": {"actions": [action.to_dict() for action in references]},
            "canonical_trajectory": steps,
            "patches": inferred["patches"],
            "unresolved_differences": inferred["unresolved_differences"],
            "quality": {
                **inferred["summary"],
                "reference_classes": dict(class_counts),
                "supervised_patch_ids": supervised_patch_ids,
                "masked_source_indices": sorted({
                    item["source_index"] for item in inferred["unresolved_differences"]
                }),
            },
            "provenance": {
                "source": str(source),
                "algorithm": "answer_blind_baseline+reference_state_expansion+minimal_structured_diff",
                "repeat_materialization": repeat_report,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        })
    return attach_inferred_handoffs(trajectories)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--gqs-dir", type=Path,
                        help="read canonical teacher.gqs files instead of source JSON")
    parser.add_argument("--max-sounding", type=int, default=16,
                        help="超过 N 个发音音后截到下一小节线；分节边界（如 <一>）永远优先")
    parser.add_argument("--materialize-repeats", action="store_true",
                        help="expand 再作 repeat instructions into sounding notes "
                             "flagged notation_omitted")
    parser.add_argument("--score-key", action="append", help="limit to one or more score keys")
    parser.add_argument("--score-key-file", type=Path,
                        help="newline-delimited score keys; combines with --score-key")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.split_manifest.exists():
        with args.split_manifest.open(encoding="utf-8-sig", newline="") as handle:
            split_by_key = {
                item["score_key"]: item["split"] for item in csv.DictReader(handle)
            }
        for row in rows:
            row["split"] = split_by_key.get(row["score_key"], row.get("split", ""))
    missing_split = [row["score_key"] for row in rows if row.get("split") not in {
        "train", "validation", "test"
    }]
    if missing_split:
        raise ValueError(f"scores missing a valid split: {missing_split[:10]}")
    if args.score_key or args.score_key_file:
        wanted = set(args.score_key or [])
        if args.score_key_file:
            wanted.update(line.strip() for line in
                          args.score_key_file.read_text(encoding="utf-8").splitlines()
                          if line.strip())
        rows = [row for row in rows if row["score_key"] in wanted]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (args.output_dir / f"inferred_trajectories_{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) for split in ("train", "validation", "test")
    }
    counts = Counter()
    classes = Counter()
    patch_types = Counter()
    supervised_patches = Counter()
    try:
        for index, row in enumerate(rows, 1):
            for item in infer_score(row, args.max_sounding, gqs_dir=args.gqs_dir,
                                    materialize_repeats_enabled=args.materialize_repeats):
                handles[item["split"]].write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[item["split"]] += 1
                classes.update(item["quality"]["reference_classes"])
                patch_types.update(patch["patch_type"] for patch in item["patches"])
                supervised_ids = set(item["quality"]["supervised_patch_ids"])
                supervised_patches.update(
                    patch["patch_type"] for patch in item["patches"]
                    if patch["patch_id"] in supervised_ids
                )
            if index % 10 == 0:
                print(f"inferred {index}/{len(rows)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "schema_version": "inverse-1.3", "scores": len(rows),
        "source_format": "teacher.gqs" if args.gqs_dir else "readable.json",
        "trajectories": dict(counts), "reference_action_classes": dict(classes),
        "all_patch_types": dict(patch_types),
        "verified_supervised_patch_types": dict(supervised_patches),
    }
    (args.output_dir / "inferred_trajectories_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
