#!/usr/bin/env python3
"""Export paired two-stage phrase IDs with a strict pitch-eligibility rule.

Each retained phase must have at least two matched parseable pitch rows, or
matched rows must account for at least half of all parseable pitch rows.  When
using canonical inferred phrase seeds, the reference plan is shared by the
two stages; the manifest still records the paired-stage contract explicitly.

The exporter also removes a phrase containing a run of five or more empty
sound-event jianzi rows after an ordinary preceding jianzi.  The run state is
carried across phrase boundaries inside one score, because phrase splitting
must not turn one long missing tail into apparently harmless leading blanks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGES = ("fingering", "guqinizer")
_AUDIT_MODULE = None


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def audit_module():
    global _AUDIT_MODULE
    if _AUDIT_MODULE is None:
        path = ROOT / "scripts" / "audit_training_phase_pitch_parse_stats.py"
        spec = importlib.util.spec_from_file_location("pitch_eligible_audit", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        _AUDIT_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_AUDIT_MODULE)
    return _AUDIT_MODULE


def classify_canonical(row: dict[str, Any]) -> dict[str, Any]:
    module = audit_module()
    return module.classify(module.build_pitch_audit(row, row.get("reference_plan") or {}, 50.0))


def validate_normalized_tuning(row: dict[str, Any], source: Path) -> None:
    """Fail closed when a frozen trajectory carries stale tuning pitches.

    ``open_strings.pitch`` and ``octave`` already encode the sounding tuning.
    A trajectory must therefore carry exactly the same seven MIDI values that
    the current canonical pitch auditor derives from its metadata.  Letting a
    stale ``normalized_tuning`` through here would poison teacher tools even
    if the current parser implementation has already been corrected.
    """
    input_data = row.get("input") or {}
    actual = (input_data.get("normalized_tuning") or {}).get("open_midi")
    module = audit_module().PITCH_AUDIT
    expected = module.parse_open_midi(input_data.get("metadata") or {})
    if (not isinstance(actual, list) or len(actual) != 7
            or [float(value) for value in actual] != [float(value) for value in expected]):
        raise ValueError(
            "normalized_tuning mismatch: "
            f"{row.get('trajectory_id')} in {source}; "
            f"actual={actual!r}, expected={expected!r}"
        )


def strict_eligible(item: dict[str, Any]) -> bool:
    classification = item.get("classification") or item
    return bool(classification.get("parseable_and_at_least_two_matches_or_half_matched"))


EMPTY_JIANZI = {"", "—", "-", "null", "None", "[空]", "[减字待填写]"}
BLANK_RUN_EXCEPTIONS = ("掐撮三声", "再做", "再作", "从头再做", "从头再作", "从ㄱ再作")


def action_text(action: dict[str, Any]) -> str:
    value = action.get("jianzi_text", action.get("text", ""))
    return "" if value is None else str(value).strip()


def is_all_empty_reference_phrase(item: dict[str, Any]) -> bool:
    """Identify phrases with no usable reference glyph supervision.

    A phrase whose reference actions are all empty must not pass pitch-only
    eligibility: it would later be skipped by teacher generation and make the
    selected phrase manifest disagree with the trainable population.  Keep
    materialized repeat omissions, because they have explicit repeat semantics
    even when their visible cells are empty.
    """
    notes = ((item.get("input") or {}).get("notes_without_jianzi") or [])
    if any(note.get("notation_omitted") for note in notes):
        return False
    repeat = ((item.get("provenance") or {}).get("repeat_materialization") or {})
    if repeat.get("markers") or repeat.get("marker_only_notes") or repeat.get("copy_markers"):
        return False
    actions = ((item.get("reference_plan") or {}).get("actions") or [])
    return not any(action_text(action) for action in actions)


def is_sounding_note(note: dict[str, Any]) -> bool:
    return (str(note.get("jianpu") or "") not in {"|", "0", "0（休止）"}
            and not note.get("notation_omitted"))


def blank_run_exclusions(rows: list[dict[str, Any]], minimum: int) -> set[str]:
    """Return IDs participating in an invalid ordinary-jianzi blank run."""
    excluded: set[str] = set()
    by_score: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_score.setdefault(str(row.get("score_key") or ""), []).append(row)
    for phrases in by_score.values():
        phrases.sort(key=lambda row: int(((row.get("input") or {}).get("event_range") or {}).get("start", 0)))
        blank_count = 0
        predecessor = ""
        predecessor_is_exception = False
        for phrase in phrases:
            phrase_id = str(phrase["trajectory_id"])
            notes = {int(note["index"]): note for note in
                     ((phrase.get("input") or {}).get("notes_without_jianzi") or [])
                     if note.get("index") is not None}
            for action in ((phrase.get("reference_plan") or {}).get("actions") or []):
                source = action.get("source_index")
                note = notes.get(int(source)) if source is not None else None
                if not note or not is_sounding_note(note):
                    continue
                text = action_text(action)
                if text not in EMPTY_JIANZI:
                    blank_count = 0
                    predecessor = text
                    predecessor_is_exception = any(marker in text for marker in BLANK_RUN_EXCEPTIONS)
                    continue
                if not predecessor:
                    continue
                blank_count += 1
                if blank_count >= minimum and not predecessor_is_exception:
                    excluded.add(phrase_id)
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path,
                        help="phase-level audit JSON from audit_training_phase_pitch_parse_stats.py")
    parser.add_argument("--inferred-dir", type=Path,
                        help="canonical inferred phrase directory; audit reference_plan directly")
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"),
                        default=("train", "validation", "test"))
    parser.add_argument("--min-ordinary-blank-run", type=int, default=5,
                        help="exclude phrases in an ordinary-predecessor empty run of this many sounding events; 0 disables")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.stats) == bool(args.inferred_dir):
        raise ValueError("provide exactly one of --stats or --inferred-dir")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    grouped: dict[str, OrderedDict[str, dict[str, dict[str, Any]]]] = {
        split: OrderedDict() for split in args.splits
    }
    source_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in args.splits}
    if args.inferred_dir:
        for split in args.splits:
            source = args.inferred_dir / f"inferred_trajectories_{split}.jsonl"
            if not source.exists():
                raise FileNotFoundError(source)
            for row in read_jsonl(source):
                phrase_id = str(row.get("trajectory_id") or "")
                if not phrase_id:
                    raise ValueError(f"missing trajectory_id in {source}")
                validate_normalized_tuning(row, source)
                source_rows[split].append(row)
                result = classify_canonical(row)
                grouped[split][phrase_id] = {stage: result for stage in STAGES}
    else:
        stats = json.loads(args.stats.read_text(encoding="utf-8"))
        for row in stats.get("details", []):
            split = str(row.get("split") or "train")
            if split not in grouped:
                continue
            phrase_id = str(row.get("trajectory_id") or "")
            stage = str(row.get("stage") or "")
            if phrase_id and stage in STAGES:
                grouped[split].setdefault(phrase_id, {})[stage] = row

    args.output_dir.mkdir(parents=True)
    all_ids: list[str] = []
    splits_manifest: dict[str, Any] = {}
    for split, phrases in grouped.items():
        blank_excluded = (blank_run_exclusions(source_rows[split], args.min_ordinary_blank_run)
                          if args.inferred_dir and args.min_ordinary_blank_run > 0 else set())
        empty_reference_excluded = (
            {str(row["trajectory_id"]) for row in source_rows[split]
             if is_all_empty_reference_phrase(row)}
            if args.inferred_dir else set()
        )
        selected = [phrase_id for phrase_id, stages in phrases.items()
                    if phrase_id not in blank_excluded
                    and phrase_id not in empty_reference_excluded
                    and all(stage in stages and strict_eligible(stages[stage]) for stage in STAGES)]
        (args.output_dir / f"pitch_eligible_phrase_ids_{split}.txt").write_text(
            "".join(f"{phrase_id}\n" for phrase_id in selected), encoding="utf-8")
        all_ids.extend(selected)
        splits_manifest[split] = {
            "source_phrase_count": len(phrases),
            "selected_phrase_count": len(selected),
            "phrase_ids": f"pitch_eligible_phrase_ids_{split}.txt",
            "ordinary_blank_run_excluded_phrase_count": len(blank_excluded),
            "ordinary_blank_run_excluded_phrase_ids": sorted(blank_excluded),
            "all_empty_reference_excluded_phrase_count": len(empty_reference_excluded),
            "all_empty_reference_excluded_phrase_ids": sorted(empty_reference_excluded),
        }
    (args.output_dir / "pitch_eligible_phrase_ids.txt").write_text(
        "".join(f"{phrase_id}\n" for phrase_id in all_ids), encoding="utf-8")
    manifest = {
        "schema_version": "pitch-eligible-trajectory-export-2.0",
        "selection": {
            "paired_stages": list(STAGES),
            "rule": "each stage has >=2 matched parseable pitch rows or matched rows >= 50% of parseable pitch rows",
            "no_parseable_pitch": "excluded",
            "ordinary_blank_run": {
                "minimum_sounding_empty_events": args.min_ordinary_blank_run,
                "exception_predecessor_substrings": list(BLANK_RUN_EXCEPTIONS),
            },
        },
        "inputs": {
            "inferred_dir": str(args.inferred_dir) if args.inferred_dir else None,
            "stats": str(args.stats) if args.stats else None,
        },
        "splits": splits_manifest,
        "total_selected_phrase_count": len(all_ids),
        "all_phrase_ids": "pitch_eligible_phrase_ids.txt",
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
