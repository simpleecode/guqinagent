#!/usr/bin/env python3
"""Build sealed evaluation pairs for the audited modern-score split.

Pairs the public runtime rows of ``train/eval_inputs/modern_audited.jsonl``
with per-phrase structured reference actions parsed from the annotated
readable sources under ``ABC_J/modern/<score_key>/mapped/`` using the same
``parse_reference_actions`` parser as the teacher pipeline.  ``runtime_item``
/``sample_id``/``input_sha256`` are copied verbatim from the audited rows so
the pairs stay byte-aligned with the public input.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.reference_parser import parse_reference_actions

DEFAULT_AUDITED = ROOT / "train/eval_inputs/modern_audited.jsonl"
DEFAULT_MODERN_ROOT = ROOT / "ABC_J/modern"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited", type=Path, default=DEFAULT_AUDITED)
    parser.add_argument("--modern-root", type=Path, default=DEFAULT_MODERN_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in
            args.audited.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    actions_by_score: dict[str, dict[int, dict]] = {}
    pairs: list[dict] = []
    empty_phrases: list[str] = []
    for row in rows:
        score_key = str(row["score_key"])
        if score_key not in actions_by_score:
            readable = args.modern_root / score_key / "mapped" / "jianpu_jianzi_readable.json"
            if not readable.exists():
                raise SystemExit(f"missing readable source: {readable}")
            source = json.loads(readable.read_text(encoding="utf-8"))
            actions_by_score[score_key] = {
                int(action.source_index): dataclasses.asdict(action)
                for action in parse_reference_actions(source)
            }
        phrase_indexes = {
            int(note["index"])
            for note in row["runtime_item"]["input"]["notes_without_jianzi"]
            if note.get("index") is not None
        }
        reference_actions = [
            actions_by_score[score_key][index]
            for index in sorted(phrase_indexes)
            if index in actions_by_score[score_key]
        ]
        if not reference_actions:
            empty_phrases.append(str(row["sample_id"]))
        pair = dict(row)
        pair["reference"] = {"actions": reference_actions}
        pairs.append(pair)

    pairs_path = args.out_dir / "evaluation_pairs_test.jsonl"
    pairs_path.write_text("".join(
        json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n"
        for pair in pairs), encoding="utf-8")

    runtime_path = args.out_dir / "test_runtime.jsonl"
    runtime_path.write_text(args.audited.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "audited": str(args.audited.relative_to(ROOT)),
        "scores": len({str(row["score_key"]) for row in rows}),
        "phrases": len(rows),
        "phrases_without_reference_actions": empty_phrases,
        "source": "ABC_J/modern/<score_key>/mapped/jianpu_jianzi_readable.json",
    }
    (args.out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "phrases_without_reference_actions"} | {
        "empty_reference_phrases": len(empty_phrases)}, ensure_ascii=False))
    if empty_phrases:
        print("phrases without reference actions: " + ", ".join(empty_phrases[:10]),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
