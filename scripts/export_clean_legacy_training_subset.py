#!/usr/bin/env python3
"""Build a conservative, reproducible clean subset from redacted trajectories.

This is deliberately a *quarantine export*: it keeps only phrases selected by
the history-aware pitch audit and removes every score whose source contains a
legacy compound-jianzi layout that could alter later inherited pitch state.
Those scores are listed separately for source re-rendering and teacher replay;
they must not silently remain in a purportedly clean training set.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, items):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redacted-dir", type=Path, required=True)
    parser.add_argument("--eligible-train", type=Path, required=True)
    parser.add_argument("--affected-score-list", type=Path, required=True,
                        help="one score_key per line; all its phrases are quarantined")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    eligible = {line.strip() for line in args.eligible_train.read_text(encoding="utf-8").splitlines() if line.strip()}
    affected = {line.strip() for line in args.affected_score_list.read_text(encoding="utf-8").splitlines() if line.strip()}
    kept, quarantined, not_selected = [], [], []
    for row in rows(args.redacted_dir / "messages_train.jsonl"):
        provenance = row.get("provenance") or {}
        phrase_id = str(provenance.get("source_trajectory_id") or "")
        score_key = str(provenance.get("score_key") or "")
        if phrase_id not in eligible:
            not_selected.append(row)
        elif score_key in affected:
            quarantined.append(row)
        else:
            kept.append(row)

    stage_counts = Counter(str(row.get("agent_stage") or "") for row in kept)
    phrase_ids = sorted({str((row.get("provenance") or {}).get("source_trajectory_id")) for row in kept})
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "messages_train.jsonl", kept)
    write_jsonl(args.output_dir / "quarantined_affected_messages.jsonl", quarantined)
    (args.output_dir / "kept_phrase_ids.txt").write_text("\n".join(phrase_ids) + "\n", encoding="utf-8")
    shutil.copy2(args.affected_score_list, args.output_dir / "quarantined_score_keys.txt")
    manifest = {
        "schema_version": "clean-legacy-quarantine-export-1.0",
        "policy": "history-aware eligible phrase AND score not affected by legacy compound state decoding",
        "source_redacted_dir": str(args.redacted_dir),
        "eligible_train": str(args.eligible_train),
        "affected_score_list": str(args.affected_score_list),
        "source_message_rows": len(kept) + len(quarantined) + len(not_selected),
        "kept_message_rows": len(kept),
        "kept_phrase_count": len(phrase_ids),
        "kept_by_stage": dict(sorted(stage_counts.items())),
        "quarantined_affected_message_rows": len(quarantined),
        "quarantined_affected_phrase_count": len({str((row.get("provenance") or {}).get("source_trajectory_id")) for row in quarantined}),
        "excluded_not_history_eligible_message_rows": len(not_selected),
        "next_step": "re-render quarantined score sources, regenerate teacher trajectories, redact, then merge replacements by sample_id",
    }
    (args.output_dir / "clean_subset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
