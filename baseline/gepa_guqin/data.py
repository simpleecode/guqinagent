"""Carve the GEPA optimization split out of the training trajectories.

The split lives entirely inside the current train half
(``renderfix_train_test_20261007/pitch_eligible_two_or_half/inferred_trajectories_train.jsonl``
by default) so the test half is only ever touched by the final reporting
pipeline.  Each prepared example is a *phrase* made independent of its
neighbours: the previous phrase of the same score is injected as a read-only
reference handoff (exactly the shape ``eval_two_stage_score.py`` injects from
the model's own previous output), and the reference rows of every selected
score are kept alongside so rollouts can replay phrase-spanning state.

Outputs (under --out-dir):
- examples_train.jsonl / examples_val.jsonl: prepared per-phrase examples.
- historical.jsonl: deduplicated source rows of all selected scores (the
  reference-side phrase history handed to RealToolRuntime).
- manifest.json: sampling bookkeeping and overlap audit.

Run as a module from the repository root:
    python -m baseline.gepa_guqin.data \
        --source ABC_J/agent_training/reference_trajectories_train.jsonl \
        --out-dir baseline/gepa_guqin/data/gepa_split_v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.gepa_guqin.compat import materialize_actions  # noqa: E402
from baseline.gepa_guqin.metric import reference_self_audit  # noqa: E402

NEW_SPLIT_DIR = ROOT / "ABC_J/agent_training/renderfix_train_test_20261007/pitch_eligible_two_or_half"
DEFAULT_SOURCE = NEW_SPLIT_DIR / "inferred_trajectories_train.jsonl"
# Any score listed here (score_key per line of a JSONL, or one per line of a
# plain text list) is refused from the optimization split.
DEFAULT_EXCLUDE = NEW_SPLIT_DIR / "inferred_trajectories_test.jsonl"


def _phrase_order_key(row: dict[str, Any]):
    indexes = [int(note["index"]) for note in
               (row.get("input") or {}).get("notes_without_jianzi") or []
               if note.get("index") is not None]
    return (min(indexes) if indexes else -1, str(row.get("phrase_id") or ""))


def _has_reference(row: dict[str, Any]) -> bool:
    actions = ((row.get("reference_plan") or {}).get("actions")) or []
    return bool(actions)


def prepare_split(source: Path, out_dir: Path, *, phrases: int, seed: int,
                  exclude_files: list[Path],
                  val_fraction: float, eligible_ids: Path | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in
            source.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Pitch-eligible export: only phrases whose reference audits cleanly against
    # the jianpu are usable as optimization targets (same filter the teacher
    # corpus build applies).  Prefer the split-specific sibling list.
    if eligible_ids is None:
        for name in ("pitch_eligible_phrase_ids_train.txt", "pitch_eligible_phrase_ids.txt"):
            sibling = source.parent / name
            if sibling.exists():
                eligible_ids = sibling
                break
    eligible: set[str] | None = None
    if eligible_ids is not None:
        eligible = {line.strip() for line in
                    eligible_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        eligible_rows = [row for row in rows if str(row["trajectory_id"]) in eligible]
        print(f"pitch-eligible filter {eligible_ids.name}: "
              f"{len(rows)} -> {len(eligible_rows)} example phrases "
              f"(history keeps all {len(rows)})")
    else:
        eligible_rows = rows

    excluded_scores: set[str] = set()
    for exclude_path in exclude_files:
        if not exclude_path.exists():
            continue
        for line in exclude_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                excluded_scores.add(str(json.loads(line).get("score_key") or ""))
            else:
                excluded_scores.add(line.strip())

    by_score: dict[str, list[dict[str, Any]]] = {}
    for row in eligible_rows:
        if not _has_reference(row):
            continue
        by_score.setdefault(str(row["score_key"]), []).append(row)
    # Full per-score history (every row with a reference, eligible or not) so
    # the true predecessor always exists for handoff and state replay.
    history_by_score: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not _has_reference(row):
            continue
        history_by_score.setdefault(str(row["score_key"]), []).append(row)
    overlap = sorted(set(by_score) & excluded_scores)
    if overlap:
        # The training file should already be disjoint from the sealed sets;
        # refuse to silently optimize on evaluation material.
        raise SystemExit(
            f"source scores overlap the evaluation protocol: {overlap[:10]}"
            f" ({len(overlap)} total); refusing to build the split"
        )

    score_keys = sorted(by_score)
    rng = random.Random(seed)
    rng.shuffle(score_keys)
    selected: list[str] = []
    total = 0
    for key in score_keys:
        if total >= phrases:
            break
        selected.append(key)
        total += len(by_score[key])

    # Balance the two halves by phrase count, then fix the split by score so
    # no phrase of one half can leak the other's handoff history.
    selected.sort(key=lambda key: (-len(by_score[key]), key))
    train_keys: list[str] = []
    val_keys: list[str] = []
    train_phrases = val_phrases = 0
    target_val = round(total * val_fraction)
    for key in selected:
        count = len(by_score[key])
        if val_phrases < target_val and (val_phrases + count) - target_val <= (train_phrases + count) - (total - target_val):
            val_keys.append(key)
            val_phrases += count
        else:
            train_keys.append(key)
            train_phrases += count

    # Full reference history of the selected scores (deduplicated rows,
    # eligible or not).  The rendered surface is materialised into jianzi_text
    # so the prompt renderer and context tools see the legacy field regardless
    # of export schema.
    historical: dict[tuple[str, str], dict[str, Any]] = {}
    for key in selected:
        for row in history_by_score[key]:
            normalized = dict(row)
            normalized["reference_plan"] = {
                "actions": materialize_actions(row["reference_plan"]["actions"]),
            }
            historical[(str(row["score_key"]), str(row["phrase_id"]))] = normalized

    def build_examples(keys: list[str], split_name: str) -> list[dict[str, Any]]:
        examples = []
        for key in keys:
            history = sorted(history_by_score[key], key=_phrase_order_key)
            position_of = {str(row["trajectory_id"]): position
                           for position, row in enumerate(history)}
            for row in sorted(by_score[key], key=_phrase_order_key):
                position = position_of[str(row["trajectory_id"])]
                previous = history[position - 1] if position else None
                # The runtime item ships clean: the actor's handoff and
                # harmonic flag are injected at rollout time from the
                # candidate's own previous output (ScoreRunState), exactly as
                # in the production serial loop.  The reference lives only in
                # the private metric evaluator below.
                item_input = dict(row["input"])
                item_input.setdefault("phrase_handoff", {}).pop("previous_phrase", None)
                item = {
                    "input": item_input,
                    "baseline_plan": row.get("baseline_plan") or {},
                }
                example = {
                    "sample_id": str(row["trajectory_id"]),
                    "split": split_name,
                    "score_key": str(row["score_key"]),
                    "phrase_id": str(row["phrase_id"]),
                    "previous_sample_id": (str(previous["trajectory_id"])
                                           if previous is not None else None),
                    "runtime_item": item,
                    "reference": {
                        "actions": historical[
                            (str(row["score_key"]), str(row["phrase_id"]))
                        ]["reference_plan"]["actions"],
                    },
                }
                # Reference-inherent failures: excluded from pitch/playability
                # components and reflection feedback so the model is never
                # blamed for events the sealed reference itself misses.
                # (Private evaluator only.)
                example["audit"] = reference_self_audit(example, historical)
                examples.append(example)
        return examples

    train_examples = build_examples(train_keys, "gepa_train")
    val_examples = build_examples(val_keys, "gepa_val")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in (("examples_train.jsonl", train_examples),
                           ("examples_val.jsonl", val_examples)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    with (out_dir / "historical.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in historical.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(source),
        "seed": seed,
        "target_phrases": phrases,
        "available_scores": len(by_score),
        "selected_scores": len(selected),
        "selected_phrases": total,
        "train_scores": len(train_keys), "train_phrases": len(train_examples),
        "val_scores": len(val_keys), "val_phrases": len(val_examples),
        "excluded_from": [str(path) for path in exclude_files],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="training trajectories in the public runtime schema "
                             "(input.notes_without_jianzi + reference_plan)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--phrases", type=int, default=600,
                        help="approximate number of phrases to sample (score-granular)")
    parser.add_argument("--val-fraction", type=float, default=0.25,
                        help="fraction of phrases assigned to the GEPA val half")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eligible-ids", type=Path, default=None,
                        help="pitch-eligible phrase-id list; defaults to "
                             "pitch_eligible_phrase_ids_train.txt next to --source")
    parser.add_argument("--exclude", type=Path, action="append",
                        default=[DEFAULT_EXCLUDE],
                        help="JSONL (or one-score-per-line text) whose score_keys must "
                             "stay out of the optimization split; repeatable")
    args = parser.parse_args()
    manifest = prepare_split(
        args.source, args.out_dir, phrases=args.phrases, seed=args.seed,
        exclude_files=args.exclude,
        val_fraction=args.val_fraction, eligible_ids=args.eligible_ids,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
