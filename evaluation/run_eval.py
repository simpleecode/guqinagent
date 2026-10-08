"""CLI for structured final-jianzipu evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aggregate import final_warning_stats, metrics, per_piece
from .parser import structured_events, structured_events_for_score
from .rules import violations


def load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row[key])] = row  # append-only prediction output: latest wins
    return result


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    predictions = load_jsonl(args.pred, "sample_id")
    references = load_jsonl(args.reference, "sample_id")
    all_events: list[dict[str, Any]] = []
    unavailable: list[str] = []
    state_replay_incomplete_scores: list[str] = []
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample_id, prediction in predictions.items():
        reference = references.get(sample_id)
        if reference is None:
            unavailable.append(sample_id)
            continue
        matched.append((prediction, reference))

    reference_ids_by_score: dict[str, set[str]] = {}
    for sample_id, reference in references.items():
        reference_ids_by_score.setdefault(str(reference.get("score_key") or ""), set()).add(sample_id)
    matched_by_score: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in matched:
        score_key = str(pair[1].get("score_key") or "")
        matched_by_score.setdefault(score_key, []).append(pair)
    for score_key, samples in matched_by_score.items():
        matched_ids = {str(reference["sample_id"]) for _, reference in samples}
        if matched_ids == reference_ids_by_score.get(score_key, set()):
            events, _ = structured_events_for_score(samples, args.pitch_tolerance)
            all_events.extend(events)
            continue
        state_replay_incomplete_scores.append(score_key)
        for prediction, reference in samples:
            events, _ = structured_events(prediction, reference, args.pitch_tolerance)
            # Mode accuracy requires a full-score replay.  Do not report
            # phrase-reset results as if they were valid state comparisons.
            for event in events:
                event["prediction"]["tone_type"] = None
                event["reference"]["tone_type"] = None
                event["tone_type_unavailable_reason"] = "incomplete_score_prediction"
            all_events.extend(events)
    all_violations = violations(all_events)
    output = args.output_root / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    config = {"experiment": args.experiment, "prediction_path": str(args.pred),
              "reference_path": str(args.reference), "model_or_checkpoint": args.model,
              "pitch_tolerance_cents": args.pitch_tolerance,
              "arguments": vars(args), "git_commit": git_commit(),
              "created_at": datetime.now(timezone.utc).isoformat()}
    summary = {"schema_version": "guqin-final-eval-1.0", "prediction_samples": len(predictions),
               "matched_samples": len(predictions) - len(unavailable), "unmatched_prediction_ids": unavailable,
               "tone_type_replay_incomplete_scores": sorted(state_replay_incomplete_scores),
               "events": len(all_events), "metrics": metrics(all_events, all_violations, args.pitch_tolerance),
               "final_warning": final_warning_stats(list(predictions.values()))}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output / "per_event.jsonl", all_events)
    write_jsonl(output / "per_piece.jsonl", per_piece(all_events, all_violations, args.pitch_tolerance))
    write_jsonl(output / "violations.jsonl", all_violations)
    return {"output": str(output), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="two-stage or one-stage prediction JSONL")
    parser.add_argument("--reference", type=Path, required=True, help="sealed evaluation_pairs JSONL")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--pitch-tolerance", type=float, default=50.0)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
