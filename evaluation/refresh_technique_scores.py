"""Refresh lexical technique metrics from saved full-score evaluation events.

Never runs inference, changes source plans, or recomputes pitch/rule semantics.
Uses the same aggregate functions as run_eval; preserves pitch/warning evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .aggregate import metrics, per_piece
from .run_eval import load_jsonl, write_jsonl
from .techniques import SCORING_VERSION, STATE_MARKERS, scoring_technique_fields

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def equivalent(first, second):
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(equivalent(first[k], second[k]) for k in first)
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(first, second, rel_tol=1e-12, abs_tol=1e-12)
    return first == second


def refresh(name):
    directory = ROOT / "evaluation_results" / name
    summary_path = directory / "summary.json"
    old = json.loads(summary_path.read_text())
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    paths = []
    for field in ("prediction_path", "reference_path"):
        path = Path(config[field])
        if str(path).startswith("/Volumes/F/work/Gunqin-agent/"):
            path = ROOT / path.relative_to("/Volumes/F/work/Gunqin-agent")
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    before = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    # Check saved event set against sealed inputs, not just its summary count.
    predictions = load_jsonl(paths[0], "sample_id")
    references = load_jsonl(paths[1], "sample_id")
    assert len(predictions) == old["prediction_samples"]
    assert len(predictions.keys() & references.keys()) == old["matched_samples"]
    events_path = directory / "per_event.jsonl"
    original_events_sha = digest(events_path)
    events = read_rows(events_path)
    assert len(events) == old["events"]
    assert len({(r["piece_id"], r["phrase_id"], r["event_id"]) for r in events}) == len(events)
    control_counts = {"prediction": 0, "reference": 0}
    for event in events:
        for side in ("prediction", "reference"):
            text_key = side + "_text"
            if text_key not in event:
                raise ValueError(f"Missing original notation: {name}/{text_key}")
            fields = scoring_technique_fields(event[text_key])
            event[side].update(fields)
            event[side]["pre_attack_ornaments"] = [
                v for v in event[side].get("pre_attack_ornaments", []) if v not in STATE_MARKERS]
            assert not set(STATE_MARKERS) & set(fields["ornaments"] + fields["fingering_techniques"])
            control_counts[side] += bool(fields["notation_state_markers"])
    violations_path = directory / "violations.jsonl"
    violations_sha = digest(violations_path)
    violations = read_rows(violations_path)
    audited_count = old["metrics"]["rule_warning_audited_phrases"]
    new = metrics(events, violations, config["pitch_tolerance_cents"], audited_phrases=audited_count)
    changed = {"ornament", "performance_technique_usage"}
    for field, value in old["metrics"].items():
        if field not in changed:
            # JSON round trips / summation order can alter the last float bit.
            assert equivalent(new[field], value), (name, field, value, new[field])
            new[field] = value  # retain the exact published unaffected value
    # Keep one compact revision record, not dated duplicate result directories.
    if old.get("technique_scoring_version") != SCORING_VERSION:
        old["technique_metric_revision"] = {
            "from_version": old.get("technique_scoring_version", "legacy-split-extractors"),
            "previous_ornament": old["metrics"]["ornament"],
            "previous_technique_similarity": old["metrics"]["performance_technique_usage"]["distribution_similarity"],
            "previous_events_sha256": original_events_sha,
        }
    old.update(schema_version="guqin-final-eval-1.2", technique_scoring_version=SCORING_VERSION,
               metrics=new)
    config["technique_metric_refresh"] = {
        "version": SCORING_VERSION, "source_sha256": before,
        "evaluation_code_sha256": {str(path.relative_to(ROOT)): digest(path)
                                   for path in (ROOT / "evaluation").glob("*.py")},
        "pitch_and_rule_evidence": "unchanged; reuses saved complete-score replay",
        "state_marker_events": control_counts,
    }
    # Audited phrase keys can be reconstructed from the original full-score lists.
    excluded = set(old["warning_replay_incomplete_scores"])
    audited = {(r["piece_id"], r["phrase_id"]) for r in events if r["piece_id"] not in excluded}
    assert len(audited) == audited_count
    write_jsonl(events_path, events)
    write_jsonl(directory / "per_piece.jsonl", per_piece(events, violations,
                config["pitch_tolerance_cents"], audited))
    summary_path.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    assert digest(violations_path) == violations_sha
    assert all(digest(path) == before[str(path.relative_to(ROOT))] for path in paths)
    print(json.dumps({"experiment": name, "ornament_f1": new["ornament"]["f1"],
                      "technique_similarity": new["performance_technique_usage"]["distribution_similarity"],
                      "protected_metrics_unchanged": True}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+")
    for experiment in parser.parse_args().experiments:
        refresh(experiment)
