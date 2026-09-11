#!/usr/bin/env python3
"""Group-held-out baseline for selecting a guqin tuning from a fixed catalog.

This is deliberately a small, inspectable ranking baseline rather than an
agent.  It scores every (score, catalog tuning) pair from sounding pitches and
resource availability, then learns a regularised pairwise logistic ranker.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "ABC_J/agent_training/exported/observed_tuning_registry.json"
MANIFEST = ROOT / "ABC_J/results/dataset_split_groups.csv"
STANDARD = np.array([48, 50, 53, 55, 57, 60, 62], dtype=float)


def load_audit():
    spec = importlib.util.spec_from_file_location("pitch_audit", ROOT / "scripts/audit_jianpu_jianzi_pitch.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit()


def load_rows():
    catalog = json.loads(REGISTRY.read_text(encoding="utf-8"))["tunings"]
    variants = []
    seen = Counter()
    for tuning in catalog:
        name = tuning["name"]
        seen[name] += 1
        display = name if seen[name] == 1 else f"{name}（变体 {seen[name]}）"
        variants.append({"name": name, "display": display, "open": np.array(tuning["open_midi"], dtype=float)})

    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    rows = []
    for item in manifest:
        source = Path(item["final_data_path"]) / "jianpu_jianzi_readable.json"
        if not source.exists():
            continue
        data = json.loads(source.read_text(encoding="utf-8"))
        meta = data.get("metadata", {})
        tuning = meta.get("tuning") or {}
        actual = tuple(float(x) for x in (tuning.get("open_midi") or []))
        if len(actual) != 7:
            # Captured files store the same information under open_strings.
            try:
                actual = tuple(AUDIT.parse_open_midi(meta))
            except (KeyError, ValueError):
                continue
        target = next((i for i, c in enumerate(variants) if tuple(c["open"]) == actual), None)
        if target is None:
            continue
        tonic = AUDIT.parse_tonic_midi(meta)
        pitches, degrees = [], Counter()
        for note in data.get("notes", []):
            token = note.get("jianpu") or note.get("jianpu_alt")
            midi = AUDIT.parse_jianpu(token, tonic)
            if midi is None:
                continue
            pitches.append(int(round(midi)))
            for degree in "1234567":
                if degree in str(token):
                    degrees[int(degree) - 1] += 1
                    break
        if pitches:
            rows.append({"id": item["score_key"], "family": item["leakage_group_id"],
                         "title": meta.get("title") or item["candidate_title"], "tonic": meta.get("tonic", ""),
                         "pitches": np.array(pitches), "degrees": degrees, "target": target})
    return rows, variants


def resource_features(row, candidate, candidate_index, candidate_count):
    """Resource features available before a tuning decision is made."""
    pitches = row["pitches"]
    opened = candidate["open"]
    # Exact open strings and exact natural-harmonic pitch classes.
    open_hit = np.isin(pitches, opened)
    harmonic_pitches = []
    for value in opened:
        for semitones in (12, 19, 24, 28, 31, 36):
            harmonic_pitches.append(value + semitones)
    harmonic_hit = np.isin(pitches, harmonic_pitches)
    stopped_reachable = np.array([np.any((p >= opened) & (p <= opened + 24)) for p in pitches])
    anchors = np.r_[pitches[:12], pitches[-12:]]
    anchor_hit = np.isin(anchors, opened) | np.isin(anchors, harmonic_pitches)
    degree = np.array([row["degrees"].get(i, 0) for i in range(7)], dtype=float)
    degree /= max(degree.sum(), 1)
    per_degree = []
    for i in range(7):
        # Absolute target pitch class of this degree in the score's active register.
        mask = np.array([str(i + 1) in "1234567" and False for _ in pitches])
        # We only retain degree histograms; pair the degree weight with catalog
        # resources, which avoids leaking the target tuning from the score file.
        per_degree.extend([degree[i] * float(np.any((opened % 12) == (pitches % 12)[0])) if len(pitches) else 0,
                           degree[i] * float(np.any((np.array(harmonic_pitches) % 12) == (pitches % 12)[0])) if len(pitches) else 0])
    pc = np.bincount(pitches % 12, minlength=12).astype(float)
    pc /= pc.sum()
    open_pc = np.zeros(12); open_pc[opened.astype(int) % 12] = 1
    harmonic_pc = np.zeros(12); harmonic_pc[np.array(harmonic_pitches, dtype=int) % 12] = 1
    # Interactions tell the model which frequently used pitch classes gain an
    # open/harmonic resource under this candidate.
    interactions = np.r_[pc * open_pc, pc * harmonic_pc]
    tonic_pc = int(round(AUDIT.parse_tonic_midi({"tonic": row["tonic"]}))) % 12
    tonic_onehot = np.eye(12)[tonic_pc]
    candidate_onehot = np.eye(candidate_count)[candidate_index]
    global_features = np.r_[degree, pc, tonic_onehot,
                            min(pitches) / 100, max(pitches) / 100, len(pitches) / 1000]
    candidate_features = np.r_[open_hit.mean(), harmonic_hit.mean(), stopped_reachable.mean(),
                                anchor_hit.mean(), np.mean(np.min(np.abs(pitches[:, None] - opened[None, :]), axis=1)),
                                np.abs(opened - STANDARD).sum() / 12, opened - STANDARD,
                                interactions, candidate_onehot]
    return np.r_[global_features, candidate_features]


def standardize(train, test):
    mean = train.mean(axis=0); scale = train.std(axis=0); scale[scale < 1e-9] = 1
    return (train - mean) / scale, (test - mean) / scale


def fit_logistic(x, y, positive_weight=9.0, l2=0.8, steps=40, rate=0.25):
    x = np.c_[np.ones(len(x)), x]
    w = np.zeros(x.shape[1])
    weights = np.where(y == 1, positive_weight, 1.0)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        grad = (x.T @ ((p - y) * weights)) / weights.sum()
        grad[1:] += l2 * w[1:] / len(y)
        w -= rate * grad
    return w


def fold_for_family(family):
    # Stable grouped split: versions of a melody always remain together.
    return sum(ord(char) for char in family) % 5


def evaluate(rows, variants):
    all_predictions = []
    n_candidates = len(variants)
    for fold in range(5):
        train_rows = [r for r in rows if fold_for_family(r["family"]) != fold]
        test_rows = [r for r in rows if fold_for_family(r["family"]) == fold]
        def matrix(items):
            x, y = [], []
            for row in items:
                for j, candidate in enumerate(variants):
                    x.append(resource_features(row, candidate, j, n_candidates))
                    y.append(float(j == row["target"]))
            return np.asarray(x), np.asarray(y)
        x_train, y_train = matrix(train_rows); x_test, _ = matrix(test_rows)
        x_train, x_test = standardize(x_train, x_test)
        model = fit_logistic(x_train, y_train)
        scores = 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(x_test)), x_test] @ model, -30, 30)))
        for offset, row in enumerate(test_rows):
            local = scores[offset * n_candidates:(offset + 1) * n_candidates]
            rank = np.argsort(-local)
            predicted = int(rank[0])
            all_predictions.append({"score_key": row["id"], "family": row["family"], "title": row["title"],
                                    "actual": variants[row["target"]]["display"], "predicted": variants[predicted]["display"],
                                    "actual_rank": int(np.where(rank == row["target"])[0][0]) + 1,
                                    "confidence": round(float(local[predicted] / max(local.sum(), 1e-9)), 4),
                                    "top3": " / ".join(variants[i]["display"] for i in rank[:3])})
    return all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "ABC_J/analysis/tuning_ranker_baseline")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows, variants = load_rows()
    predictions = evaluate(rows, variants)
    top1 = sum(p["actual_rank"] == 1 for p in predictions) / len(predictions)
    top3 = sum(p["actual_rank"] <= 3 for p in predictions) / len(predictions)
    baseline = max(Counter(p["actual"] for p in predictions).values()) / len(predictions)
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    result = {"samples": len(predictions), "catalog_variants": len(variants),
              "grouped_folds": 5, "majority_baseline_top1": round(baseline, 4),
              "ranker_top1": round(top1, 4), "ranker_top3": round(top3, 4),
              "catalog": [v["display"] for v in variants]}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
