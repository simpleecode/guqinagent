#!/usr/bin/env python3
"""Export one leakage-safe tuning-decision sample per annotated score."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIT_PATH = ROOT / "scripts/audit_jianpu_jianzi_pitch.py"
spec = importlib.util.spec_from_file_location("tuning_export_audit", AUDIT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AUDIT)


def melody_features(data: dict) -> dict:
    metadata = data["metadata"]
    tonic = AUDIT.parse_tonic_midi(metadata)
    pitches = []
    degree_counts = Counter()
    for note in data.get("notes", []):
        value = AUDIT.parse_jianpu(note.get("jianpu"), tonic)
        if value is None:
            value = AUDIT.parse_jianpu(note.get("jianpu_alt"), tonic)
        if value is not None:
            pitches.append(value)
        text = str(note.get("jianpu") or note.get("jianpu_alt") or "")
        for degree in "1234567":
            if degree in text:
                degree_counts[degree] += 1
                break
    return {
        "tonic": metadata.get("tonic"), "meter": metadata.get("meter"),
        "sounding_notes": len(pitches), "pitch_min_midi": min(pitches) if pitches else None,
        "pitch_max_midi": max(pitches) if pitches else None,
        "degree_histogram": dict(sorted(degree_counts.items())),
        "sections": metadata.get("sections") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "ABC_J/results/dataset_split_groups.csv")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "ABC_J/results/agent_dataset_splits.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "ABC_J/agent_training/exported")
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.split_manifest.open(encoding="utf-8-sig", newline="") as handle:
        split_by_key = {row["score_key"]: row["split"] for row in csv.DictReader(handle)}
    for row in rows:
        row["split"] = split_by_key.get(row["score_key"], "")
    family_splits: dict[str, set[str]] = {}
    for row in rows:
        family_splits.setdefault(row["leakage_group_id"], set()).add(row["split"])
    leaking = {key: sorted(value) for key, value in family_splits.items() if len(value) > 1}
    if leaking:
        raise ValueError(f"families cross splits: {leaking}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {"train": (args.output_dir / "sft_tuning_decision_train.jsonl").open(
        "w", encoding="utf-8", newline="\n")}
    counts, labels, errors = Counter(), Counter(), []
    registry: dict[str, Counter[tuple[float, ...]]] = {}
    try:
        for row in rows:
            split = row.get("split")
            if split not in {"train", "validation", "test"}:
                errors.append({"score_key": row.get("score_key"), "error": "invalid_split"})
                continue
            path = Path(row["final_data_path"]) / "jianpu_jianzi_readable.json"
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                target_open = AUDIT.parse_open_midi(data["metadata"])
                tuning = data["metadata"].get("tuning") or {}
                target_name = str(tuning.get("name") or "")
                # Candidate ranking is generated at runtime from canonical ABC;
                # the training input intentionally stores melody features only.
                fingerprint = hashlib.sha256(
                    json.dumps(target_open, separators=(",", ":")).encode()
                ).hexdigest()[:16]
                item = {
                    "schema_version": "tuning-sft-1.0",
                    "sample_id": f"{row['score_key']}-tuning", "score_key": row["score_key"],
                    "score_family_id": row["leakage_group_id"], "split": split,
                    "task": "select_guqin_tuning",
                    "input": {"melody_features": melody_features(data), "tuning_locked": False},
                    "target": {
                        "selected_name": target_name, "open_midi": target_open,
                        "fingerprint": fingerprint, "provenance": tuning.get("source", "score_metadata"),
                    },
                    "loss_mask": {"target": True, "input": False},
                    "source": str(path),
                }
                if split == "train":
                    handles[split].write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[split] += 1
                labels[target_name or fingerprint] += 1
                registry.setdefault(target_name or fingerprint, Counter())[tuple(target_open)] += 1
            except Exception as exc:
                errors.append({"score_key": row.get("score_key"), "error": f"{type(exc).__name__}: {exc}"})
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "schema_version": "tuning-export-1.0", "counts": dict(counts),
        "label_distribution": dict(labels), "errors": errors,
        "leakage_unit": "score_family_id", "one_sample_per_score": True,
    }
    registry_rows = []
    for name, variants in sorted(registry.items()):
        for open_midi, count in variants.most_common():
            registry_rows.append({
                "name": name, "open_midi": list(open_midi), "score_count": count,
                "status": "observed", "conflict": len(variants) > 1,
            })
    (args.output_dir / "observed_tuning_registry.json").write_text(
        json.dumps({"schema_version": "tuning-registry-1.0", "tunings": registry_rows},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "tuning_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
