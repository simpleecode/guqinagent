#!/usr/bin/env python3
"""Persist user-approved tonic corrections while retaining App provenance."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_OVERRIDES = {
    "SMcokJse": "F",
    "SaN3X9UQ": "C",
    "S0ejf3Pz": "B",
    "SaljUbT2": "Eb",
    "SzyUVJiY": "Bb",
    "S74NcKEx": "Bb",
}

APP_VERIFIED_TONICS = {
    "SaljUbT2": "bE",
    "SzyUVJiY": "bB",
    "S74NcKEx": "bB",
}

# ``SzyUVJiY`` is an App-specific low-register bB score.  Most explicit
# ``1=bB`` displays (including S74NcKEx) use the ordinary B-flat-4 register.
TONIC_DEGREE1_MIDI_OVERRIDES = {
    "SzyUVJiY": 58,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()
    args.backup_root.mkdir(parents=True, exist_ok=True)
    changed = []
    for key, tonic in DEFAULT_OVERRIDES.items():
        raw = args.final_root / key / "out" / "raw_data.json"
        if not raw.exists():
            raise FileNotFoundError(raw)
        backup = args.backup_root / key / "raw_data.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(raw, backup)
        doc = json.loads(raw.read_text(encoding="utf-8-sig"))
        meta = doc.setdefault("metadata", {})
        app_tonic = meta.get("tonic")
        if "tonic_app_original" not in meta:
            meta["tonic_app_original"] = app_tonic
        if key in APP_VERIFIED_TONICS:
            meta["tonic_capture_original"] = app_tonic
            meta["tonic_app_verified"] = APP_VERIFIED_TONICS[key]
            meta["tonic_source"] = "user_verified_app_display"
        else:
            meta["tonic_source"] = "user_approved_pitch_audit_override"
        meta["tonic"] = tonic
        if key in TONIC_DEGREE1_MIDI_OVERRIDES:
            meta["tonic_degree1_midi"] = TONIC_DEGREE1_MIDI_OVERRIDES[key]
            meta["tonic_register_source"] = "user_verified_app_pitch_audit"
        else:
            meta.pop("tonic_degree1_midi", None)
            meta.pop("tonic_register_source", None)
        meta["tonic_override_updated_at"] = datetime.now().isoformat(timespec="seconds")
        score = doc.get("raw_score_data")
        if isinstance(score, dict):
            score["tonic"] = tonic
        raw.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed.append({"score_key": key, "app_tonic": app_tonic, "override_tonic": tonic})
    print(json.dumps({"changed": changed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
