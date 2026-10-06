#!/usr/bin/env python3
"""Materialise captured scores using their direct runtime tuning object.

``note_tuning`` supplies a Nab object: its first seven integers describe
individual string offsets and its eighth integer is the App's global key
transpose.  Do not replace that final slot with a guessed zero.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_score_runtime_windows import extract_runtime_tuning  # noqa: E402

# Directly verified against the App's runtime Nab objects and the displayed
# key signature of captured scores.  Values are the eighth Nab element.
TONIC_BY_GLOBAL_TRANSPOSE = {0: "F", 5: "C", 7: "B", -7: "C", -2: "G"}


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def materialize(score_dir: Path) -> None:
    manifest = json.loads((score_dir / "evidence" / "capture_manifest.json").read_text(encoding="utf-8"))
    capture = score_dir / "evidence" / "runtime_live.jsonl"
    tuning, candidates = extract_runtime_tuning(capture)
    if tuning is None:
        raise RuntimeError(f"{score_dir.name}: expected exactly one runtime tuning, got {candidates!r}")
    values = tuning["value"]
    transpose = values[7]
    tonic = TONIC_BY_GLOBAL_TRANSPOSE.get(transpose)
    if tonic is None:
        raise RuntimeError(f"{score_dir.name}: unverified global transpose {transpose}; refusing to infer tonic")
    values_arg = ",".join(map(str, values))
    run([
        sys.executable, "scripts/extract_score_runtime_windows.py", "--input", str(capture),
        "--out-dir", str(score_dir), "--score-id", str(manifest["score_id"]),
        "--score-key", manifest["score_key"], "--title", manifest["title"],
        "--tonic", tonic, "--tuning-name", tuning["name"], "--tuning-values", values_arg,
        "--assume-complete-runtime-window",
    ])
    run([
        sys.executable, "scripts/extract_jianpu_jianzi.py", "--raw", str(score_dir / "raw_data.json"),
        "--out-dir", str(score_dir / "mapped"), "--tonic", tonic,
        "--tuning-name", tuning["name"], "--tuning-values", values_arg,
    ])
    run([
        sys.executable, "scripts/audit_jianpu_jianzi_pitch.py",
        str(score_dir / "mapped" / "jianpu_jianzi_readable.json"),
        "--output", str(score_dir / "pitch_audit.json"),
    ])
    record = {"tonic": tonic, "tuning_name": tuning["name"], "tuning_values": values,
              "source": "direct_runtime_note_tuning"}
    (score_dir / "direct_runtime_tuning.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"score_key": score_dir.name, **record}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for score_dir in args.score_dirs:
        materialize(score_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
