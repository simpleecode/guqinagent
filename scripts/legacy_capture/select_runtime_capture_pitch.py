#!/usr/bin/env python3
"""Select a captured score's tuning and numbered-notation tonic by pitch audit.

The mobile runtime may contain cached tuning objects from another score, so its
first observed tuning object is deliberately *not* treated as ground truth.
For every supplied score this program reconstructs temporary score JSON for a
small, explicit set of historical guqin tunings and ``1=X`` labels, maps it,
and ranks the result using the existing jianpu--jianzipu pitch auditor.  Only
a clearly passing candidate is materialised in the requested score directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

# Seven string offsets plus the App's global transpose slot (always zero here:
# the displayed tonic is tested separately by the auditor).
TUNINGS = {
    "正调": [0, 0, 0, 0, 0, 0, 0, 0],
    "紧一二三四六七": [1, 1, 1, 0, 1, 1, 1, 0],
    "紧二五七": [0, 1, 0, 0, 1, 0, 1, 0],
    "紧二五": [0, 1, 0, 0, 1, 0, 0, 0],
    "紧五慢一": [-1, 0, 0, 0, 1, 0, 0, 0],
    "慢一三六": [-1, 0, -1, 0, 0, -1, 0, 0],
    "紧五": [0, 0, 0, 0, 1, 0, 0, 0],
    "慢三": [0, 0, -1, 0, 0, 0, 0, 0],
    "慢二": [0, -1, 0, 0, 0, 0, 0, 0],
}
TONICS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def run(command: list[str], *, quiet: bool = True) -> None:
    subprocess.run(
        command, cwd=ROOT, check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def extract_and_audit(capture: Path, meta: dict, tonic: str, name: str,
                      values: list[int], out: Path) -> dict:
    values_arg = ",".join(map(str, values))
    run([
        PYTHON, "scripts/extract_score_runtime_windows.py", "--input", str(capture),
        "--out-dir", str(out), "--score-id", str(meta["score_id"]),
        "--score-key", meta["score_key"], "--title", meta["title"],
        "--tonic", tonic, "--tuning-name", name, "--tuning-values", values_arg,
        "--assume-complete-runtime-window",
    ])
    mapped = out / "mapped"
    run([
        PYTHON, "scripts/extract_jianpu_jianzi.py", "--raw", str(out / "raw_data.json"),
        "--out-dir", str(mapped), "--tonic", tonic, "--tuning-name", name,
        "--tuning-values", values_arg,
    ])
    audit_path = out / "pitch_audit.json"
    run([
        PYTHON, "scripts/audit_jianpu_jianzi_pitch.py",
        str(mapped / "jianpu_jianzi_readable.json"), "--output", str(audit_path),
    ])
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    return {
        "tonic": tonic, "tuning_name": name, "tuning_values": values,
        **summary, "_out": str(out),
    }


def rank(row: dict) -> tuple[float, int, float]:
    # Primary target is an accurate matching explanation, not merely a tiny
    # subset of easy notes.  Count exact matched notes after match rate.
    return (float(row.get("match_rate") or -1), int(row.get("matched") or 0),
            -float(row.get("mean_absolute_cents") or 1e9))


def process(score_dir: Path, *, min_rate: float, min_compared: int,
            margin: float, keep_candidates: bool) -> int:
    manifest_path = score_dir / "evidence" / "capture_manifest.json"
    if not manifest_path.exists():
        print(f"skip {score_dir.name}: no cold-capture manifest")
        return 0
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture = score_dir / "evidence" / "runtime_live.jsonl"
    if not capture.exists():
        print(f"skip {score_dir.name}: missing runtime evidence")
        return 0
    candidate_root = score_dir / "candidate_audit"
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir()
    rows: list[dict] = []
    for tuning_name, values in TUNINGS.items():
        for tonic in TONICS:
            candidate = candidate_root / f"{tuning_name}_{tonic.replace('#', 'sharp')}"
            try:
                rows.append(extract_and_audit(capture, meta, tonic, tuning_name, values, candidate))
            except subprocess.CalledProcessError:
                continue
    rows.sort(key=rank, reverse=True)
    compact = [{key: value for key, value in row.items() if key != "_out"} for row in rows]
    (score_dir / "pitch_candidate_audit.json").write_text(
        json.dumps({"score": meta, "candidates": compact}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not rows:
        print(f"failed {score_dir.name}: no usable candidate")
        return 1
    best = rows[0]
    next_rate = float(rows[1].get("match_rate") or -1) if len(rows) > 1 else -1
    acceptable = (best.get("compared_notes", 0) >= min_compared
                  and (best.get("match_rate") or 0) >= min_rate
                  and (float(best.get("match_rate") or 0) - next_rate >= margin))
    print(
        f"{score_dir.name} {meta['title']}: 1={best['tonic']} {best['tuning_name']} "
        f"{best['matched']}/{best['compared_notes']}={best['match_rate']:.2%} "
        f"({'selected' if acceptable else 'REVIEW'})",
        flush=True,
    )
    if acceptable:
        selected = Path(best["_out"])
        for name in ("raw_data.json", "data.json", "pitch_audit.json"):
            shutil.copy2(selected / name, score_dir / name)
        final_mapped = score_dir / "mapped"
        if final_mapped.exists():
            shutil.rmtree(final_mapped)
        shutil.copytree(selected / "mapped", final_mapped)
        (score_dir / "pitch_selection.json").write_text(
            json.dumps({key: value for key, value in best.items() if key != "_out"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if not keep_candidates:
        shutil.rmtree(candidate_root)
    return 0 if acceptable else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_dirs", nargs="+", type=Path)
    parser.add_argument("--min-rate", type=float, default=0.70)
    parser.add_argument("--min-compared", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args()
    outcomes = [process(path, min_rate=args.min_rate, min_compared=args.min_compared,
                        margin=args.margin, keep_candidates=args.keep_candidates)
                for path in args.score_dirs]
    return 1 if any(outcome == 1 for outcome in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
