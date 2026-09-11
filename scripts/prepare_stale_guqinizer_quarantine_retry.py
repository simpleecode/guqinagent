#!/usr/bin/env python3
"""Prepare the two quality-quarantined Guqinizer rows for another attempt."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "ABC_J/agent_training/stale_guqinizer_rerun_v1"
IDS = ["S8Y5kmNg-p0024", "SumLbkVi-p0008"]

def main() -> int:
    rows = {json.loads(line)["trajectory_id"]: json.loads(line)
            for line in (RUN / "fingering_intermediates.jsonl").open(encoding="utf-8")}
    (RUN / "quarantine_retry_ids.txt").write_text("".join(x + "\n" for x in IDS), encoding="utf-8")
    with (RUN / "quarantine_retry_intermediates.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for value in IDS:
            out.write(json.dumps(rows[value], ensure_ascii=False) + "\n")
    print(json.dumps({"ids": IDS}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
