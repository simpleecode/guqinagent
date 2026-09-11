#!/usr/bin/env python3
"""Write the small retry list left by a stale Guqinizer repair run."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "ABC_J/agent_training/stale_guqinizer_rerun_v1"
RETRY_IDS = [
    "S0FKGjFt-p0008", "S0FKGjFt-p0038", "SrbyFUb1-p0043",
    "S8Y5kmNg-p0024", "SumLbkVi-p0008",
]


def main() -> int:
    ids = set(RETRY_IDS)
    source = {json.loads(line)["trajectory_id"]: json.loads(line)
              for line in (RUN / "fingering_intermediates.jsonl").open(encoding="utf-8")}
    selected = [source[value] for value in RETRY_IDS if value in source]
    (RUN / "retry_ids.txt").write_text(
        "".join(value + "\n" for value in RETRY_IDS), encoding="utf-8"
    )
    with (RUN / "retry_fingering_intermediates.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"requested": RETRY_IDS, "written": [row["trajectory_id"] for row in selected]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
