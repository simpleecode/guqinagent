"""Expand trajectory IDs into stage-specific sample IDs for merge curation."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ids = [line.strip() for line in args.trajectory_ids.read_text(encoding="utf-8").splitlines()
           if line.strip()]
    sample_ids = [f"{trajectory_id}-{stage}-teacher-tools"
                  for trajectory_id in ids
                  for stage in ("fingering_agent", "guqinization")]
    args.output.write_text("".join(sample_id + "\n" for sample_id in sample_ids), encoding="utf-8")
    print(f"expanded {len(ids)} trajectory IDs to {len(sample_ids)} sample IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
