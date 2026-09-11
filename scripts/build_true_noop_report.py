#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.audit.read_text(encoding="utf-8"))
    # A true no-op is an exact surface match.  All-empty non-repeat phrases
    # are excluded by the data-quality policy even when Base is also empty.
    ids = [row["trajectory_id"] for row in rows
           if row.get("base_target_mismatches") == 0
           and not (row.get("all_actual_empty") and not row.get("prior_repeat_markers"))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"no_op_ids": ids}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"true_no_op": len(ids), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
