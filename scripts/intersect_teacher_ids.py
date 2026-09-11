#!/usr/bin/env python3
"""Select trajectory IDs present in a teacher JSONL corpus."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wanted = {line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines()
              if line.strip()}
    found: set[str] = set()
    for line in args.teacher_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        trajectory_id = str(row.get("trajectory_id") or "")
        if not trajectory_id:
            match = re.match(r"^(.*)-(?:fingering_agent|guqinization)-teacher-tools$",
                             str(row.get("sample_id") or ""))
            trajectory_id = match.group(1) if match else ""
        if trajectory_id in wanted:
            found.add(trajectory_id)
    args.out.write_text("\n".join(sorted(found)) + "\n", encoding="utf-8")
    print(json.dumps({"requested": len(wanted), "found": len(found),
                      "missing": len(wanted - found)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
