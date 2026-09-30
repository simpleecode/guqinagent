#!/usr/bin/env python3
"""Filter a public JSONL evaluation input by an allow-listed score key file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--score-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    allowed = load_ids(args.score_ids)
    kept: list[str] = []
    found: set[str] = set()
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        score_key = str(row.get("score_key") or "")
        if score_key in allowed:
            kept.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            found.add(score_key)
    missing = sorted(allowed - found)
    if missing:
        raise SystemExit(f"score IDs absent from public input: {', '.join(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(json.dumps({"scores": len(found), "phrases": len(kept), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
