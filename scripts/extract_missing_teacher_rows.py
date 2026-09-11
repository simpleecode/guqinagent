#!/usr/bin/env python3
"""Extract raw teacher rows absent from an existing redacted candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--present", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    present = {row["sample_id"] for row in read(args.present / "messages_train.jsonl")}
    for filename in ("messages_train.jsonl", "teacher_trajectory_audit.jsonl"):
        rows = [row for row in read(args.raw / filename) if row.get("sample_id") not in present]
        write(args.output / filename, rows)
    write(args.output / "messages_validation_report.json", [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
