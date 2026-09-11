"""Find trajectories whose editable prompt exposed the repeat omission marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUFFIX = "-fingering_agent-teacher-tools"
MARKER = "无（由于是再作部分，省略）"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path,
                        help="optional messages_train.jsonl whose stage IDs are skipped")
    args = parser.parse_args()
    excluded = set()
    if args.exclude:
        for row in read_jsonl(args.exclude):
            sample_id = str(row.get("sample_id") or "")
            if sample_id.endswith(SUFFIX):
                excluded.add(sample_id[: -len(SUFFIX)])
    ids = []
    for row in read_jsonl(args.input):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id.endswith(SUFFIX):
            continue
        trajectory_id = sample_id[: -len(SUFFIX)]
        if trajectory_id in excluded:
            continue
        for message in row.get("messages") or []:
            if message.get("role") != "user":
                continue
            text = str(message.get("content") or "")
            current = text.find("【当前段")
            if current >= 0 and MARKER in text[current:]:
                ids.append(trajectory_id)
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(value + "\n" for value in ids), encoding="utf-8")
    print(json.dumps({"affected_trajectories": len(ids), "output": str(args.output)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
