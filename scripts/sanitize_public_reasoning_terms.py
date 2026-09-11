#!/usr/bin/env python3
"""Apply conservative local replacements for residual private-source wording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPLACEMENTS = {
    "当前段与 GQS 参考 GQS 逐行一致": "当前段与谱面逐行一致",
    "与GQS方向一致": "与谱面方向一致",
    "参考谱面": "当前谱面",
    "参考谱": "当前谱面",
    "系统提示": "当前要求",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    for source in args.input_dir.iterdir():
        if source.is_file() and source.name != "messages_train.jsonl":
            (args.output_dir / source.name).write_bytes(source.read_bytes())

    rows = changed = replacements = 0
    with (args.input_dir / "messages_train.jsonl").open(encoding="utf-8") as source, \
            (args.output_dir / "messages_train.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            row_changed = False
            for message in row.get("messages") or []:
                content = str(message.get("content") or "")
                updated = content
                for before, after in REPLACEMENTS.items():
                    count = updated.count(before)
                    replacements += count
                    updated = updated.replace(before, after)
                if updated != content:
                    message["content"] = updated
                    row_changed = True
            changed += int(row_changed)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "schema_version": "teacher-public-term-sanitize-1.0",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "rows": rows,
        "changed_rows": changed,
        "replacements": replacements,
        "replacement_map": REPLACEMENTS,
    }
    (args.output_dir / "public_term_sanitize_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
