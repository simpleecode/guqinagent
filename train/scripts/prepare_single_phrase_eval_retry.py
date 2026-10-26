"""Prepare one phrase for a sequential eval rerun, freezing earlier history."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    score_rows = [row for row in read_rows(args.inputs)
                  if row["score_key"] == args.score]
    score_rows.sort(key=lambda row: row["phrase_id"])
    ids = [row["sample_id"] for row in score_rows]
    target = f"{args.score}-{args.phrase}"
    if target not in ids or len(ids) != len(set(ids)):
        raise ValueError("target missing or duplicate score phrase IDs")
    selected = score_rows[:ids.index(target) + 1]
    prior = {row["sample_id"]: row for row in read_rows(args.predictions)}
    frozen = []
    for row in selected[:-1]:
        old = prior[row["sample_id"]]
        if (old.get("input_sha256") != row.get("input_sha256")
                or not old.get("protocol_valid")
                or not (old.get("guqinizer") or {}).get("ok")):
            raise ValueError(f"invalid frozen history: {row['sample_id']}")
        frozen.append(old)
    if (target in prior and prior[target].get("input_sha256")
            != selected[-1].get("input_sha256")):
        raise ValueError(f"target input changed: {target}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    workers = args.output_dir / "workers"
    workers.mkdir()
    (args.output_dir / "inputs.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    (workers / "predictions_00.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frozen),
        encoding="utf-8",
    )
    print(json.dumps({"target": target, "frozen_prefix": len(frozen),
                      "input_phrases": len(selected),
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
