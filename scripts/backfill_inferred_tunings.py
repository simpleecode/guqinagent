"""Backfill only explicitly labelled, name-based tuning inferences.

Runtime captures remain authoritative whenever they contain all seven string
offsets.  This helper supplies the standard seven-offset pattern only for
captures where the App did not emit its separate tuning callback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PATTERNS = [
    ("紧一二三四六七", [1, 1, 1, 0, 1, 1, 1]),
    ("紧二五七", [0, 1, 0, 0, 1, 0, 1]),
    ("紧二五", [0, 1, 0, 0, 1, 0, 0]),
    ("紧五慢一", [-1, 0, 0, 0, 1, 0, 0]),
    ("慢一三六", [-1, 0, -1, 0, 0, -1, 0]),
    ("紧五", [0, 0, 0, 0, 1, 0, 0]),
    ("慢三", [0, 0, -1, 0, 0, 0, 0]),
    ("慢二", [0, -1, 0, 0, 0, 0, 0]),
    ("正调", [0, 0, 0, 0, 0, 0, 0]),
]


def infer(name: str) -> list[int] | None:
    # Keep “调”: labels such as “正调定弦” must still match “正调”.
    normalized = name.replace("定弦", "")
    for label, offsets in PATTERNS:
        if label in normalized:
            return offsets
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    names = {row["score_key"]: row.get("tuning", "") for row in selection}
    changed = []
    unresolved = []
    for raw_path in args.final_root.glob("*/out/raw_data.json"):
        doc = json.loads(raw_path.read_text(encoding="utf-8"))
        meta = doc.setdefault("metadata", {})
        tuning = meta.get("tuning") or {}
        if isinstance(tuning.get("value"), list) and len(tuning["value"]) >= 7:
            continue
        score_key = meta.get("score_key") or raw_path.parents[1].name
        candidate_name = names.get(score_key, "") or tuning.get("name", "")
        offsets = infer(candidate_name)
        if offsets is None:
            unresolved.append(score_key)
            continue
        meta["tuning"] = {
            "name": candidate_name.replace("定弦", "") or None,
            "value": offsets,
            "source": "candidate_tuning_name_inferred",
            "inference_note": "Seven string semitone offsets inferred from the candidate tuning name; App register value was not captured.",
        }
        raw_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed.append(score_key)
    print(json.dumps({"changed": changed, "unresolved": unresolved}, ensure_ascii=False))


if __name__ == "__main__":
    main()
