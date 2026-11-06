"""Build the frozen V9 prompt JSON used by the formal sequential evaluator."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "guqinagent"
sys.path.insert(0, str(REPO))

from ABC_J.scripts.generate_teacher_tool_trajectories import public_system_for


def main() -> None:
    production = public_system_for("single_stage", basic=False).rstrip()
    addendum = (ROOT / "single_stage_minimal_tone_compound_v9_addendum.txt").read_text(
        encoding="utf-8"
    ).strip()
    payload = {
        "single_stage": production + "\n\n" + addendum,
        "version": "v9-frozen",
        "composition": "production prompt + V9 addendum; fixed handbook added by evaluator",
    }
    output = ROOT / "v9_frozen_prompt.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
