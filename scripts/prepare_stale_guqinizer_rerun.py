#!/usr/bin/env python3
"""Find Guqinizer rows whose visible current draft is stale vs paired Base.

The final message stream can contain a newer Fingering row together with an
older no-op Guqinizer row.  This helper compares the Guqinizer user prompt to
the Base accepted plan recorded in the private audit and writes the exact
intermediate plans needed for a Guqinizer-only rerun.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def parse_current_user(content: str) -> dict[int, str]:
    start = content.find("【当前段")
    if start < 0:
        return {}
    end = content.find("\n更早段", start)
    if end < 0:
        end = len(content)
    values: dict[int, str] = {}
    lines = content[start:end].splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        fields = line.split("｜")
        if not fields or not re.fullmatch(r"\d+", fields[0].strip()):
            line_index += 1
            continue
        index = int(fields[0].strip())
        value = fields[-1].strip() if len(fields) >= 5 else ""
        # Long jianzi text is rendered across physical lines inside the final
        # brackets.  Rejoin it before comparing against accepted_plan.
        while value.startswith("[") and not value.endswith("]") and line_index + 1 < len(lines):
            line_index += 1
            value += "\n" + lines[line_index]
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if value in {"", "—", "-", "减字待填写"}:
            value = "" if value != "减字待填写" else value
        values[index] = value
        line_index += 1
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    messages = {row["sample_id"]: row for row in read_jsonl(args.messages)}
    audits = {row["sample_id"]: row for row in read_jsonl(args.audit)}
    source = {row["trajectory_id"]: row for row in read_jsonl(args.source)}
    base_plans: dict[str, dict] = {}
    for sample_id, row in audits.items():
        if not sample_id.endswith("-fingering_agent-teacher-tools"):
            continue
        base_id = sample_id[:-len("-fingering_agent-teacher-tools")]
        plan = (row.get("teacher_private") or {}).get("accepted_plan")
        if isinstance(plan, dict) and isinstance(plan.get("actions"), list):
            base_plans[base_id] = plan

    affected: list[dict] = []
    for sample_id, row in messages.items():
        suffix = "-guqinization-teacher-tools"
        if not sample_id.endswith(suffix):
            continue
        base_id = sample_id[:-len(suffix)]
        plan = base_plans.get(base_id)
        if not plan or base_id not in source:
            continue
        # A wholly empty annotation is not a supervision target and should
        # not acquire a new two-stage trajectory during repair.
        reference_actions = (source[base_id].get("reference_plan") or {}).get("actions") or []
        if not any(str(action.get("jianzi_text") or "") for action in reference_actions):
            continue
        user = next((m.get("content") or "" for m in row.get("messages", [])
                     if m.get("role") == "user"), "")
        actual = parse_current_user(user)
        diffs = []
        for action in plan["actions"]:
            index = int(action["source_index"])
            expected = str(action.get("jianzi_text") or "")
            if expected in {"—", "-"}:
                expected = ""
            got = actual.get(index, "")
            if expected != got:
                diffs.append({"source_index": index, "expected": expected, "actual": got})
        if not diffs:
            continue
        src = source[base_id]
        affected.append({
            "trajectory_id": base_id,
            "score_key": src["score_key"],
            "phrase_id": src["phrase_id"],
            "normalized_tuning": src["input"]["normalized_tuning"],
            "plan": plan,
            "verification": (audits.get(base_id + "-fingering_agent-teacher-tools") or {})
                .get("verification", {}),
            "diffs": diffs,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stale_guqinizer_ids.txt").write_text(
        "".join(item["trajectory_id"] + "\n" for item in affected), encoding="utf-8"
    )
    with (args.output_dir / "fingering_intermediates.jsonl").open("w", encoding="utf-8", newline="\n") as out:
        for item in affected:
            out.write(json.dumps({k: item[k] for k in (
                "trajectory_id", "score_key", "phrase_id", "normalized_tuning", "plan", "verification"
            )}, ensure_ascii=False) + "\n")
    (args.output_dir / "stale_guqinizer_report.json").write_text(
        json.dumps({
            "affected_count": len(affected),
            "affected_ids": [item["trajectory_id"] for item in affected],
            "diff_cell_count": sum(len(item["diffs"]) for item in affected),
            "examples": affected[:10],
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"affected_count": len(affected),
                      "diff_cell_count": sum(len(item["diffs"]) for item in affected)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
