#!/usr/bin/env python3
"""Audit newly scalar-auditable Guqinizer walking gestures.

The pitch auditor used to skip every standalone left-hand walk.  This report
isolates rows that were previously forced into that skip path but now have an
inherited string and explicit endpoint, so their scalar pitch can be checked.
It groups repeated phrase surfaces so review is not dominated by repetitions.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from ABC_J.scripts.generate_teacher_tool_trajectories import pitch_audit_notes
from scripts.audit_jianpu_jianzi_pitch import (
    RIGHT_HAND_TECHNIQUES,
    audit,
    first_token,
)


ROOT = Path(__file__).resolve().parents[1]
WALK_PREFIXES = ("绰", "注", "淌", "浒", "引上", "上", "下", "进", "退")
DEFAULT_SOURCE = ROOT / "ABC_J/agent_training/inferred_gqs_v12_tuningfix_20260914/inferred_trajectories_train.jsonl"
DEFAULT_AUDIT = ROOT / "ABC_J/agent_training/messages_glm_full_two_stage_harmonicstatefix_lipitchfix_p0063harmonicwarning_statewarning38_harmonicparsefix_20260915/teacher_trajectory_audit.jsonl"


def rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for value in values:
            target.write(json.dumps(value, ensure_ascii=False) + "\n")


def render_review_html(summary: dict, groups: list[dict]) -> str:
    """Create a compact human review page, grouped by repeated pattern."""
    blocks = []
    for number, group in enumerate(groups, 1):
        representative = group["representative"]
        rows = "".join(
            "<tr>"
            f"<td>{html.escape(str(row.get('source_index')))}</td>"
            f"<td>{html.escape(str(row.get('jianpu')))}</td>"
            f"<td>{html.escape(str(row.get('jianzi')))}</td>"
            f"<td>{html.escape(json.dumps(row.get('pairs') or [], ensure_ascii=False))}</td>"
            "</tr>"
            for row in representative["rows"]
        )
        phrase_ids = "、".join(group["trajectory_ids"])
        blocks.append(
            "<details class='group'>"
            f"<summary>#{number}｜重复 {group['phrase_count']} 条｜"
            f"代表 {html.escape(representative['trajectory_id'])}</summary>"
            "<table><thead><tr><th>原始序号</th><th>简谱</th><th>走手减字</th><th>音高配对</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
            f"<p><b>同一模式的 phrase：</b>{html.escape(phrase_ids)}</p>"
            "</details>"
        )
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>Guqinizer 走手音高复核</title><style>
body{{max-width:1200px;margin:24px auto;padding:0 18px;background:#181818;color:#e8e4dc;font:15px/1.55 system-ui,'Microsoft YaHei',sans-serif}}
h1{{font-size:23px}}.summary{{background:#252525;padding:12px 16px;border-radius:8px}}details{{margin:9px 0;background:#252525;border:1px solid #444;border-radius:7px}}summary{{cursor:pointer;padding:9px 12px;font-weight:600}}table{{border-collapse:collapse;width:100%;margin:0 12px 10px;width:calc(100% - 24px)}}th,td{{border-bottom:1px solid #494949;padding:7px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{color:#c9dfef}}p{{padding:0 12px;color:#c9c4b9}}code{{font-family:ui-monospace,monospace}}</style>
<h1>Guqinizer 新增走手音高审计｜模式聚类复核</h1>
<div class='summary'>phrase 记录：{summary['newly_mismatched_walk_phrases']}；不匹配行：{summary['newly_mismatched_walk_rows']}；去重后的模式：{summary['unique_mismatch_patterns']}。<br>每个折叠项只展示一个代表 phrase；展开后可见所有重复 phrase ID。</div>
{''.join(blocks)}</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--teacher-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_by_id = {str(row["trajectory_id"]): row for row in rows(args.source)}
    accepted: dict[str, dict] = {}
    suffix = "-guqinization-teacher-tools"
    for row in rows(args.teacher_audit):
        sample_id = str(row.get("sample_id") or "")
        plan = (row.get("teacher_private") or {}).get("accepted_plan")
        if sample_id.endswith(suffix) and isinstance(plan, dict):
            accepted[sample_id.removesuffix(suffix)] = plan

    by_score: dict[str, list[dict]] = defaultdict(list)
    for trajectory_id, plan in accepted.items():
        source = source_by_id.get(trajectory_id)
        if source is None:
            continue
        item = dict(source)
        item["reference_plan"] = plan
        by_score[str(item["score_key"])].append(item)

    affected: list[dict] = []
    for score_key, phrases in by_score.items():
        historical: dict[tuple[str, str], dict] = {}
        for item in sorted(phrases, key=lambda row: int(row["input"]["event_range"]["start"])):
            # ``pitch_audit_notes`` deliberately replays all preceding phrases
            # to reconstruct left-hand / harmonic state.  Its audit report
            # therefore contains both history and this phrase.  Only attribute
            # rows belonging to the current phrase to this trajectory; without
            # this filter, one early mismatch is duplicated into every later
            # phrase of the same score.
            current_indices = {
                int(note["index"])
                for note in item["input"].get("notes_without_jianzi") or []
                if note.get("index") is not None
            }
            action_map = {
                int(action["source_index"]): action
                for action in item["reference_plan"].get("actions") or []
                if action.get("source_index") is not None
            }
            report = audit({
                "metadata": item["input"].get("metadata") or {},
                "open_midi": (item["input"].get("normalized_tuning") or {}).get("open_midi"),
                "notes": pitch_audit_notes(item, action_map, historical=historical),
            }, 50.0)
            for detail in report.get("details") or []:
                if detail.get("index") not in current_indices:
                    continue
                text = str(detail.get("jianzi") or "")
                if (not text.startswith(WALK_PREFIXES)
                        or first_token(text, RIGHT_HAND_TECHNIQUES) is not None
                        or detail.get("status") == "skipped"):
                    continue
                # This exact surface had been forcibly skipped by the old
                # audit branch.  It now has a scalar endpoint result.
                affected.append({
                    "trajectory_id": item["trajectory_id"],
                    "score_key": item["score_key"],
                    "phrase_id": item["phrase_id"],
                    "source_index": detail.get("index"),
                    "jianpu": detail.get("jianpu"),
                    "jianzi": text,
                    "status": detail.get("status"),
                    "pairs": detail.get("pairs") or [],
                })
            historical[(score_key, str(item["phrase_id"]))] = item

    mismatch = [row for row in affected if row["status"] == "mismatched"]
    phrase_rows: dict[str, list[dict]] = defaultdict(list)
    for row in mismatch:
        phrase_rows[row["trajectory_id"]].append(row)

    # Group on the actionable mismatch pattern, rather than whole phrase
    # identity.  Repeated musical material therefore appears once with a
    # multiplicity count and representative phrase.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for trajectory_id, rows_for_phrase in phrase_rows.items():
        signature_rows = [
            (row["jianpu"], row["jianzi"], tuple(
                round(float(pair.get("delta_cents") or 0), 1)
                for pair in row["pairs"]
            ))
            for row in rows_for_phrase
        ]
        signature = hashlib.sha256(
            json.dumps(signature_rows, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:16]
        grouped[signature].append({
            "trajectory_id": trajectory_id,
            "rows": rows_for_phrase,
        })

    groups = []
    for signature, members in grouped.items():
        representative = members[0]
        groups.append({
            "signature": signature,
            "phrase_count": len(members),
            "representative": representative,
            "trajectory_ids": [member["trajectory_id"] for member in members],
        })
    groups.sort(key=lambda group: (-group["phrase_count"], group["signature"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_counts = Counter(row["status"] for row in affected)
    prefix_counts = Counter(
        next(prefix for prefix in WALK_PREFIXES if row["jianzi"].startswith(prefix))
        for row in affected
    )
    summary = {
        "source_phrase_count": len(source_by_id),
        "guqinizer_plan_count": len(accepted),
        "joined_plan_count": sum(len(values) for values in by_score.values()),
        "newly_auditable_walk_rows_by_status": dict(status_counts),
        "newly_auditable_walk_phrases": len({row["trajectory_id"] for row in affected}),
        "newly_mismatched_walk_rows": len(mismatch),
        "newly_mismatched_walk_phrases": len(phrase_rows),
        "unique_mismatch_patterns": len(groups),
        "newly_auditable_rows_by_prefix": dict(prefix_counts),
        "review_rule": "Rows are standalone walks with no right-hand attack, explicit text prefix, and a newly computed scalar endpoint; they were forced to skip under the old audit branch.",
    }
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "affected_mismatch_rows.jsonl", mismatch)
    write_jsonl(args.output_dir / "mismatch_pattern_groups.jsonl", groups)
    (args.output_dir / "affected_trajectory_ids.txt").write_text(
        "\n".join(sorted(phrase_rows)) + "\n", encoding="utf-8", newline="\n"
    )
    (args.output_dir / "review.html").write_text(
        render_review_html(summary, groups), encoding="utf-8", newline="\n"
    )
    print(json.dumps({**summary, "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
