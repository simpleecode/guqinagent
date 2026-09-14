#!/usr/bin/env python3
"""统一训练数据质量筛选入口。

把过去分散的音高资格、源谱空减字、整条全空和指定 phrase 排除集中在
一个命令中。每个子命令都写出可追溯的 JSON/文本报告，且所有按 phrase
的筛选均保持 Fingering 与 Guqinizer 成对，避免产生半条训练样本。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EMPTY = {"", "—", "-", "null", "None", "[空]", "[减字待填写]"}
STAGES = ("fingering", "guqinizer")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_md_row(line: str) -> list[str]:
    """Split Markdown cells without mistaking an escaped bar for a separator."""
    parts, buf, escaped = [], [], False
    for char in line:
        if char == "|" and not escaped:
            parts.append("".join(buf).strip()); buf = []
        else:
            buf.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    parts.append("".join(buf).strip())
    return parts


def mapped_score_record(path: Path) -> dict[str, Any]:
    """Classify a mapped score by coverage and a terminal blank run."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = []
    for line in text.splitlines():
        if not line.startswith("|") or not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = split_md_row(line)
        if cells and cells[0] == "": cells = cells[1:]
        if cells and cells[-1] == "": cells = cells[:-1]
        if len(cells) >= 6 and cells[1] not in {"|", r"\|"}:
            rows.append({"index": int(cells[0]), "jianzi": cells[-1], "empty": cells[-1] in EMPTY})
    nonempty = sum(not row["empty"] for row in rows)
    trailing = next((i for i, row in enumerate(reversed(rows)) if not row["empty"]), len(rows))
    ratio = nonempty / len(rows) if rows else 0.0
    return {"score_key": path.parent.name, "path": str(path), "notes": len(rows), "nonempty": nonempty,
            "nonempty_ratio": ratio, "trailing_empty_notes": trailing,
            "last_nonempty_index": rows[-trailing - 1]["index"] if trailing < len(rows) else None,
            "classification": "drop_score" if ratio < .5 else "trim_trailing" if trailing >= 10 else "keep"}


def mapped_path(key: str) -> Path:
    """Resolve a score's mapped source without duplicating final/round2 logic."""
    final = ROOT / "ABC_J/final" / key / "jianpu_jianzi_mapped.md"
    return final if final.exists() else ROOT / "ABC_J/round2/final" / key / "jianpu_jianzi_mapped.md"


def cmd_source_audit(args: argparse.Namespace) -> None:
    sources = list(read_jsonl(args.source))
    keys = sorted({str(row["score_key"]) for row in sources})
    records = []
    for key in keys:
        records.append(mapped_score_record(mapped_path(key)))
    summary = {"training_score_keys": len(records), "drop_score_keys": [r["score_key"] for r in records if r["classification"] == "drop_score"],
               "trim_trailing_keys": [r["score_key"] for r in records if r["classification"] == "trim_trailing"],
               "empty_definition": sorted(EMPTY), "ratio_threshold": .5, "trailing_threshold_notes": 10}
    summary.update({"drop_score_count": len(summary["drop_score_keys"]), "trim_trailing_count": len(summary["trim_trailing_keys"]), "keep_count": len(records) - len(summary["drop_score_keys"]) - len(summary["trim_trailing_keys"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def canonical_pitch_class(item: dict[str, Any]) -> dict[str, Any]:
    module = load_module(ROOT / "scripts/audit_training_phase_pitch_parse_stats.py", "pitch_stats_for_filter")
    return module.classify(module.build_pitch_audit(item, item.get("reference_plan") or {}, 50.0))


def cmd_pitch_select(args: argparse.Namespace) -> None:
    selected, total = [], 0
    for split in args.splits:
        path = args.inferred_dir / f"inferred_trajectories_{split}.jsonl"
        if not path.exists(): continue
        for item in read_jsonl(path):
            total += 1
            result = canonical_pitch_class(item)
            keep = bool(result["parseable_and_at_least_one_match"]) if args.require_match else bool(result["no_parseable_pitch"] or result["parseable_and_at_least_one_match"])
            if keep: selected.append(str(item["trajectory_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pitch_eligible_phrase_ids.txt").write_text("".join(f"{value}\n" for value in selected), encoding="utf-8")
    payload = {"schema_version": "training-data-filter-1.0", "rule": "pitch_eligible", "inputs": {"inferred_dir": str(args.inferred_dir), "splits": args.splits}, "require_match": args.require_match, "source_phrase_count": total, "selected_phrase_count": len(selected)}
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


def text_of(action: dict[str, Any]) -> str:
    value = action.get("jianzi_text")
    return "" if value is None else str(value).strip()


def is_plain_attack(action: dict[str, Any]) -> bool:
    """Conservative definition used by the historical five-blank cleanup.

    Compound/continuation marks can legitimately cover following events, so
    they are intentionally excluded.  Only a nonempty ordinary attack starts
    a suspicious run.
    """
    value = text_of(action)
    complex_marks = "掐撮三声再作走手走猱泛起泛止复双弹连涓打"
    return bool(action.get("attack", True) and value and not any(mark in value for mark in complex_marks))


def cmd_basic_blank_audit(args: argparse.Namespace) -> None:
    flagged = []
    for item in read_jsonl(args.source):
        actions = (item.get("reference_plan") or {}).get("actions") or []
        # ``attack`` in a parsed empty reference row is false, even when the
        # underlying Jianpu row is a sounding note.  Determine soundingness
        # from the source notes instead, and ignore only bars/rests/omissions.
        notes = {int(note["index"]): note for note in (item.get("input") or {}).get("notes_without_jianzi", []) if note.get("index") is not None}
        for pos, action in enumerate(actions):
            if not is_plain_attack(action): continue
            following = []
            for row in actions[pos + 1:]:
                note = notes.get(int(row.get("source_index", -1))) or {}
                if note.get("jianpu") in {"|", "0", "0（休止）"} or note.get("notation_omitted"):
                    continue
                following.append(row)
                if len(following) == args.min_following_blanks:
                    break
            if len(following) == args.min_following_blanks and all(not text_of(row) for row in following):
                flagged.append({"trajectory_id": item["trajectory_id"], "score_key": item.get("score_key"), "start_source_index": action.get("source_index"), "blank_source_indices": [row.get("source_index") for row in following]})
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": "basic-attack-blank-run-1.0", "min_following_blanks": args.min_following_blanks, "count": len(flagged), "phrases": flagged}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(flagged), "output": str(args.output)}, ensure_ascii=False))


def ids_from_file(path: Path) -> set[str]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "all_empty_ids" in payload:
            return {str(value) for value in payload["all_empty_ids"]}
        rows = payload.get("phrases", payload) if isinstance(payload, dict) else payload
        return {str(row["trajectory_id"]) for row in rows}
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def row_id(row: dict[str, Any]) -> str:
    if row.get("trajectory_id"): return str(row["trajectory_id"])
    return re.sub(r"-(?:fingering_agent|guqinization)-teacher-tools$", "", str(row.get("sample_id") or ""))


def cmd_filter_corpus(args: argparse.Namespace) -> None:
    drop = set().union(*(ids_from_file(path) for path in args.exclude))
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {"schema_version": "training-data-filter-1.0", "rule": "phrase_id_exclusion", "dropped_phrase_ids": sorted(drop), "files": {}}
    for source in args.input.iterdir():
        target = args.output / source.name
        if not source.is_file() or source.suffix != ".jsonl":
            if source.is_file(): shutil.copy2(source, target)
            continue
        rows = list(read_jsonl(source)); kept = [row for row in rows if row_id(row) not in drop]
        write_jsonl(target, kept); report["files"][source.name] = {"before": len(rows), "after": len(kept), "dropped": len(rows)-len(kept)}
    (args.output / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["files"], ensure_ascii=False))


def cmd_apply_source_audit(args: argparse.Namespace) -> None:
    """Apply source-score exclusions while preserving unaffected rows bytewise."""
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    drops = {row["score_key"] for row in audit["records"] if row["classification"] == "drop_score"}
    cutoffs = {row["score_key"]: int(row["last_nonempty_index"]) for row in audit["records"] if row["classification"] == "trim_trailing" and row.get("last_nonempty_index") is not None}
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report: dict[str, Any] = {"schema_version": "training-data-filter-1.0", "rule": "mapped_source_quality", "dropped_score_keys": sorted(drops), "trim_cutoffs": cutoffs, "splits": {}}
    for split in ("train", "validation", "test"):
        old = list(read_jsonl(args.old_dir / f"inferred_trajectories_{split}.jsonl"))
        rebuilt = list(read_jsonl(args.rebuilt_dir / f"inferred_trajectories_{split}.jsonl"))
        rebuilt_by_id = {str(row["trajectory_id"]): row for row in rebuilt}
        kept, counts = [], Counter(old=len(old))
        for row in old:
            key = str(row.get("score_key")); bounds = (row.get("input") or {}).get("event_range") or {}
            start, end = bounds.get("start"), bounds.get("end_exclusive")
            if key in drops: counts["drop_score"] += 1; continue
            if key in cutoffs and isinstance(start, int) and isinstance(end, int):
                if start > cutoffs[key]: counts["drop_trailing_phrase"] += 1; continue
                if end - 1 > cutoffs[key]:
                    replacement = rebuilt_by_id.get(str(row["trajectory_id"]))
                    if replacement is None: raise ValueError(f"missing rebuilt crossing phrase: {row['trajectory_id']}")
                    kept.append(replacement); counts["replace_crossing_phrase"] += 1; continue
            kept.append(row); counts["keep"] += 1
        write_jsonl(args.output / f"inferred_trajectories_{split}.jsonl", kept)
        report["splits"][split] = dict(counts)
    (args.output / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("source-audit", help="审计低覆盖率与连续空尾"); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_source_audit)
    p = sub.add_parser("pitch-select", help="选择至少一音命中的 phrase"); p.add_argument("--inferred-dir", type=Path, required=True); p.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=("train",)); p.add_argument("--require-match", action="store_true"); p.add_argument("--output-dir", type=Path, required=True); p.set_defaults(func=cmd_pitch_select)
    p = sub.add_parser("basic-blank-audit", help="检测基础单发后连续空减字"); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--min-following-blanks", type=int, default=5); p.set_defaults(func=cmd_basic_blank_audit)
    p = sub.add_parser("filter-corpus", help="按 phrase ID 成对移除教师轨迹"); p.add_argument("--input", type=Path, required=True); p.add_argument("--exclude", type=Path, action="append", required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_filter_corpus)
    p = sub.add_parser("apply-source-audit", help="应用低覆盖率与连续空尾源谱规则"); p.add_argument("--audit", type=Path, required=True); p.add_argument("--old-dir", type=Path, required=True); p.add_argument("--rebuilt-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(func=cmd_apply_source_audit)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args(); args.func(args)
