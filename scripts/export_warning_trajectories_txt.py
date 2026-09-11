"""Export selected warning trajectories as human-readable UTF-8 text."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def readable(value: object) -> str:
    if value is None:
        return "null"
    text = str(value)
    # Nested tool JSON commonly contains literal escaped line breaks/quotes.
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace('\\"', '"')
    )


def parse_annotation(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in text.splitlines():
        if not line.startswith("音｜"):
            continue
        try:
            payload = json.loads(line.split("音｜", 1)[1])
            if isinstance(payload, list) and payload:
                result[int(payload[0])] = "" if len(payload) < 3 or payload[2] is None else str(payload[2])
        except (ValueError, json.JSONDecodeError):
            continue
    return result


def build_comparison(sample_id: str, source_path: Path, run_root: Path) -> list[str]:
    base_id = re.sub(r"-(?:fingering_agent|guqinization)-teacher-tools$", "", sample_id)
    source_record = None
    with source_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                candidate = json.loads(raw)
                if candidate.get("trajectory_id") == base_id:
                    source_record = candidate
                    break
    if source_record is None:
        return ["对比表｜未找到源 phrase"]

    stage = "fingering" if "fingering_agent" in sample_id else "guqinization"
    base_actions: dict[int, str] = {}
    worker_root = run_root / ".parallel_workers"
    intermediate_paths = list(worker_root.glob("*/output/fingering_intermediates.jsonl"))
    direct_intermediate = run_root / "fingering_intermediates.jsonl"
    if direct_intermediate.exists():
        intermediate_paths.append(direct_intermediate)
    for path in intermediate_paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    candidate = json.loads(raw)
                    if candidate.get("trajectory_id") == base_id:
                        base_actions = {
                            int(a["source_index"]): str(a.get("jianzi_text", ""))
                            for a in candidate.get("plan", {}).get("actions", [])
                            if isinstance(a, dict) and "source_index" in a
                        }
                        break
        if base_actions:
            break

    final_actions = dict(base_actions)
    if stage == "guqinization":
        audit_paths = list(worker_root.glob("*/output/teacher_trajectory_audit.jsonl"))
        direct_audit = run_root / "teacher_trajectory_audit.jsonl"
        if direct_audit.exists():
            audit_paths.append(direct_audit)
        for path in audit_paths:
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    if raw.strip():
                        candidate = json.loads(raw)
                        if candidate.get("sample_id") != sample_id:
                            continue
                        for patch in candidate.get("teacher_private", {}).get("accepted_patches", []):
                            idx = patch.get("source_index")
                            after = patch.get("after", {})
                            if idx is not None and isinstance(after, dict) and "jianzi_text" in after:
                                final_actions[int(idx)] = str(after.get("jianzi_text") or "")
                        break
            if final_actions != base_actions:
                break

    notes = source_record.get("input", {}).get("notes_without_jianzi", [])
    reference_actions = source_record.get("reference_plan", {}).get("actions", [])
    annotation = {
        int(action["source_index"]): str(action.get("jianzi_text") or "")
        for action in reference_actions
        if isinstance(action, dict) and "source_index" in action
    }
    rows = ["", "最终结果 vs 标注版本｜空字符串显示为〈空〉", "音序｜简谱｜Base最终｜Guqinizer最终｜标注"]
    for note in notes:
        idx = int(note.get("index"))
        event_index = note.get("event_index")
        display_index = str(event_index) if event_index is not None else "小节线"
        jp = str(note.get("jianpu", ""))
        base = base_actions.get(idx, "")
        final = final_actions.get(idx, base) if stage == "guqinization" else "〈未运行〉"
        ann = annotation.get(idx, "")
        rows.append(f"{display_index}｜{jp}｜{base or '〈空〉'}｜{final or '〈空〉'}｜{ann or '〈空〉'}")
    return rows


def render(record: dict, source_path: Path | None = None, run_root: Path | None = None) -> str:
    lines = [
        "轨迹文本查看｜状态｜已读",
        f"样本ID｜{record.get('sample_id', '')}",
        f"阶段｜{record.get('agent_stage', '')}",
        "说明｜工具结果中的转义换行已展开为真实换行。",
        "",
    ]
    if source_path and run_root:
        lines.extend(build_comparison(record.get("sample_id", ""), source_path, run_root))
        lines.append("")
    for n, message in enumerate(record.get("messages", []), 1):
        role = message.get("role", "")
        lines.append(f"【回合 {n}｜{role}】")
        content = message.get("content")
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False, indent=2)
        lines.append(readable(content))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("sample_ids", nargs="+", help="exact sample_id values")
    parser.add_argument("--source", type=Path, help="source inferred trajectories JSONL")
    parser.add_argument("--run-root", type=Path, help="parallel run root containing worker outputs")
    args = parser.parse_args()

    wanted = set(args.sample_ids)
    found: dict[str, dict] = {}
    with args.input.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            sample_id = record.get("sample_id")
            if sample_id in wanted:
                found[sample_id] = record

    missing = wanted - found.keys()
    if missing:
        raise SystemExit(f"missing sample_id: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, record in found.items():
        path = args.output_dir / f"{sample_id}.txt"
        path.write_text(render(record, args.source, args.run_root), encoding="utf-8", newline="\n")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
