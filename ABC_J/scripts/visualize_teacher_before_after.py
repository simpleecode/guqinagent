#!/usr/bin/env python3
"""Render a before/after Guqinizer trajectory comparison page."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read_map(path: Path) -> dict[str, dict]:
    return {row["sample_id"]: row for row in
            (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())}


def read_source_map(path: Path) -> dict[str, dict[int, dict]]:
    """Load source_index/event_index and reference text for each trajectory."""
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    score_notes: dict[str, dict[int, dict]] = {}
    for row in rows:
        score_key = str(row.get("score_key") or row.get("score_family_id") or "")
        collected = score_notes.setdefault(score_key, {})
        for note in ((row.get("input") or {}).get("notes_without_jianzi") or []):
            if note.get("index") is not None:
                collected[int(note["index"])] = note
    derived_events: dict[str, dict[int, int]] = {}
    for score_key, notes in score_notes.items():
        event_index = 0
        mapping: dict[int, int] = {}
        for source_index, note in sorted(notes.items()):
            is_bar = (str(note.get("jianpu") or "").strip() == "|"
                      or str(note.get("abc") or "").strip() == "|"
                      or str(note.get("duration") or "").strip() == "小节线")
            if is_bar:
                continue
            mapping[source_index] = int(note.get("event_index", event_index))
            event_index += 1
        derived_events[score_key] = mapping
    result = {}
    for row in rows:
        trajectory_id = row.get("trajectory_id")
        if not trajectory_id:
            continue
        score_key = str(row.get("score_key") or row.get("score_family_id") or "")
        notes = ((row.get("input") or {}).get("notes_without_jianzi") or [])
        mapped = {}
        for note in notes:
            if note.get("index") is None:
                continue
            source_index = int(note["index"])
            visible = dict(note)
            if not (str(note.get("jianpu") or "").strip() == "|"
                    or str(note.get("abc") or "").strip() == "|"):
                visible["event_index"] = derived_events.get(score_key, {}).get(
                    source_index, source_index
                )
            mapped[source_index] = visible
        result[str(trajectory_id)] = mapped
    return result


def text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def assistant_texts(row: dict | None) -> list[str]:
    if not row:
        return []
    return [text(message.get("content")) for message in row.get("messages", [])
            if message.get("role") == "assistant" and text(message.get("content"))]


def calls(row: dict | None) -> int:
    return sum(len(message.get("tool_calls") or [])
               for message in (row or {}).get("messages", [])
               if message.get("role") == "assistant")


def warning_turns(row: dict | None) -> int:
    return sum(
        ":warning:" in text(message.get("content"))
        or "警告｜" in text(message.get("content"))
        for message in (row or {}).get("messages", [])
        if message.get("role") == "tool"
    )


def canonical(value) -> str:
    return "".join(text(value).split()).translate(str.maketrans("一二三四五六七", "1234567"))


def target_map(row: dict | None) -> dict[int, str]:
    private = (row or {}).get("teacher_private") or {}
    return {
        int(patch["source_index"]): patch.get("after", {}).get("jianzi_text") or ""
        for patch in private.get("target_patches", [])
        if patch.get("source_index") is not None
    }


def target_score(row: dict | None) -> tuple[int, int]:
    accepted = plan(row)
    targets = target_map(row)
    return sum(canonical(accepted.get(index)) == canonical(value)
               for index, value in targets.items()), len(targets)


def plan(row: dict | None) -> dict[int, str]:
    actions = ((row or {}).get("teacher_private") or {}).get("accepted_plan", {}).get("actions", [])
    return {int(action["source_index"]): text(action.get("jianzi_text"))
            for action in actions if action.get("source_index") is not None}


def esc(value) -> str:
    return html.escape(text(value), quote=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-input", type=Path,
                        help="Optional inferred trajectory JSONL for visible event_index and labels")
    parser.add_argument("--sample-id-file", type=Path,
                        help="Optional newline-delimited sample IDs to include")
    args = parser.parse_args()
    before = read_map(args.before / "messages_train.jsonl")
    after = read_map(args.after / "messages_train.jsonl")
    before_audit = read_map(args.before / "teacher_trajectory_audit.jsonl")
    after_audit = read_map(args.after / "teacher_trajectory_audit.jsonl")
    source_map = read_source_map(args.source_input) if args.source_input else {}
    # The after directory is the explicit comparison selection; do not pull
    # unrelated rows from a full before directory into the page.
    ids = sorted(set(after))
    if args.sample_id_file:
        selected = {
            line.strip() for line in args.sample_id_file.open(encoding="utf-8")
            if line.strip()
        }
        ids = [sample_id for sample_id in ids if sample_id in selected]
    articles = []
    changed_total = 0
    old_reason_total = new_reason_total = 0
    old_hits = old_targets = new_hits = new_targets = 0
    old_warning_total = new_warning_total = 0
    for sid in ids:
        old, new = before.get(sid), after.get(sid)
        old_a, new_a = before_audit.get(sid), after_audit.get(sid)
        old_plan, new_plan = plan(old_a), plan(new_a)
        targets = target_map(new_a or old_a)
        base_id = sid.rsplit("-guqinization-teacher-tools", 1)[0]
        source_notes = source_map.get(base_id, {})
        indexes = sorted(set(old_plan) | set(new_plan) | set(targets))
        changed = [i for i in indexes if old_plan.get(i, "") != new_plan.get(i, "")]
        changed_total += len(changed)
        rows = []
        for index in indexes:
            old_value, new_value = old_plan.get(index, ""), new_plan.get(index, "")
            note = source_notes.get(index, {})
            event_index = note.get("event_index")
            display_index = event_index if event_index is not None else index
            target_value = targets.get(index, "")
            cls = "changed" if old_value != new_value else "same"
            rows.append(
                f'<tr class="{cls}"><th>{esc(display_index)}</th>'
                f'<td>{esc(target_value) or "（无修改标注）"}</td>'
                f'<td>{esc(old_value) or "（空）"}</td>'
                f'<td>{esc(new_value) or "（空）"}</td></tr>'
            )
        old_reason = "\n\n".join(assistant_texts(old)) or "（无公开 assistant 内容）"
        new_reason = "\n\n".join(assistant_texts(new)) or "（无公开 assistant 内容）"
        old_score, old_count = target_score(old_a)
        new_score, new_count = target_score(new_a)
        old_reason_total += len(old_reason)
        new_reason_total += len(new_reason)
        old_hits += old_score
        old_targets += old_count
        new_hits += new_score
        new_targets += new_count
        old_warning_total += warning_turns(old)
        new_warning_total += warning_turns(new)
        label = sid.rsplit("-guqinization-teacher-tools", 1)[0]
        articles.append(
            f'<details class="phrase"><summary>{esc(label)}｜修改行数：{len(changed)}｜'
            f'标注命中：旧 {old_score}/{old_count} → 新 {new_score}/{new_count}｜'
            f'推理字数：旧 {len(old_reason)} → 新 {len(new_reason)}｜'
            f'警告轮次：旧 {warning_turns(old)} → 新 {warning_turns(new)}｜'
            f'工具调用：旧 {calls(old)} → 新 {calls(new)}</summary>'
            f'<div class="grid"><section><h3>减字差异</h3><div class="table-wrap"><table>'
            f'<thead><tr><th>音序</th><th>标注</th><th>旧版 Guqinizer</th><th>新版 GLM Guqinizer</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></section>'
            f'<section><h3>公开 reasoning｜旧版</h3><pre>{esc(old_reason)}</pre>'
            f'<h3>公开 reasoning｜新版 GLM</h3><pre>{esc(new_reason)}</pre></section></div></details>'
        )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guqinizer 旧版与新版对比</title><style>
:root{{--bg:#f7f4ec;--fg:#27241f;--muted:#716b61;--surface:#fffdf8;--line:#d9d1c3;--accent:#8d3f2d;--soft:#f1e7d9;--same:#fffdf8;--changed:#fff0e5}}@media(prefers-color-scheme:dark){{:root{{--bg:#181714;--fg:#eee9df;--muted:#aaa397;--surface:#22201c;--line:#49443c;--accent:#e4967e;--soft:#332821;--same:#22201c;--changed:#422d25}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:24px;margin:0 0 4px}}p{{color:var(--muted)}}.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}.stat{{background:var(--surface);border:1px solid var(--line);padding:9px 13px;border-radius:8px}}.stat b{{font-size:18px}}details{{border-top:1px solid var(--line);padding:10px 0}}summary{{cursor:pointer;font-weight:600}}.grid{{display:grid;grid-template-columns:minmax(360px,1fr) minmax(360px,1fr);gap:18px;margin-top:10px}}h3{{font-size:15px;font-weight:600;margin:10px 0 6px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:var(--surface)}}th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}}thead th{{color:var(--muted);font-weight:500}}tr.changed{{background:var(--changed)}}tr.same{{background:var(--same)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--surface);border-left:3px solid var(--line);padding:10px;margin:0 0 10px}}.phrase{{margin-left:10px}}@media(max-width:800px){{main{{padding:14px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Guqinizer 旧版与新版对比</h1><p>新版为 GLM-5.3；橙色行表示两版最终接受的减字不同。</p><div class="stats"><div class="stat">对比 phrase<br><b>{len(ids)}</b></div><div class="stat">标注命中<br><b>{old_hits}/{old_targets} → {new_hits}/{new_targets}</b></div><div class="stat">推理总字数<br><b>{old_reason_total} → {new_reason_total}</b></div><div class="stat">警告轮次<br><b>{old_warning_total} → {new_warning_total}</b></div><div class="stat">减字差异行<br><b>{changed_total}</b></div></div>{"".join(articles)}</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "phrases": len(ids), "changed_rows": changed_total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
