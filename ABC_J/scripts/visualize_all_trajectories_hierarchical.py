#!/usr/bin/env python3
"""Build a two-level, searchable review page for the complete agent set."""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_jianpu_jianzi_pitch import audit as audit_pitch

# Keep this viewer independent from the optional agent runtime (langgraph and
# model dependencies); it only needs the display marker text.
OMITTED_PLACEHOLDER = "无（由于是再作部分，省略）"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def attach_derived_event_indices(rows: list[dict]) -> list[dict]:
    """Fill missing public event indexes from each score's non-bar rows."""
    score_notes: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        score_key = str(row.get("score_key") or row.get("score_family_id") or "")
        for note in ((row.get("input") or {}).get("notes_without_jianzi") or []):
            if note.get("index") is not None:
                score_notes[score_key][int(note["index"])] = note
    event_maps: dict[str, dict[int, int]] = {}
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
        event_maps[score_key] = mapping
    enriched = []
    for row in rows:
        copied = dict(row)
        copied_input = dict(row.get("input") or {})
        score_key = str(row.get("score_key") or row.get("score_family_id") or "")
        copied_notes = []
        for note in copied_input.get("notes_without_jianzi") or []:
            visible = dict(note)
            if note.get("index") is not None:
                source_index = int(note["index"])
                if source_index in event_maps.get(score_key, {}):
                    visible["event_index"] = event_maps[score_key][source_index]
            copied_notes.append(visible)
        copied_input["notes_without_jianzi"] = copied_notes
        copied["input"] = copied_input
        enriched.append(copied)
    return enriched


def action_text(action: dict | None) -> str:
    if not action:
        return ""
    value = action.get("jianzi_text")
    if value is None:
        value = action.get("text")
    return "" if value is None else str(value)


def parse_gqs(value: str | None) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for line in (value or "").splitlines():
        if not line.startswith("音｜"):
            continue
        try:
            cells = json.loads(line.split("｜", 1)[1])
            if len(cells) not in {3, 4}:
                continue
            result[int(cells[0])] = {
                "jianpu": "" if cells[1] is None else str(cells[1]),
                "text": "" if cells[2] is None else str(cells[2]),
            }
        except (ValueError, json.JSONDecodeError, IndexError):
            continue
    return result


def align_annotation_to_source(annotation: dict[int, dict[str, str]],
                               source_item: dict | None, *,
                               event_indexed: bool = False) -> dict[int, dict[str, str]]:
    """Map GQS 音序 back to source indexes without confusing bars with notes.

    New annotation GQS uses continuous event indexes; older audit rows used
    source indexes (which include barlines).  Matching the jianpu alongside an
    event index makes the conversion lossless while retaining legacy fallback.
    """
    if not source_item:
        return annotation
    notes = (source_item.get("input") or {}).get("notes_without_jianzi") or []
    by_source = {int(note["index"]): note for note in notes
                 if note.get("index") is not None}
    by_event = {int(note["event_index"]): note for note in notes
                if note.get("event_index") is not None}
    aligned: dict[int, dict[str, str]] = {}
    for raw_index, value in annotation.items():
        candidate = by_event.get(int(raw_index))
        candidate_jianpu = str((candidate or {}).get("jianpu") or "")
        expected_jianpu = str(value.get("jianpu") or "")
        if event_indexed and candidate is not None:
            target = int(candidate["index"])
        elif candidate is not None and canon(candidate_jianpu) == canon(expected_jianpu):
            target = int(candidate["index"])
        elif int(raw_index) in by_source:
            target = int(raw_index)
        elif candidate is not None:
            target = int(candidate["index"])
        else:
            target = int(raw_index)
        aligned[target] = value
    return aligned


def canon(value: str) -> str:
    return re.sub(r"\s+", "", value).translate(
        str.maketrans("一二三四五六七", "1234567")
    )


def pitch_symbols(source_item: dict | None, actions: dict[int, str],
                  score_history: list[dict] | None = None) -> dict[int, str]:
    """Return a compact, non-ambiguous pitch-audit display symbol.

    ✓/✗ are resolved audit outcomes.  ``—`` means the score row has no new
    sounding pitch (rest/tie); ``？`` means it is sounding notation for which
    the deterministic parser cannot currently establish a reliable pitch.
    Keeping those two cases separate avoids presenting skipped rows as a
    neutral third verdict beside actual matches and mismatches.
    """
    if not source_item:
        return {}
    input_data = source_item.get("input") or {}
    current_notes = list(input_data.get("notes_without_jianzi") or [])
    handoff = input_data.get("phrase_handoff") or {}
    previous = handoff.get("previous_phrase") or {}
    previous_notes = list(previous.get("notes") or [])
    previous_actions = {
        int(action["source_index"]): action
        for action in (previous.get("actions") or [])
        if action.get("source_index") is not None
    }
    prefix_phrases = []
    if score_history:
        current_start = int((input_data.get("event_range") or {}).get("start", 0))
        for older in sorted(
            (row for row in score_history
             if int((row.get("input", {}).get("event_range") or {}).get("start", 0))
             < current_start),
            key=lambda row: int((row.get("input", {}).get("event_range") or {}).get("start", 0)),
        ):
            prefix_phrases.append((
                list((older.get("input") or {}).get("notes_without_jianzi") or []),
                list((older.get("reference_plan") or {}).get("actions") or []),
            ))
    if not prefix_phrases and previous:
        prefix_phrases.append((previous_notes, list(previous.get("actions") or [])))
    all_prefix_actions = [action for _, actions_ in prefix_phrases for action in actions_]
    harmonic_active = False
    for action in all_prefix_actions:
        text = str(action.get("text") or action.get("jianzi_text") or "")
        for marker in re.findall(r"泛起|泛止", text):
            harmonic_active = marker == "泛起"
    seed_hui = next(
        (action.get("hui") for action in reversed(all_prefix_actions)
         if harmonic_active and action.get("mode") == "harmonic"
         and action.get("hui") is not None),
        None,
    )
    current_indices = {
        int(note["index"]) for note in current_notes if note.get("index") is not None
    }
    notes = []
    if seed_hui is not None:
        notes.append({
            "index": -1, "jianpu": None, "abc": "", "duration": "",
            "jianzi": f"泛起勾一弦{int(seed_hui) if float(seed_hui).is_integer() else seed_hui}徽",
        })
    prefix_notes = [note for notes, _ in prefix_phrases for note in notes]
    prefix_action_map = {
        int(action["source_index"]): action
        for _, actions_ in prefix_phrases
        for action in actions_
        if action.get("source_index") is not None
    }
    for note in prefix_notes + current_notes:
        if note.get("index") is None:
            continue
        row = dict(note)
        source_index = int(note["index"])
        if source_index in current_indices:
            row["jianzi"] = actions.get(source_index, "") or ""
        else:
            action = prefix_action_map.get(source_index) or previous_actions.get(source_index)
            row["jianzi"] = (action.get("jianzi_text") if action else "") or ""
            if not row["jianzi"] and action:
                row["jianzi"] = str(action.get("text") or "")
        notes.append(row)
    try:
        report = audit_pitch({
            "metadata": dict(input_data.get("metadata") or {}),
            "open_midi": list((input_data.get("normalized_tuning") or {}).get("open_midi") or []),
            "notes": notes,
        }, 50.0)
    except Exception:
        return {int(note["index"]): "？" for note in notes}
    result: dict[int, str] = {}
    for detail in report.get("details") or []:
        index = detail.get("index")
        if index is None or int(index) not in current_indices:
            continue
        status = detail.get("status")
        if status == "matched":
            result[int(index)] = "✓"
        elif status == "mismatched":
            result[int(index)] = "✗"
        elif detail.get("reason") == "no_sounding_jianpu":
            result[int(index)] = "—"
        else:
            result[int(index)] = "？"
    return result


def last_assistant(sample: dict) -> str:
    messages = sample.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            return json.dumps(content, ensure_ascii=False)
    return ""


def readable_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    # Some archived model/tool payloads contain escaped newlines inside an
    # otherwise ordinary string. Display those as actual line breaks.
    return value.replace("\\r\\n", "\n").replace("\\n", "\n")


def public_trajectory(sample: dict) -> list[dict]:
    result = []
    for turn, message in enumerate(sample.get("messages") or [], start=1):
        item = {
            "turn": turn,
            "role": str(message.get("role") or "unknown"),
            "content": readable_text(message.get("content")),
        }
        if message.get("name"):
            item["name"] = str(message["name"])
        if message.get("tool_call_id"):
            item["tool_call_id"] = str(message["tool_call_id"])
        calls = message.get("tool_calls") or []
        if calls:
            item["tool_calls"] = readable_text(calls)
        result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", action="append",
                        help="Optional exact sample ID to include; may be repeated")
    args = parser.parse_args()

    public = read_jsonl(args.input)
    if args.sample_id:
        wanted = set(args.sample_id)
        public = [row for row in public if row.get("sample_id") in wanted]
    private = {row["sample_id"]: row for row in read_jsonl(args.audit)}
    source_rows = attach_derived_event_indices(read_jsonl(args.source))
    source = {row["trajectory_id"]: row for row in source_rows}
    score_history: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        score_history[str(row.get("score_key") or row.get("score_family_id") or "")].append(row)
    grouped: dict[str, dict] = {}
    for sample in public:
        tid = (sample.get("provenance") or {}).get("source_trajectory_id")
        if not tid:
            continue
        group = grouped.setdefault(tid, {
            "id": tid,
            "score_key": (sample.get("provenance") or {}).get("score_key", ""),
            "phrase_id": tid.rsplit("-", 1)[-1],
            "title": "",
            "stages": {},
            "annotation": {},
            "notes": {},
        })
        stage = str(sample.get("agent_stage") or "")
        audit = private.get(sample.get("sample_id"), {})
        teacher = audit.get("teacher_private") or {}
        plan = teacher.get("accepted_plan") or {}
        actions = plan.get("actions") or []
        if not actions and stage == "fingering_agent":
            actions = (sample.get("plan") or {}).get("actions") or []
        group["stages"][stage] = {
            # Accepted plans use internal source indexes. Keep all lookup keys
            # as integers; mixing string action keys with integer source rows
            # made every real action look missing and appended a duplicate
            # block of "extra" rows after the structured score.
            "actions": {int(a["source_index"]): action_text(a) for a in actions
                        if a.get("source_index") is not None},
            "objective": teacher.get("objective", ""),
            "termination": (sample.get("termination") or {}).get("kind", ""),
            "messages": len(sample.get("messages") or []),
            "tool_calls": sum(len(m.get("tool_calls") or [])
                               for m in sample.get("messages") or []),
            "last": last_assistant(sample),
            "trajectory": public_trajectory(sample),
            "teacher_io_trace": teacher.get("teacher_io_trace") or [],
        }
        annotation_gqs = str(teacher.get("annotation_gqs") or "")
        annotation = parse_gqs(annotation_gqs)
        annotation = align_annotation_to_source(
            annotation, source.get(tid),
            event_indexed="GQS｜annotation-phrase-1.2" in annotation_gqs,
        )
        if annotation:
            # Fingering and Guqinizer audits can both carry annotation_gqs.
            # A Guqinizer review_noop record historically stores an all-empty
            # copy; never let that placeholder erase the real annotation
            # already collected from the paired Fingering record.
            current_annotation = group["annotation"]
            current_jianpu = group.get("annotation_jianpu", {})
            for index, value in annotation.items():
                if value["text"] or index not in current_annotation:
                    current_annotation[index] = value["text"]
                if value["jianpu"] or index not in current_jianpu:
                    current_jianpu[index] = value["jianpu"]
            group["annotation_jianpu"] = current_jianpu
        item = source.get(tid)
        if item:
            metadata = item.get("input", {}).get("metadata", {})
            group["title"] = metadata.get("score_title") or group["title"]
            for note in item.get("input", {}).get("notes_without_jianzi") or []:
                if note.get("index") is not None:
                    written = str(note.get("jianpu") or "")
                    alt = str(note.get("jianpu_alt") or "")
                    abc = str(note.get("abc") or "")
                    group["notes"][int(note["index"])] = (
                        f"{written} {alt}" if alt and abc.lstrip().startswith("[") else written
                    )

    records = []
    for group in grouped.values():
        # Older audit files stored repeat-omitted rows as empty GQS cells.
        # Recover the semantic display marker from the inferred source so the
        # comparison table does not confuse omission with missing annotation.
        item = source.get(group["id"])
        if item:
            for note in item.get("input", {}).get("notes_without_jianzi") or []:
                if not note.get("notation_omitted"):
                    continue
                index = int(note["index"])
                is_bar = (str(note.get("jianpu") or "").strip() == "|"
                          or str(note.get("abc") or "").strip() == "|")
                if not is_bar and not group["annotation"].get(index):
                    group["annotation"][index] = OMITTED_PLACEHOLDER
        annotation = group.get("annotation", {})
        ann_jianpu = group.get("annotation_jianpu", {})
        indices = set(group["notes"]) | set(annotation)
        for stage in ("fingering_agent", "guqinization"):
            indices |= set((group["stages"].get(stage) or {}).get("actions", {}))
        rows = []
        note_by_index = {
            int(note["index"]): note
            for note in (item.get("input", {}).get("notes_without_jianzi") or [])
            if note.get("index") is not None
        } if item else {}
        finger_pitch = pitch_symbols(
            item, (group["stages"].get("fingering_agent") or {}).get("actions", {}),
            score_history.get(str(group.get("score_key") or "")),
        )
        guqin_pitch = pitch_symbols(
            item, (group["stages"].get("guqinization") or {}).get("actions", {}),
            score_history.get(str(group.get("score_key") or "")),
        )
        last_marker = None
        # Walk source order so structural rows are visible and the three
        # columns remain aligned even when annotation uses event indexes.
        for index, note in note_by_index.items():
            section = note.get("section") or {}
            marker = str(section.get("marker") or "")
            if marker and (section.get("start") or marker != last_marker):
                rows.append({"kind": "section", "marker": marker})
            if marker:
                last_marker = marker
            is_bar = (str(note.get("jianpu") or "").strip() == "|"
                      or str(note.get("abc") or "").strip() == "|"
                      or str(note.get("duration") or "").strip() == "小节线")
            if is_bar:
                rows.append({"kind": "bar"})
                continue
            ann = annotation.get(index, "")
            row = {
                "kind": "note",
                # Public tools and GQS 1.2 use contiguous event indexes;
                # barlines have no event index. Internal source indexes are
                # used only for the lookups above.
                "index": int(note.get("event_index"))
                if note.get("event_index") is not None else int(index),
                "jianpu": group["notes"].get(index) or ann_jianpu.get(index, ""),
                "finger": (group["stages"].get("fingering_agent") or {}).get("actions", {}).get(index, ""),
                "guqin": (group["stages"].get("guqinization") or {}).get("actions", {}).get(index, ""),
                "annotation": ann,
                "finger_pitch": finger_pitch.get(index, "？"),
                "guqin_pitch": guqin_pitch.get(index, "？"),
            }
            for key in ("finger", "guqin"):
                value = row[key]
                row[key + "_status"] = (
                    "both_empty" if not value and not ann else
                    "match" if canon(value) == canon(ann) else
                    "annotation_empty" if value and not ann else "different"
                )
            rows.append(row)
        # Preserve any sparse action/annotation index absent from source.
        source_indices = set(note_by_index)
        for index in sorted(indices - source_indices, key=lambda x: int(x)):
            ann = annotation.get(index, "")
            row = {
                "kind": "note", "index": int(index),
                "jianpu": ann_jianpu.get(index, ""),
                "finger": (group["stages"].get("fingering_agent") or {}).get("actions", {}).get(index, ""),
                "guqin": (group["stages"].get("guqinization") or {}).get("actions", {}).get(index, ""),
                "annotation": ann,
                "finger_pitch": finger_pitch.get(index, "？"),
                "guqin_pitch": guqin_pitch.get(index, "？"),
            }
            for key in ("finger", "guqin"):
                value = row[key]
                row[key + "_status"] = (
                    "both_empty" if not value and not ann else
                    "match" if canon(value) == canon(ann) else
                    "annotation_empty" if value and not ann else "different")
            rows.append(row)
        guqin = group["stages"].get("guqinization") or {}
        # ``no_changes`` only describes the final response. A trajectory may
        # reach it after many real edits, so only the explicit review_noop
        # objective earns the no-op label.
        group["no_op"] = guqin.get("objective") == "review_noop"
        group["rows"] = rows
        records.append(group)

    records.sort(key=lambda row: (row["score_key"], row["phrase_id"]))
    data = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    score_count = len({row["score_key"] for row in records})
    no_op_count = sum(bool(row["no_op"]) for row in records)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全部训练轨迹｜分层复核</title>
<style>
:root{{--bg:#101216;--fg:#e8ecf1;--muted:#a6b0bd;--line:#303741;--card:#181c22;--green:#173b2a;--greenfg:#9ae6b4;--red:#481f25;--redfg:#ffb4bd;--amber:#4c3914;--amberfg:#ffdc8c;--gray:#252b33;--blue:#8bb7ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1380px;margin:0 auto;padding:22px 18px 50px}}h1{{font-size:22px;margin:0 0 5px}}.sub{{color:var(--muted);margin:0 0 14px}}
.stats{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 16px}}.stat{{background:var(--card);border:1px solid var(--line);padding:8px 13px;min-width:125px;border-radius:7px}}.stat b{{display:block;font-size:19px}}.stat span{{font-size:12px;color:var(--muted)}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 14px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:2}}#search{{flex:1;min-width:260px;padding:8px 10px;border:1px solid var(--line);border-radius:6px;font:inherit}}button{{padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);cursor:pointer;font:inherit}}button:hover{{border-color:var(--blue)}}
details.score,details.phrase{{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:7px 0;overflow:hidden}}details.score>summary{{padding:10px 13px;font-weight:600;cursor:pointer}}details.phrase{{margin:7px 12px 10px}}details.phrase>summary{{padding:8px 11px;cursor:pointer;display:flex;gap:9px;align-items:center;flex-wrap:wrap}}.count{{font-weight:400;color:var(--muted);font-size:12px}}.chip{{display:inline-block;padding:1px 6px;border-radius:4px;background:#e9edf5;color:#42526b;font-size:11px;font-weight:500}}.chip.noop{{background:#eee7fb;color:#61479c}}
.body{{padding:0 11px 13px}}.meta{{color:var(--muted);font-size:12px;margin:0 0 8px}}.stage-summary{{display:flex;gap:8px;flex-wrap:wrap;margin:5px 0 10px}}.stage-summary span{{border:1px solid var(--line);padding:4px 7px;border-radius:5px;background:#fafbfc;font-size:12px}}.stage-summary b{{font-weight:600}}
.reasoning{{white-space:pre-wrap;word-break:break-word;background:#fafbfc;border-left:3px solid var(--blue);padding:8px 10px;margin:8px 0;font-size:13px;max-height:240px;overflow:auto}}.table-wrap{{overflow:auto;max-height:65vh;border:1px solid var(--line);border-radius:5px}}table{{width:100%;border-collapse:collapse;background:var(--card);font-size:12px}}th,td{{padding:6px 8px;border-bottom:1px solid #edf0f3;text-align:left;vertical-align:top;white-space:pre-wrap;word-break:break-word}}th{{position:sticky;top:0;background:#f1f3f6;color:var(--muted);font-weight:600;z-index:1}}td.idx{{width:66px;color:var(--muted)}}td.jp{{width:100px}}td.jz{{min-width:170px}}td.match{{background:var(--green);color:var(--greenfg)}}td.different{{background:var(--red);color:var(--redfg)}}td.annotation_empty{{background:var(--amber);color:var(--amberfg)}}td.both_empty{{background:var(--gray);color:var(--muted)}}
.trajectory{{margin:9px 0}}.turn{{border-top:1px solid var(--line);padding:9px 4px}}.turn:first-child{{border-top:0}}.turn-head{{display:flex;gap:8px;align-items:center;margin-bottom:5px;color:var(--muted);font-size:12px}}.role{{font-weight:600;color:var(--fg)}}.turn-content,.tool-call{{white-space:pre-wrap;word-break:break-word;margin:0;font:13px/1.55 ui-monospace,SFMono-Regular,Consolas,"Microsoft YaHei",monospace}}.tool-call{{margin-top:7px;padding:7px 9px;background:#f3f5f8;border-left:3px solid var(--blue)}}
.legend{{color:var(--muted);font-size:12px;margin:8px 0}}.legend i{{display:inline-block;width:11px;height:11px;margin:0 3px 0 10px;vertical-align:-1px;border:1px solid var(--line)}}.legend i:first-child{{margin-left:0;background:var(--green)}}.legend i:nth-child(2){{background:var(--red)}}.legend i:nth-child(3){{background:var(--amber)}}.legend i:nth-child(4){{background:var(--gray)}}.pitch{{font-size:18px;font-weight:700;text-align:center;width:78px}}.pitch.match{{color:var(--greenfg);background:var(--green)}}.pitch.mismatch{{color:var(--redfg);background:var(--red)}}.pitch.unresolved{{color:var(--muted);background:var(--gray)}}.structure td{{font-weight:600;color:var(--muted);background:#252c35;text-align:center}}.section-row td{{color:#b9d2ff;background:#182d4d}}.empty{{color:var(--muted);padding:20px 4px}}
.chip{{background:#263447;color:#bdd2ef}}.chip.noop{{background:#342551;color:#d4c1ff}}.stage-summary span,.reasoning,.tool-call{{background:#20262e}}th{{background:#252c35}}th,td{{border-bottom-color:#2b323b}}
</style></head><body><main>
<h1>全部训练轨迹｜曲谱 → phrase 分层复核</h1>
<p class="sub">曲谱 ID 和 phrase ID 两级展开；phrase 内同时查看 Fingering、Guqinizer 与标注。绿色＝文字归一化后一致，红色＝双方都有文字但不同，黄色＝只有一方有文字，灰色＝双方均为空。</p>
<div class="stats"><div class="stat"><b>{len(records)}</b><span>phrase 轨迹</span></div><div class="stat"><b>{score_count}</b><span>曲谱 ID</span></div><div class="stat"><b>{sum(len(r["rows"]) for r in records)}</b><span>对比行</span></div><div class="stat"><b>{no_op_count}</b><span>no-op phrase</span></div></div>
<div class="controls"><input id="search" placeholder="搜索曲谱 ID、phrase ID 或曲名"><button id="expand" type="button">展开匹配项</button><button id="collapse" type="button">全部收起</button></div>
<p class="legend"><i></i>一致 <i></i>双方不同 <i></i>单方有文字 <i></i>双方均为空</p><section id="tree"></section>
</main><script>
const data={data};
const tree=document.getElementById('tree'), search=document.getElementById('search');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const stageLabel={{fingering_agent:'Fingering',guqinization:'Guqinizer'}};
function stageInfo(t,s){{const x=t.stages[s]||{{}};return `<span><b>${{stageLabel[s]}}</b> · ${{x.messages||0}} 消息 · ${{x.tool_calls||0}} 工具调用 · ${{esc(x.termination||'')}}</span>`}}
function trajectory(t,s){{const x=t.stages[s]||{{}};const turns=x.trajectory||[];if(!turns.length)return '<p class="empty">[无该阶段轨迹]</p>';return `<div class="trajectory">${{turns.map(m=>`<section class="turn"><div class="turn-head"><span>第 ${{m.turn}} 轮</span><span class="role">${{esc(m.role)}}</span>${{m.name?`<span>工具：${{esc(m.name)}}</span>`:''}}</div><pre class="turn-content">${{esc(m.content||'[空内容]')}}</pre>${{m.tool_calls?`<pre class="tool-call">工具调用\n${{esc(m.tool_calls)}}</pre>`:''}}</section>`).join('')}}</div>`}}
function teacherTrace(t,s){{const x=t.stages[s]||{{}};const traces=x.teacher_io_trace||[];if(!traces.length)return '<p class="empty">[无私有教师调用审计]</p>';return `<div class="trajectory">${{traces.map((v,i)=>`<section class="turn"><div class="turn-head"><span>教师 API 第 ${{i+1}} 轮</span></div><pre class="turn-content">${{esc(JSON.stringify(v,null,2))}}</pre></section>`).join('')}}</div>`}}
function pitchCell(value){{const cls=value==='✓'?'match':value==='✗'?'mismatch':value==='—'?'not-applicable':'unresolved';const title=value==='—'?'无新音高（休止或延音）':value==='？'?'有音高但当前无法可靠解析': '✓ 匹配｜✗ 不匹配';return `<td class="pitch ${{cls}}" title="${{title}}">${{esc(value||'？')}}</td>`}}
function table(t){{return `<div class="table-wrap"><table><thead><tr><th>序号</th><th>简谱</th><th>Fingering 最终版</th><th>Guqinizer 最终版</th><th>标注版本</th><th>Fingering 音高</th><th>Guqinizer 音高</th></tr></thead><tbody>${{t.rows.map(r=>r.kind==='section'?`<tr class="structure section-row"><td colspan="7">段落标记｜${{esc(r.marker)}}</td></tr>`:r.kind==='bar'?`<tr class="structure bar-row"><td colspan="7">小节线</td></tr>`:`<tr><td class="idx">${{r.index}}</td><td class="jp">${{esc(r.jianpu)}}</td><td class="jz ${{r.finger_status}}">${{esc(r.finger||'[空]')}}</td><td class="jz ${{r.guqin_status}}">${{esc(r.guqin||'[空]')}}</td><td class="jz ${{r.annotation?'':'both_empty'}}">${{esc(r.annotation||'[空]')}}</td>${{pitchCell(r.finger_pitch)}}${{pitchCell(r.guqin_pitch)}}</tr>`).join('')}}</tbody></table></div>`}}
function phrase(t,open){{const noop=t.no_op?'<span class="chip noop">no-op</span>':'';return `<details class="phrase" data-id="${{esc(t.id)}}" ${{open?'open':''}}><summary><span>${{esc(t.phrase_id)}}</span>${{noop}}<span class="count">${{t.rows.length}} 行 · ${{esc(t.id)}}</span></summary><div class="body"><p class="meta">曲谱：${{esc(t.score_key)}}${{t.title?'｜'+esc(t.title):''}} · 标注对比及两阶段最终结果</p><div class="stage-summary">${{stageInfo(t,'fingering_agent')}}${{stageInfo(t,'guqinization')}}</div><details><summary>Fingering 完整轨迹（逐轮）</summary>${{trajectory(t,'fingering_agent')}}</details><details><summary>Guqinizer 完整轨迹（逐轮）</summary>${{trajectory(t,'guqinization')}}</details><details><summary>Guqinizer 教师原始输入输出（私有审计）</summary>${{teacherTrace(t,'guqinization')}}</details>${{table(t)}}</div></details>`}}
function draw(){{const q=search.value.trim().toLowerCase(); tree.innerHTML=''; const buckets=new Map(); for(const t of data){{const hit=!q||[t.id,t.score_key,t.phrase_id,t.title].join(' ').toLowerCase().includes(q);if(!hit)continue; if(!buckets.has(t.score_key))buckets.set(t.score_key,[]);buckets.get(t.score_key).push(t)}} if(!buckets.size){{tree.innerHTML='<p class="empty">没有匹配的曲谱或 phrase。</p>';return}} for(const [score,items] of buckets){{const title=items[0].title||'';const d=document.createElement('details');d.className='score';d.open=Boolean(q);d.innerHTML=`<summary>${{esc(score)}}${{title?'｜'+esc(title):''}} <span class="count">${{items.length}} 个 phrase</span></summary><div>${{items.map(t=>phrase(t,Boolean(q))).join('')}}</div>`;tree.appendChild(d)}}}}
search.addEventListener('input',draw);document.getElementById('expand').addEventListener('click',()=>document.querySelectorAll('details.score,details.phrase').forEach(d=>d.open=true));document.getElementById('collapse').addEventListener('click',()=>document.querySelectorAll('details.score,details.phrase').forEach(d=>d.open=false));draw();
</script></body></html>'''
    output.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(output), "phrases": len(records),
                      "scores": score_count, "no_op": no_op_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
