"""Shared public tool runtime for teacher generation and inference."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from agents.abc_to_jianzipu.inverse_baseline import AUDIT
from agents.abc_to_jianzipu.jianzi_renderer import hui_label, render_jianzi_surface
from agents.abc_to_jianzipu.models import ScoreEvent
from agents.abc_to_jianzipu.pitch_candidates import abc_chord_midis, abc_chord_tones, candidates_for_event
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER
from agents.abc_to_jianzipu.teacher_trajectory import (
    _render_phrase_lines, chord_jianpu_label,
)
from agents.abc_to_jianzipu.trajectory_replay import compare_replay_to_patch_targets, replay_patches

def target_midis(item: dict, source_index: int) -> list[float]:
    """Every sounding target of a note: all ABC chord tones, else jianpu.

    Chord tokens are authoritative for 撮-style double stops; the jianpu field
    only carries one of the tones, so it is used solely as fallback.
    """
    note = next(note for note in item["input"]["notes_without_jianzi"]
                if int(note["index"]) == source_index)
    tonic = AUDIT.parse_tonic_midi(item["input"]["metadata"])
    # 和弦（撮等）的音高以存储双谱字为准：ABC 转换在降号调上存在半音偏差，
    # 记谱本身才是真相；无双谱字时回退到 ABC 和弦。
    if abc_chord_tones(str(note.get("abc") or "")):
        values = [AUDIT.parse_jianpu(note.get(field), tonic)
                  for field in ("jianpu", "jianpu_alt")]
        values = [float(value) for value in values if value is not None]
        if len(values) == 2:
            return values
        midis = abc_chord_midis(str(note.get("abc") or ""))
        if midis:
            return midis
    values = [AUDIT.parse_jianpu(note.get(field), tonic) for field in ("jianpu", "jianpu_alt")]
    values = [value for value in values if value is not None]
    if not values:
        raise ValueError(f"source_index {source_index} has no sounding target")
    return [float(value) for value in values]

def jianpu_symbol_for_target(note: dict, target_midi: float, tonic_midi: float) -> str:
    """Return the written jianpu component that actually denotes ``target_midi``.

    A chord stores its two tones in ``jianpu`` and ``jianpu_alt``. Candidate
    rows are grouped per *tone*, so displaying the combined chord label in a
    per-tone table makes the header point at the wrong note.
    """
    for field in ("jianpu", "jianpu_alt"):
        value = str(note.get(field) or "").strip()
        parsed = AUDIT.parse_jianpu(value, tonic_midi)
        if value and parsed is not None and abs(float(parsed) - target_midi) < 1e-6:
            return value
    return chord_jianpu_label(
        note.get("abc"), note.get("jianpu") or note.get("jianpu_alt") or "",
        note.get("jianpu_alt"),
    )

def render_pitch_candidate_table(target: float, candidates: list[dict],
                                 modes: tuple[str, ...]) -> str:
    mode_names = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}
    lines = [
        f"指定音高｜MIDI {target:g}｜范围｜{'、'.join(mode_names.get(m, m) for m in modes)}",
        "序号｜方式｜弦徽｜实得MIDI｜可信度",
    ]
    for rank, candidate in enumerate(candidates, 1):
        mode = mode_names.get(candidate.get("mode"), str(candidate.get("mode") or "未知"))
        position = f'{candidate.get("string")}弦'
        if candidate.get("mode") not in {"open"} and candidate.get("hui") is not None:
            position += candidate.get("hui_label") or hui_label(candidate.get("hui"))
        confidence = "精确" if candidate.get("confidence") == "exact" else "近似"
        lines.append(
            f'{rank}｜{mode}｜{position}｜{float(candidate.get("sounding_midi")):.1f}｜{confidence}'
        )
    if not candidates:
        lines.append("无符合容差的候选")
    return "\n".join(lines)

def pitch_label(item: dict, source_index: int) -> str:
    """Jianpu label of a note; chord notes show both stored digits."""
    note = next(note for note in item["input"]["notes_without_jianzi"]
                if int(note["index"]) == source_index)
    written = note.get("jianpu") or note.get("jianpu_alt") or "休止"
    return chord_jianpu_label(note.get("abc"), written, note.get("jianpu_alt"))

def event_index_for_source(item: dict, source_index: int) -> int | None:
    """Translate an internal sparse source index to the visible continuous 音序."""
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        if int(note.get("index")) == int(source_index):
            value = note.get("event_index")
            return int(value) if value is not None else None
    return None

def source_index_for_event(item: dict, event_index: int) -> int | None:
    for note in item.get("input", {}).get("notes_without_jianzi", []):
        value = note.get("event_index")
        if value is not None and int(value) == int(event_index):
            return int(note["index"])
    return None

def render_grouped_candidate_table(target: float, candidates: list[dict],
                                   modes: tuple[str, ...],
                                   sources: list[dict]) -> str:
    """Render one physical candidate list with every score event that reuses it."""
    missing = [source.get("source_index") for source in sources
               if source.get("event_index") is None]
    if missing:
        raise ValueError(
            "public candidate rendering requires event_index; missing source_index="
            + "、".join(str(index) for index in missing)
        )
    source_text = "、".join(str(source["event_index"]) for source in sources)
    # Source entries carry the one jianpu component corresponding to this
    # target MIDI, not the entire chord label. Keep every event source but
    # display the deduplicated target symbol once.
    jianpu_values = list(dict.fromkeys(
        str(source.get("jianpu") or "") for source in sources
        if str(source.get("jianpu") or "")
    ))
    lines = render_pitch_candidate_table(target, candidates, modes).splitlines()
    lines[0] = f"目标音高｜MIDI {target:g}｜来源｜{source_text}"
    if jianpu_values:
        lines[0] += f"｜简谱｜{jianpu_values[0]}"
    return "\n".join(lines)

def harmonic_region_at_phrase_start(
    item: dict, historical: dict[tuple[str, str], dict]
) -> bool:
    """Replay confirmed 泛起/泛止 markers before the current phrase."""
    current_start = int(item["input"]["event_range"]["start"])
    earlier = sorted(
        (row for (score_key, _), row in historical.items()
         if score_key == item["score_key"]
         and int(row["input"]["event_range"]["start"]) < current_start),
        key=lambda row: int(row["input"]["event_range"]["start"]),
    )
    active = False
    for row in earlier:
        actions = sorted(
            row.get("reference_plan", {}).get("actions", []),
            key=lambda action: int(action["source_index"]),
        )
        for action in actions:
            text = str(action.get("text") or action.get("jianzi_text") or "")
            for marker in re.findall(r"泛起|泛止", text):
                active = marker == "泛起"
    return active

def pitch_audit_notes(
    item: dict, actions_by_source: dict[int, dict],
    historical: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Build audit rows by replaying every confirmed earlier phrase."""
    handoff = (item.get("input") or {}).get("phrase_handoff") or {}
    previous = handoff.get("previous_phrase") or {}
    current_notes = list((item.get("input") or {}).get("notes_without_jianzi") or [])
    prefix_phrases: list[tuple[list[dict], list[dict]]] = []
    if historical:
        current_start = int(item["input"]["event_range"]["start"])
        earlier = sorted(
            (row for (score_key, _), row in historical.items()
             if score_key == item["score_key"]
             and int(row["input"]["event_range"]["start"]) < current_start),
            key=lambda row: int(row["input"]["event_range"]["start"]),
        )
        prefix_phrases = [
            (list(row["input"].get("notes_without_jianzi") or []),
             list(row.get("reference_plan", {}).get("actions") or []))
            for row in earlier
        ]
    if not prefix_phrases and previous:
        prefix_phrases = [(
            list(previous.get("notes") or []),
            list(previous.get("actions") or []),
        )]

    def render_rows(notes: list[dict], action_map: dict[int, dict]) -> list[dict]:
        rows = []
        for note in notes:
            row = deepcopy(note)
            action = action_map.get(int(note["index"]))
            if action is None:
                row["jianzi"] = ""
            else:
                value = action.get("jianzi_text")
                if value is None:
                    value = action.get("text") or ""
                row["jianzi"] = str(value)
            rows.append(row)
        return rows

    prefix = []
    # A phrase may begin in the middle of a harmonic passage.  The previous
    # phrase's structured action state then tells us the harmonic hui even if
    # its readable text omits ``泛起`` and the hui as an inherited field.
    all_prefix_actions = [action for _, actions in prefix_phrases for action in actions]
    seed_allowed = item.get("harmonic_region_at_start")
    if seed_allowed is None:
        seed_allowed = True
        for action in all_prefix_actions:
            text = str(action.get("text") or action.get("jianzi_text") or "")
            for marker in re.findall(r"泛起|泛止", text):
                seed_allowed = marker == "泛起"
    seed_hui = next(
        (action.get("hui") for action in reversed(all_prefix_actions)
         if seed_allowed and action.get("mode") == "harmonic"
         and action.get("hui") is not None),
        None,
    )
    if seed_hui is not None:
        prefix.append({
            "index": -1,
            "jianpu": None,
            "abc": "",
            "duration": "",
            # The pitch auditor parses numeric string designators.  Keep this
            # synthetic state seed in the same surface form as normal tool
            # input, otherwise “一弦” is treated as having no string and the
            # harmonic mode is never activated.
            "jianzi": f"泛起勾1弦{hui_label(seed_hui)}",
        })
    for notes, actions in prefix_phrases:
        action_map = {
            int(action["source_index"]): action
            for action in actions if action.get("source_index") is not None
        }
        prefix.extend(render_rows(notes, action_map))
    return prefix + render_rows(current_notes, actions_by_source)

def render_edit_preview(actions: list[dict], patches: list[dict], valid: bool,
                        errors: list[dict], item: dict | None = None,
                        warnings: list[dict] | None = None) -> str:
    changed = {int(patch["source_index"]) for patch in patches
               if patch.get("source_index") is not None}
    by_index = {int(action["source_index"]): action for action in actions}
    patch_types: dict[int, list[str]] = {}
    for patch in patches:
        patch_types.setdefault(int(patch["source_index"]), []).append(
            str(patch.get("patch_type") or "UNKNOWN"))
    mode_names = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}
    lines = [f'预览｜{"可应用" if valid else "不可应用"}｜变更{len(changed)}音',
             "序号｜简谱｜减字显示｜谱面减字"]
    pitch_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "jianzi_pitch_mismatch"
        and warning.get("source_index") is not None
    }
    for index in sorted(changed):
        action = by_index.get(index)
        visible_index = event_index_for_source(item, index) if item is not None else None
        visible_index = index if visible_index is None else visible_index
        if action is None:
            lines.append(
                f'{visible_index}｜{pitch_label(item, index) if item is not None else ""}｜'
                f'未找到动作｜-')
            continue
        mode = mode_names.get(action.get("mode"), action.get("mode") or "待定")
        partner = ""
        if isinstance(action.get("string2"), int):
            if action.get("mode2") == "open" and action.get("mode") != "open":
                partner = f'＋散{action["string2"]}弦'
            else:
                partner = f'＋{action["string2"]}弦'
        if action.get("mode") == "open":
            position = f'{action.get("string")}弦{partner}'
        else:
            position = f'{hui_label(action.get("hui"))}{action.get("string")}弦{partner}'
        left = action.get("left_finger") or "左手待定"
        right = action.get("right_finger") or ("续音" if not action.get("attack") else "右手待定")
        surface = render_jianzi_surface(
            action, omitted_placeholder=OMITTED_PLACEHOLDER
        ) or "—"
        explicit = action.get("jianzi_text")
        display = ("待填写" if explicit is None else
                   ("已置空" if explicit == "" else "已填写"))
        jianpu = pitch_label(item, index) if item is not None else ""
        warning = pitch_warnings.get(index)
        warning_suffix = (
            ":warning:音高不匹配（当前可能仍处于泛音状态；若此音应为按音，请明确写“按音”或先泛止）"
            if warning and warning.get("possible_harmonic_state_mismatch")
            else ":warning:音高不匹配" if warning else ""
        )
        lines.append(
            f'{visible_index}｜{jianpu}｜{display}｜{surface}{warning_suffix}'
        )
    # ``edit_plan`` previews only the submitted rows above.  A cumulative
    # audit can nevertheless find pitch warnings inherited from the current
    # Base/earlier plan.  They must be shown to the model: otherwise the
    # acceptance gate keeps the turn alive for an issue it cannot identify.
    unmodified_warning_indices = sorted(set(pitch_warnings) - changed)
    if unmodified_warning_indices:
        lines.append("仍有未修改的音高警告｜以下行未包含在本轮提交；请核对或查询候选：")
    for index in unmodified_warning_indices:
        action = by_index.get(index)
        visible_index = event_index_for_source(item, index) if item is not None else None
        visible_index = index if visible_index is None else visible_index
        if action is None:
            lines.append(
                f'{visible_index}｜{pitch_label(item, index) if item is not None else ""}｜'
                "未找到动作｜-:warning:音高不匹配"
            )
            continue
        surface = render_jianzi_surface(
            action, omitted_placeholder=OMITTED_PLACEHOLDER
        ) or "—"
        warning = pitch_warnings[index]
        warning_suffix = (
            ":warning:音高不匹配（当前可能仍处于泛音状态；若此音应为按音，请明确写“按音”或先泛止）"
            if warning.get("possible_harmonic_state_mismatch")
            else ":warning:音高不匹配"
        )
        lines.append(
            f'{visible_index}｜{pitch_label(item, index) if item is not None else ""}｜'
            f'未修改｜{surface}{warning_suffix}'
        )
    if errors:
        lines.append("错误｜" + json.dumps(errors, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)

def advance_pitch_warning_state(
    pending: set[int],
    repair_attempted: set[int],
    current_warnings: set[int],
    candidate_informed_edits: set[int],
) -> set[int]:
    """Record one edit-plan result and return warnings still requiring a turn.

    A warning produced by this very edit cannot also count as repaired by the
    same edit, even if its candidates were queried earlier.  The teacher must
    receive that warning and take at least one later turn before choosing
    whether to repair it or retain the notation.  A later candidate-informed
    edit may resolve the warning or leave it as an advisory result.
    """
    previously_pending = set(pending)
    pending.update(current_warnings)
    later_repairs = candidate_informed_edits & previously_pending
    repair_attempted.update(later_repairs)
    # If the later repair made the warning disappear, it no longer needs to
    # remain pending.  If it remains visible, the later attempted repair is
    # sufficient to let the model end on this advisory warning.
    pending.difference_update(later_repairs - current_warnings)
    return pending - repair_attempted

def compact_context_catalog(entries: list[dict]) -> str:
    """Render a short, human-readable catalog of expandable phrase IDs.

    ``list_context`` is a discovery tool, not a score dump.  Preserve each
    section marker and phrase-id availability, but collapse consecutive
    ``pNNNN`` IDs into inclusive ranges.
    """
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        marker = str(entry.get("section_marker") or "未分段").strip() or "未分段"
        grouped.setdefault(marker, []).append(str(entry["phrase_id"]))

    def compact(ids: list[str]) -> str:
        runs: list[tuple[str, str]] = []
        start = previous = ids[0]
        previous_number = int(match.group(1)) if (match := re.search(r"(\d+)$", previous)) else None
        for phrase_id in ids[1:]:
            match = re.search(r"(\d+)$", phrase_id)
            number = int(match.group(1)) if match else None
            if number is not None and previous_number is not None and number == previous_number + 1:
                previous, previous_number = phrase_id, number
                continue
            runs.append((start, previous))
            start = previous = phrase_id
            previous_number = number
        runs.append((start, previous))
        return "、".join(first if first == last else f"{first}–{last}" for first, last in runs)

    lines = [f"可展开更早段｜共 {len(entries)} 段"]
    lines.extend(f"{marker}：{compact(ids)}" for marker, ids in grouped.items())
    return "\n".join(lines)

class RealToolRuntime:
    def __init__(self, item: dict, historical: dict[tuple[str, str], dict], *,
                 basic_fingering: bool = False,
                 toward_reference: bool = False,
                 target_patches: list[dict] | None = None) -> None:
        self.item = item
        self.historical = historical
        self.basic_fingering = basic_fingering
        self.toward_reference = toward_reference
        self.target_patches = list(target_patches or [])
        self.calls: list[dict] = []
        notes = item.get("input", {}).get("notes_without_jianzi", [])
        missing_event_indices = [
            int(note["index"]) for note in notes
            if note.get("index") is not None
            and not (str(note.get("abc") or "").strip() == "|"
                     or str(note.get("jianpu") or "").strip() == "|"
                     or note.get("duration") == "小节线")
            and note.get("event_index") is None
        ]
        if missing_event_indices:
            raise ValueError(
                "public tool protocol requires continuous event_index for every "
                f"non-bar note; missing source_index={missing_event_indices}"
            )
        self.event_to_source = {
            int(note["event_index"]): int(note["index"])
            for note in notes
            if note.get("event_index") is not None
        }
        self.source_to_event = {source: event for event, source in self.event_to_source.items()}
        if len(self.event_to_source) != len(self.source_to_event):
            raise ValueError("public tool protocol has duplicate event_index values")
        # edit_plan 的累计 patch：多批次提交时，预览与审计始终针对累计状态，
        # 否则后续批次会被误报为“其余音全部待定”。重复序号按序覆盖。
        self.accumulated_patches: list[dict] = []

    def _event_to_source(self, value: Any) -> int:
        """Resolve a required public continuous 音序 to its source row."""
        index = int(value)
        if index not in self.event_to_source:
            raise ValueError(f"unknown public event_index: {index}")
        return self.event_to_source[index]

    def _source_to_event(self, value: Any) -> int:
        index = int(value)
        if index not in self.source_to_event:
            raise ValueError(f"source_index has no public event_index: {index}")
        return self.source_to_event[index]

    def invoke(self, name: str, args: dict) -> dict:
        try:
            audit_result = None
            if name == "expand_context":
                key = (self.item["score_key"], str(args["phrase_id"]))
                previous = ((self.item.get("input") or {}).get("phrase_handoff") or {}).get("previous_phrase") or {}
                if str(previous.get("phrase_id") or "") == key[1]:
                    raise ValueError(
                        "expand_context must not repeat the read-only previous phrase already shown in user content"
                    )
                older = self.historical[key]
                if (int(older["input"]["event_range"]["start"]) >=
                        int(self.item["input"]["event_range"]["start"])):
                    raise ValueError("expand_context only permits earlier phrases")
                result = {
                    "phrase_id": key[1], "readonly": True,
                    "text": (
                        f"【只读更早段 {key[1]}｜confirmed_readonly】\n"
                        "序号｜简谱｜ABC｜时值｜谱面减字\n"
                        + _render_phrase_lines(
                            older["input"]["notes_without_jianzi"],
                            older["reference_plan"]["actions"], readonly=True,
                        )
                    ),
                }
            elif name == "list_context":
                current_start = int(self.item["input"]["event_range"]["start"])
                entries = []
                for (score_key, phrase_id), older in self.historical.items():
                    if score_key != self.item["score_key"]:
                        continue
                    if int(older["input"]["event_range"]["start"]) >= current_start:
                        continue
                    section = (older["input"].get("phrase_handoff") or {}).get("section") or {}
                    entries.append({"phrase_id": phrase_id,
                                    "section_marker": section.get("marker", ""),
                                    "event_range": older["input"]["event_range"]})
                entries.sort(key=lambda entry: int(entry["event_range"]["start"]))
                result = {
                    "text": compact_context_catalog(entries),
                    "readonly": True,
                }
            elif name == "get_pitch_candidates":
                explicit_target = args.get("target_midi")
                skipped_indices = []
                if explicit_target is not None:
                    groups = [{"target_midi": float(explicit_target), "sources": []}]
                    is_batch_query = False
                else:
                    if args.get("event_index") is not None:
                        indices = [self._event_to_source(args["event_index"])]
                    elif args.get("event_indices") is not None:
                        indices = [self._event_to_source(value) for value in args.get("event_indices") or []]
                    else:
                        raise ValueError(
                            "event_index, event_indices or target_midi is required; "
                            "source_index is internal-only"
                        )
                    if not indices:
                        raise ValueError("event_index, event_indices or target_midi is required")
                    is_batch_query = len(indices) > 1
                    valid_entries = []
                    skipped_indices = []
                    tonic_midi = AUDIT.parse_tonic_midi(self.item["input"]["metadata"])
                    for index in indices:
                        try:
                            note = next(n for n in self.item["input"]["notes_without_jianzi"]
                                        if int(n["index"]) == index)
                            tones = abc_chord_tones(str(note.get("abc") or ""))
                            targets = target_midis(self.item, index)
                        except (StopIteration, ValueError):
                            skipped_indices.append(index)
                            continue
                        valid_entries.append((index, note, tones, targets))
                    if not valid_entries:
                        raise ValueError("no source index with a parseable sounding target")
                    grouped: dict[float, dict] = {}
                    for index, note, tones, targets in valid_entries:
                        # 和弦逐音查询：标签用各自 ABC 音名（简谱只承载单音）。
                        for position, target in enumerate(targets, 1):
                            label = None
                            if len(targets) > 1:
                                token = tones[position - 1][0] if position <= len(tones) else ""
                                label = f'和弦音{position}（{token}）' if token else f'和弦音{position}'
                            target_value = float(target)
                            key = round(target_value, 6)
                            group = grouped.setdefault(
                                key, {"target_midi": target_value, "sources": []}
                            )
                            group["sources"].append({
                                "event_index": self._source_to_event(index),
                                "source_index": index,
                                "jianpu": jianpu_symbol_for_target(
                                    note, target_value, tonic_midi
                                ),
                                "label": label or f'简谱{note.get("jianpu") or note.get("jianpu_alt") or "未知"}',
                            })
                    groups = list(grouped.values())
                type_value = args.get("类型")
                requested_types = (type_value if isinstance(type_value, list)
                                   else [type_value])
                type_mapping = {"泛音": "harmonic", "散音": "open", "按音": "stopped"}
                type_modes = list(dict.fromkeys(
                    type_mapping[value] for value in requested_types
                    if value in type_mapping
                ))
                # 接受文档中的单字符串，也接受模型常见的字符串数组形式。
                modes = (tuple(type_modes) if type_modes else
                         tuple(args.get("modes") or ("stopped", "open", "harmonic")))
                audit_queries, tables = [], []
                for group in groups:
                    target = float(group["target_midi"])
                    sources = list(group.get("sources") or [])
                    index = (int(sources[0]["source_index"]) if sources
                             else args.get("source_index"))
                    event = ScoreEvent(
                        id=(f"src-{index:05d}" if index is not None else "pitch"),
                        kind="note", abc="",
                        pitches_midi=[round(target)], onset_ticks=0,
                        duration_ticks=960, bar=1, beat=1.0, attack=True,
                        section_id="sec", phrase_id=self.item["phrase_id"],
                    )
                    candidates = candidates_for_event(
                        event, self.item["input"]["normalized_tuning"]["open_midi"],
                        tolerance_cents=50.0, modes=modes,
                        # 批量查询默认 8 条候选，单音查询默认 12；显式传入优先。
                        max_candidates=int(args.get(
                            "max_candidates", 8 if is_batch_query else 12)),
                    )
                    candidate_rows = [candidate.to_dict() for candidate in candidates]
                    audit_queries.append({
                        "event_indices": list(dict.fromkeys(
                            int(source["event_index"]) for source in sources
                        )),
                        # Internal audit compatibility; this field is not
                        # included in the rendered tool observation.
                        "source_indices": list(dict.fromkeys(
                            int(source["source_index"]) for source in sources
                        )),
                        "sources": sources,
                        "target_midi": target,
                        "candidates": candidate_rows,
                    })
                    if sources:
                        tables.append(render_grouped_candidate_table(
                            target, candidate_rows, modes, sources))
                    else:
                        tables.append(render_pitch_candidate_table(
                            target, candidate_rows, modes))
                audit_result = {"queries": audit_queries}
                if skipped_indices:
                    tables.append(
                        "跳过不可解析或非发音序号｜"
                        + "、".join(str(self._source_to_event(index)) for index in dict.fromkeys(skipped_indices))
                    )
                result = {"text": "\n\n".join(tables)}
            elif name == "edit_plan":
                if "patches" in args:
                    raise ValueError("edit_plan only accepts jianzi_rows")
                patches = []
                current_replay = replay_patches(
                    self.item["baseline_plan"], self.accumulated_patches,
                    strict_before=False,
                )
                current_text = {
                    int(action["source_index"]): action.get("jianzi_text")
                    for action in current_replay.actions
                }
                current_actions = {
                    int(action["source_index"]): action
                    for action in current_replay.actions
                }
                skipped_not_editable: list[int] = []
                # Public rows are [event_index, text]. Convert to sparse
                # source indexes before replay; bars have no event index and
                # therefore fall into the existing non-editable warning path.
                public_rows = []
                unchanged_sources: list[int] = []
                for row in list(args.get("jianzi_rows") or []):
                    if isinstance(row, list) and row:
                        row = list(row)
                        try:
                            row[0] = self._event_to_source(row[0])
                        except (TypeError, ValueError):
                            pass
                        if (len(row) == 2 and isinstance(row[0], int)
                                and isinstance(row[1], str)
                                and row[0] in current_text
                                and current_text[row[0]] == row[1]):
                            unchanged_sources.append(row[0])
                    public_rows.append(row)
                patches.extend(expand_jianzi_rows(
                    public_rows, current_text=current_text,
                    allowed_indices=set(current_actions),
                    skipped_not_editable=skipped_not_editable,
                ))
                validate_patch_vocabulary(patches)
                batch_replay = replay_patches(self.item["baseline_plan"], patches,
                                              strict_before=False)
                batch_target_errors = (compare_replay_to_patch_targets(
                    batch_replay.actions, patches
                ) if batch_replay.valid else [])
                if batch_replay.valid and not batch_target_errors:
                    self.accumulated_patches.extend(patches)
                    cumulative = self.accumulated_patches
                else:
                    cumulative = None
                replay = (replay_patches(self.item["baseline_plan"], cumulative,
                                          strict_before=False)
                          if cumulative is not None else batch_replay)
                preview_valid = replay.valid
                preview_errors = list(replay.errors) + list(batch_target_errors)
                preview_warnings = ([{
                    "code": "ignored_noneditable_jianzi_rows",
                    "event_indices": [self._source_to_event(index)
                                      for index in skipped_not_editable],
                }] if skipped_not_editable else [])
                if replay.valid:
                    target_errors = compare_replay_to_patch_targets(
                        replay.actions, cumulative if cumulative is not None else patches
                    )
                    if target_errors:
                        preview_valid = False
                        preview_errors.extend(target_errors)
                if preview_valid and self.basic_fingering:
                    preview_valid, basic_report = validate_jianzi_only(
                        self.item, cumulative, toward_reference=False,
                        require_complete=False, historical=self.historical)
                    preview_errors = list(basic_report["problems"])
                    preview_warnings.extend(basic_report.get("warnings") or [])
                elif preview_valid:
                    # Pitch validation is public and must run for every
                    # Guqinizer invocation, including inference where
                    # ``toward_reference`` is false.  The optional private
                    # reference comparison inside validate_jianzi_only stays
                    # disabled in that case, so no sealed target text can
                    # leak into the tool observation.
                    preview_valid, quality_report = validate_jianzi_only(
                        self.item, cumulative, toward_reference=self.toward_reference,
                        require_complete=False, historical=self.historical)
                    preview_errors = list(quality_report["problems"])
                    preview_warnings.extend(quality_report.get("warnings") or [])
                audit_result = {
                    "valid": preview_valid,
                    "errors": preview_errors,
                    "warnings": preview_warnings,
                    "preview_actions": replay.actions,
                }
                # 公开结果只保留教师决策需要的信息；紧凑行无 patch_id，
                # applied_patch_ids 只是空串回执，公开与审计均不保留。
                preview_text = render_edit_preview(
                    replay.actions, cumulative or patches, preview_valid,
                    preview_errors, self.item, preview_warnings)
                unchanged_events = sorted({
                    self._source_to_event(index) for index in unchanged_sources
                })
                if unchanged_events:
                    preview_text += "\n未变更｜" + "、".join(
                        str(index) for index in unchanged_events
                    ) + "｜提交值与当前减字相同"
                remaining_warnings = [
                    warning for warning in preview_warnings
                    if warning.get("code") != "jianzi_pitch_mismatch"
                ]
                if remaining_warnings:
                    preview_text += "\n警告｜" + json.dumps(
                        remaining_warnings, ensure_ascii=False, separators=(",", ":"))
                result = {
                    "valid": preview_valid,
                    "text": preview_text,
                }
            else:
                raise ValueError(f"unknown tool: {name}")
            envelope = {"ok": True, "result": result}
            audit_envelope = {"ok": True, "result": audit_result or result}
        except Exception as exc:
            envelope = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            audit_envelope = envelope
        self.calls.append({"name": name, "arguments": deepcopy(args),
                           "result": audit_envelope,
                           "public_result": envelope})
        return envelope

def normalize_patches(patches: list[dict], *, prefix: str = "teacher") -> list[dict]:
    normalized = []
    for number, patch in enumerate(patches, 1):
        item = deepcopy(patch)
        item.setdefault("patch_id", f"{prefix}-{number:04d}")
        item.setdefault("before", {})
        item.setdefault("after", {})
        normalized.append(item)
    return normalized

def canonical_jianzi_text(text: str | None) -> str | None:
    """Normalize harmless surface variants before private reference review."""
    if text is None:
        return None
    digits = str.maketrans("一二三四五六七", "1234567")
    return re.sub(r"\s+", "", str(text)).replace("弦", "弦").translate(digits)

def infer_jianzi_text_patches(
    baseline: dict, references: list[dict | ReferenceAction]
) -> dict[str, list[dict]]:
    """Find Guqinizer targets from the written jianzi text only.

    Whether a second stage is needed is a surface-notation decision.  Do not
    infer it from mode/string/hui/finger fields: those fields may be partial,
    inherited, or marked conflicting even when the annotation contains an
    explicit text that should be reviewed.  A blank annotation is an explicit
    empty display target, regardless of parser confidence: surface comparison
    must be able to clear redundant glyphs covered by 吟/猱 or a compound
    multi-sound gesture.  Data-quality filtering handles wholly blank phrases
    before either model stage is generated.
    """
    baseline_by_index = {
        int(action["source_index"]): action
        for action in baseline.get("actions", [])
    }
    patches: list[dict] = []
    for reference in references:
        if isinstance(reference, dict):
            source_index = reference.get("source_index")
            confidence_class = reference.get("confidence_class", "weak")
            confidence = reference.get("confidence", 0.5)
            evidence = list(reference.get("evidence") or [])
            if reference.get("jianzi_text") is not None:
                expected = reference.get("jianzi_text")
            else:
                expected = reference.get("text")
        else:
            source_index = reference.source_index
            confidence_class = reference.confidence_class
            confidence = reference.confidence
            evidence = list(reference.evidence)
            expected = (reference.jianzi_text
                        if reference.jianzi_text is not None
                        else reference.text)
        try:
            source_index = int(source_index)
        except (TypeError, ValueError):
            continue
        # Reference parsing uses None for an ordinary blank source cell and
        # "" for a known compound-covered continuation.  At the surface layer
        # both mean the same target: render no glyph on this row.
        if expected is None:
            expected = ""
        if not isinstance(expected, str):
            continue
        before_action = baseline_by_index.get(source_index)
        if before_action is None:
            continue
        before = before_action.get("jianzi_text")
        if canonical_jianzi_text(before) == canonical_jianzi_text(expected):
            continue
        patches.append({
            "patch_id": f"text-target-{len(patches) + 1:04d}",
            "patch_type": "SET_JIANZI_TEXT",
            "source_index": source_index,
            "before": {"jianzi_text": before},
            "after": {"jianzi_text": expected},
            "confidence_class": confidence_class,
            "confidence": confidence,
            "evidence": evidence + ["jianzi_text_surface_diff"],
            "required_tools": [],
        })
    return {"patches": patches}

def is_all_empty_reference_phrase(item: dict) -> bool:
    """True when a phrase has no annotated glyph target at all.

    Such phrases provide no supervision for either Base/Fingering or
    Guqinizer and are excluded before API generation.  Materialized repeat
    rows are different: their visible target is the explicit repeat-omission
    placeholder, so a phrase containing one is not classified as all-empty.
    """
    notes = item.get("input", {}).get("notes_without_jianzi", [])
    if any(note.get("notation_omitted") for note in notes):
        return False
    # v9 repeat-semantics inference keeps repeat markers in provenance rather
    # than copying them into ``notation_omitted`` on each note.  Such a phrase
    # can legitimately have an all-empty visible target while still carrying
    # educational information (e.g. a ``再作``/``从ㄱ再作`` omission).  Keep it
    # eligible for the reasoned no-op final reply pass.
    repeat = item.get("provenance", {}).get("repeat_materialization", {}) or {}
    if repeat.get("markers") or repeat.get("marker_only_notes") or repeat.get("copy_markers"):
        return False
    references = item.get("reference_plan", {}).get("actions", [])
    for reference in references:
        value = (reference.get("jianzi_text")
                 if reference.get("jianzi_text") is not None
                 else reference.get("text"))
        if isinstance(value, str) and value.strip():
            return False
    return True

def expand_jianzi_rows(
    rows: list, *, current_text: dict[int, str | None] | None = None,
    allowed_indices: set[int] | None = None,
    skipped_not_editable: list[int] | None = None,
) -> list[dict]:
    """Expand concise ``[source_index, jianzi_text]`` rows into typed patches."""
    expanded = []
    seen = set()
    not_editable: list[int] = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"jianzi_rows[{number}] must be [source_index, jianzi_text]")
        index, text = row
        # Models occasionally serialize a JSON integer as a quoted decimal.
        # This is a recoverable transport error, unlike floats/booleans or
        # arbitrary strings, so normalize it before validation.
        if isinstance(index, str) and re.fullmatch(r"[0-9]+", index):
            index = int(index)
        if (not isinstance(index, int) or isinstance(index, bool)
                or not isinstance(text, str)):
            raise ValueError(f"jianzi_rows[{number}] has invalid types")
        if index in seen:
            raise ValueError(f"duplicate jianzi source_index: {index}")
        if allowed_indices is not None and index not in allowed_indices:
            # Do not abort on the first bad row.  Models often submit a
            # broad range containing bars/rests; returning the complete set
            # lets the next turn correct the whole batch in one pass.
            not_editable.append(index)
            seen.add(index)
            continue
        seen.add(index)
        before = (current_text or {}).get(index)
        if current_text is not None and before == text:
            continue
        expanded.append({"patch_type": "SET_JIANZI_TEXT", "source_index": index,
                         "before": {"jianzi_text": before},
                         "after": {"jianzi_text": text}})
    if not_editable and skipped_not_editable is not None:
        skipped_not_editable.extend(not_editable)
    elif not_editable:
        indices = ", ".join(str(index) for index in not_editable)
        raise ValueError(f"jianzi source_index is not editable: {indices}")
    return expanded

def validate_patch_vocabulary(patches: list[dict]) -> None:
    """The public edit protocol may only produce final jianzi text patches."""
    for patch in patches:
        if patch.get("patch_type") != "SET_JIANZI_TEXT":
            raise ValueError("edit_plan only accepts SET_JIANZI_TEXT patches")

def validate_jianzi_only(
    item: dict, patches: list[dict], *, toward_reference: bool = False,
    require_complete: bool = True,
    historical: dict[tuple[str, str], dict] | None = None,
) -> tuple[bool, dict]:
    """Validate the minimal text-only protocol.

    Syntax/coverage and safe reference omissions are hard conditions. Pitch is
    computed from the authored notation itself and is always advisory during
    the model conversation; the same report exposes a private eligibility bit
    for filtering training exports.
    """
    patches = normalize_patches(patches, prefix="jianzi")
    illegal = [patch for patch in patches
               if patch.get("patch_type") != "SET_JIANZI_TEXT"]
    problems = ([{"code": "non_jianzi_patch_in_minimal_protocol",
                  "patch_type": patch.get("patch_type")} for patch in illegal])
    replay = replay_patches(item["baseline_plan"], patches, strict_before=False)
    problems.extend(replay.errors)
    by_index = {int(action["source_index"]): action for action in replay.actions}
    if require_complete:
        for action in replay.actions:
            if action.get("jianzi_text") is None:
                problems.append({"source_index": int(action["source_index"]),
                                 "code": "pending_jianzi_text"})

    reference_review: list[dict] = []
    if toward_reference:
        # A Base/Fingering repeat-omission marker is an explicit display
        # value, not an ordinary empty target.  Guqinizer may leave it
        # unchanged, but must not erase it by submitting an empty string.
        baseline_by_index = {
            int(action["source_index"]): action.get("jianzi_text")
            for action in (item.get("baseline_plan", {}).get("actions") or [])
        }
        reference_by_index = {
            int(action["source_index"]): (
                action.get("jianzi_text")
                if action.get("jianzi_text") is not None
                else action.get("text")
            )
            for action in (item.get("reference_plan", {}).get("actions") or [])
        }
        for source_index, baseline_text in baseline_by_index.items():
            if (baseline_text == OMITTED_PLACEHOLDER
                    and reference_by_index.get(source_index) == OMITTED_PLACEHOLDER
                    and by_index.get(source_index, {}).get("jianzi_text") == ""):
                problems.append({
                    "source_index": source_index,
                    "code": "repeat_omission_marker_cleared",
                })
        for reference in item.get("reference_plan", {}).get("actions", []):
            if reference.get("confidence_class") not in {"verified", "weak"}:
                continue
            action = by_index.get(int(reference["source_index"]))
            if action is None:
                continue
            expected = reference.get("jianzi_text")
            actual = action.get("jianzi_text")
            equivalent = canonical_jianzi_text(actual) == canonical_jianzi_text(expected)
            if (reference.get("confidence_class") == "verified"
                    and expected is not None and not equivalent):
                reference_review.append({
                    "source_index": int(reference["source_index"]),
                    "code": "reference_jianzi_text_mismatch",
                    "expected": expected, "actual": actual,
                })
            elif bool(expected) and actual == "":
                reference_review.append({
                    "source_index": int(reference["source_index"]),
                    "code": "nonempty_reference_jianzi_dropped",
                    "expected": expected,
                })

    warnings: list[dict] = []
    pitch_report = {"counts": {"matched": 0, "mismatched": 0, "skipped": 0},
                    "details": []}
    try:
        current_source_indices = {
            int(note["index"])
            for note in item["input"].get("notes_without_jianzi", [])
            if note.get("index") is not None
        }
        notes = pitch_audit_notes(item, by_index, historical=historical)
        pitch_report = AUDIT.audit({
            "metadata": deepcopy(item["input"].get("metadata") or {}),
            "open_midi": deepcopy(
                (item["input"].get("normalized_tuning") or {}).get("open_midi")
            ),
            "notes": notes,
        }, 50.0)
        for detail in pitch_report.get("details", []):
            if (int(detail.get("index")) in current_source_indices
                    and detail.get("status") == "mismatched"):
                warnings.append({
                    "source_index": detail.get("index"),
                    "code": "jianzi_pitch_mismatch",
                    "pairs": detail.get("pairs") or [],
                    "possible_harmonic_state_mismatch": bool(
                        detail.get("possible_harmonic_state_mismatch")
                    ),
                })
    except Exception as exc:
        # Tool/parser availability never blocks creative notation generation.
        pitch_report = {"counts": {"matched": 0, "mismatched": 0, "skipped": 0},
                        "details": [], "unavailable": f"{type(exc).__name__}: {exc}"}

    current_pitch_details = [
        detail for detail in pitch_report.get("details", [])
        if detail.get("index") is not None
        and int(detail["index"]) in current_source_indices
    ]
    current_mismatch_count = sum(
        detail.get("status") == "mismatched" for detail in current_pitch_details
    )
    current_counts = {
        status: sum(detail.get("status") == status for detail in current_pitch_details)
        for status in ("matched", "mismatched", "skipped")
    }
    pitch_report["current_phrase_summary"] = {
        "total_notes": len(current_pitch_details),
        **current_counts,
        "match_rate": (
            round(current_counts["matched"] /
                  max(current_counts["matched"] + current_counts["mismatched"], 1), 6)
            if current_counts["matched"] + current_counts["mismatched"] else None
        ),
    }
    return not problems, {
        "valid": not problems,
        "problems": problems,
        "warnings": warnings,
        "private_reference_review": reference_review,
        "plan": {"actions": replay.actions},
        "patches": patches,
        "pitch_audit": pitch_report,
        "offline_quality": {
            # Pitch comparison is advisory.  A reliably parsed mismatch must
            # be shown to the teacher, but must not reject an otherwise valid
            # edit or silently remove the trajectory from training output.
            "training_eligible": not problems,
            "high_confidence_pitch_mismatches": current_mismatch_count,
        },
    }

def public_pitch_warning_source_indices(
    item: dict, *, historical: dict[tuple[str, str], dict] | None = None,
) -> set[int]:
    """Return current-phrase pitch warnings safe to expose in a Guqinizer prompt.

    This deliberately reuses the same audit path as ``edit_plan``.  It reads
    only the public Base plan, notation and prior model-generated history;
    sealed reference actions are never consulted.
    """
    _, report = validate_jianzi_only(
        item, [], toward_reference=False, require_complete=False,
        historical=historical,
    )
    return {
        int(warning["source_index"])
        for warning in report.get("warnings") or []
        if warning.get("code") == "jianzi_pitch_mismatch"
        and warning.get("source_index") is not None
    }

def can_accept_empty_tool_turn(
    item: dict, patches: list[dict] | None = None, *, historical: dict | None = None
) -> bool:
    """Return whether an empty tool call can legitimately finish a stage.

    ``target_patches`` are private reference guidance, not a compulsory edit
    contract.  Once the current plan is complete, the teacher may reasonably
    retain it after public musical analysis even if a private target exists.
    An incomplete Fingering scaffold, however, cannot become a no-op.
    """
    return validate_jianzi_only(
        item, patches or [], toward_reference=False, historical=historical
    )[0]
