from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable

from .jianzi_renderer import hui_label, render_jianzi_surface
from .pitch_candidates import abc_chord_tones
from .repeat_materializer import OMITTED_PLACEHOLDER
from .trajectory_replay import compare_replay_to_patch_targets, replay_patches


STAGE_PATCH_TYPES = {
    "fingering_agent": {"SET_JIANZI_TEXT"},
    "guqinization": {"SET_JIANZI_TEXT"},
}

STAGE_PROMPTS = {
    "fingering_agent": (
        "你是 Fingering Agent。输入是已确定的弦徽主干；依据当前谱面和工具结果补全、优化左右手指法。"
        "不得看到或猜测标注答案；改换弦徽必须核验音高。用简短可核验摘要衔接局部编辑，最终输出 JSON patch。"
    ),
    "guqinization": (
        "你是 Guqinizer。输入是左右手已经完整的演奏主干。使用工具结果处理走手、技法和装饰，"
        "不得改变节奏或无依据地改变音高。调用 edit_plan 前先审阅完整当前段，合并处理明确且有价值的"
        "局部改进；一次通过的 edit_plan 会直接提交并结束，所以不要只修第一个差异。允许保留音高正确、"
        "可弹且符合上下文的替代指法，不要求机械复刻。edit_plan 按轮累计；收到反馈后只提交新增或"
        "纠正的局部 patch，不要重发未变化的整段。用简短可核验摘要衔接调用，最终输出 JSON patch。"
    ),
    "plan_auditor": (
        "你是只读审计 Agent。核对 patch 重放、目标字段和音高，不得自行改写答案。"
        "只能根据工具结果给出 passed 或 failed。"
    ),
}


def _assistant_tool_call(call_id: str, name: str, arguments: dict,
                         summary: str) -> dict[str, Any]:
    return {
        "role": "assistant", "content": summary,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    }


def _tool_message(call_id: str, name: str, result: dict) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "name": name,
            "content": json.dumps(result, ensure_ascii=False)}


def _public_input(item: dict) -> dict:
    metadata = deepcopy(item["input"]["metadata"])
    # Avoid repeating bulky score-level sections and raw tuning callbacks in
    # every phrase. Normalized tuning and the local section live in handoff.
    metadata.pop("sections", None)
    metadata.pop("tuning", None)
    handoff = deepcopy(item["input"].get("phrase_handoff") or {})
    older_refs = handoff.get("older_context_refs") or []
    phrase_context = {
        "current_phrase_id": handoff.get("phrase_id", item.get("phrase_id")),
        "current_phrase": deepcopy(handoff.get(
            "current_phrase", item["input"]["notes_without_jianzi"])),
        "previous_phrase": deepcopy(handoff.get("previous_phrase")),
        "older_history": {
            "available": bool(older_refs),
            "phrase_count": len(older_refs),
            "instruction": (
                "默认不加载；确有需要时先调用 list_context，再用 expand_context 展开一个 phrase。"
                if older_refs else "无更早 phrase"
            ),
        },
    }
    def publicize_actions(plan: dict | list | None, notes: list[dict[str, Any]]) -> dict | list:
        if isinstance(plan, list):
            result = deepcopy(plan)
            action_list = result
        else:
            result = deepcopy(plan or {})
            action_list = result.get("actions") or []
        source_to_event = {
            int(note["index"]): note.get("event_index")
            for note in notes if note.get("event_index") is not None
        }
        notes_by_source = {
            int(note["index"]): note for note in notes
            if note.get("index") is not None
        }
        visible_actions = []
        for action in action_list:
            source = action.pop("source_index", None)
            if source is None:
                action["event_index"] = None
                visible_actions.append(action)
                continue
            source = int(source)
            if source not in source_to_event:
                note = notes_by_source.get(source) or {}
                # Legacy phrase handoffs sometimes retain an empty action for
                # a barline.  It is not an event and cannot appear in the
                # public event-index protocol; omit only that non-sounding
                # artifact.  A real note without event_index remains a hard
                # schema failure below.
                if (str(note.get("abc") or "") == "|"
                        or str(note.get("duration") or "") == "小节线"
                        or str(note.get("jianpu") or "") == "|"):
                    continue
                raise ValueError(
                    f"public trajectory protocol requires event_index for source_index={source}"
                )
            action["event_index"] = source_to_event[source]
            visible_actions.append(action)
        if isinstance(result, list):
            result[:] = visible_actions
        else:
            result["actions"] = visible_actions
        return result

    current_notes = phrase_context["current_phrase"]
    public_baseline = publicize_actions(item.get("baseline_plan"), current_notes)
    previous = phrase_context.get("previous_phrase")
    if previous:
        previous = deepcopy(previous)
        previous["actions"] = publicize_actions(previous.get("actions"), previous.get("notes") or [])
        phrase_context["previous_phrase"] = previous
    return {
        "score_context": {
            "metadata": metadata,
            "tuning": deepcopy(item["input"].get("normalized_tuning")),
        },
        "phrase_context": phrase_context,
        "notes": handoff.get(
            "current_phrase",
            handoff.get("current_notes", item["input"]["notes_without_jianzi"]),
        ),
        "baseline_plan": public_baseline,
    }


SLIDE_DIRECTION = {"绰": "上", "注": "下"}
# 左手技法全集：attack=false（无右手续音）合法化的技法集。
# 独立式标注（单独一行“猱”/“进复七徽”/“带起”等，约 3,300 个）全部是
# attack=false 无右手；复合式（“勾6弦吟”）则 attack=true 带右手。
SLIDE_TECHNIQUES = {"绰", "注", "上", "下", "淌", "吟", "猱", "往来", "进复",
                    "退复", "撞", "逗", "带起", "掐起", "抓起", "滔起"}


def slide_surface(action: dict) -> str | None:
    """走手续音的谱面减字，如 [绰上七徽九分]；非走手续音返回 None。"""
    techniques = [technique for technique in (action.get("techniques") or [])
                  if technique in SLIDE_TECHNIQUES]
    if action.get("attack") or not techniques:
        return None
    if all(technique == "吟" for technique in techniques):
        return "[" + "".join(techniques) + "]"
    parts = [technique + SLIDE_DIRECTION.get(technique, "") for technique in techniques]
    return "[" + "".join(parts) + hui_label(action.get("hui")) + "]"


def _action_columns(action: dict[str, Any] | None, *, readonly: bool = False) -> tuple[str, str, str, str, str]:
    if action is None:
        # A read-only handoff is already confirmed context: absence of a
        # materialized action means the glyph is intentionally not displayed,
        # never that the student should fill it.  The editable/current phrase
        # keeps its explicit pending marker.
        return "未完成", "左手待定", "右手待定", "无", "[空]" if readonly else "[待定]"
    explicit = action.get("jianzi_text")
    legacy_pending = str(explicit or "").strip().strip("[]") in {
        "减字待填写", "待填写", "待定",
    }
    if action.get("mode") is None:
        surface = (("[空]" if readonly else "[减字待填写]")
                   if explicit is None or (readonly and (legacy_pending or explicit == "")) else
                   ("" if explicit == "" else f"[{explicit}]"))
        return ("取音方式／弦徽待定", "左手待定",
                "右手待定" if action.get("attack") else "续音",
                "无", surface)
    mode = {"open": "散音", "stopped": "按音", "harmonic": "泛音"}.get(
        action.get("mode"), str(action.get("mode") or "待定"))
    string = action.get("string")
    position = f"{string}弦" if string is not None else "弦待定"
    if action.get("mode") != "open" and action.get("hui") is not None:
        position += hui_label(action.get("hui"))
    partner = ""
    if isinstance(action.get("string2"), int):
        partner_mode = action.get("mode2")
        partner = (f'＋散{action["string2"]}弦'
                   if partner_mode == "open" and action.get("mode") != "open"
                   else f'＋{action["string2"]}弦')
    left = action.get("left_finger") or ("左手待定" if action.get("mode") == "stopped" else "无")
    compound = action.get("compound_gesture")
    right = (f"复合动作：{compound}" if compound else
             action.get("right_finger") or ("右手待定" if action.get("attack") else "续音"))
    prefixes = set(action.get("pre_attack_techniques") or [])
    techniques = "、".join(
        f"{technique}（前置）" if technique in prefixes else technique
        for technique in (action.get("techniques") or [])
    ) or "无"
    surface = render_jianzi_surface(
        action, prefer_text=True, omitted_placeholder=OMITTED_PLACEHOLDER
    )
    if readonly and (explicit is None or legacy_pending or explicit == ""):
        surface = "[空]"
    return mode + position + partner, left, right, techniques, surface


def chord_jianpu_label(abc: Any, written: Any, jianpu_alt: Any = None) -> str:
    """和弦音的完整简谱：第二个谱字本就存储在 jianpu_alt 中，直接拼用。

    非和弦音符的 jianpu_alt 是装饰性替代音，不参与显示。
    """
    written = str(written)
    if jianpu_alt and abc_chord_tones(str(abc or "")):
        return f"{written} {jianpu_alt}"
    return written


def _render_phrase_lines(
    notes: list[dict[str, Any]], actions: list[dict[str, Any]], *,
    readonly: bool = False,
    pitch_warning_source_indices: set[int] | None = None,
) -> str:
    event_to_source = {
        int(note["event_index"]): int(note["index"])
        for note in notes if note.get("event_index") is not None
    }
    by_index = {}
    for action in actions:
        if action.get("source_index") is not None:
            key = int(action["source_index"])
        elif action.get("event_index") is not None:
            event_index = int(action["event_index"])
            if event_index not in event_to_source:
                raise ValueError(f"unknown public event_index: {event_index}")
            key = event_to_source[event_index]
        else:
            continue
        by_index[key] = action
    lines = ["序号｜简谱｜ABC｜时值｜谱面减字"]
    pitch_warning_source_indices = pitch_warning_source_indices or set()
    for note in notes:
        source_index = int(note["index"])
        index = note.get("event_index")
        written = note.get("jianpu") or note.get("jianpu_alt") or "休止"
        # 和弦音（撮等）的简谱列显示完整双谱字（第二个谱字存储于 jianpu_alt）。
        jianpu = chord_jianpu_label(note.get("abc"), written, note.get("jianpu_alt"))
        stored_action = by_index.get(source_index)
        if note.get("duration") == "小节线" or note.get("abc") == "|":
            lines.append("小节线")
            continue
        if index is None:
            raise ValueError(
                f"public trajectory protocol requires event_index for source_index={source_index}"
            )
        # A stored empty action is how the plan represents a rest or tie.  In
        # a read-only handoff it must retain that musical meaning, rather than
        # being rendered as a generic empty glyph.
        elif (stored_action is None or (readonly and not str(
                stored_action.get("jianzi_text") or "").strip())) and (
                    str(note.get("abc") or "").startswith("z")
                    or str(jianpu).startswith("0")):
            action, left, right, techniques, surface = "休止", "—", "—", "—", "[—]"
        elif (stored_action is None or (readonly and not str(
                stored_action.get("jianzi_text") or "").strip())) and (
                    str(note.get("abc") or "").startswith("-")
                    or "延音" in str(jianpu)):
            action, left, right, techniques, surface = "延音", "—", "承接前音", "—", "[续音]"
        else:
            action, left, right, techniques, surface = _action_columns(
                stored_action, readonly=readonly
            )
        if source_index in pitch_warning_source_indices:
            surface += ":warning:音高不匹配"
        lines.append(
            f'{index}｜{jianpu}｜{note.get("abc") or "-"}｜{note.get("duration") or "-"}｜'
            f'{surface}'
        )
    return "\n".join(lines)


def render_public_prompt(item: dict, stage: str) -> str:
    """Render the student-visible task as compact, compiler-like score text."""
    public = _public_input(item)
    score = public["score_context"]
    metadata = score.get("metadata") or {}
    tuning = score.get("tuning") or {}
    context = public["phrase_context"]
    previous = context.get("previous_phrase")

    def section_marker(notes: list[dict[str, Any]]) -> str | None:
        for note in notes:
            section = note.get("section") or {}
            marker = str(section.get("marker") or "").strip()
            # Every note carries its owning section marker.  Only a note
            # marked as the actual section start should emit the paragraph
            # label; continuation phrases must not repeat it.
            if marker and section.get("start"):
                return marker
        return None
    lines = [
        f'谱名｜{metadata.get("title") or metadata.get("score_title") or item.get("score_key", "未命名")}',
        f'调弦｜{tuning.get("name") or "未知"}｜{tuning.get("open_midi") or []}',
        f'当前段｜{context.get("current_phrase_id")}',
        ('泛音区间｜此段开始时仍然处于泛音区间；无须在段首重复添加泛起；要结束泛音区间，使用泛止'
         if item.get("harmonic_region_at_start") else
         '泛音区间｜段首未处于泛音区间'),
        "泛音区间提示｜常规写法：要进入泛音区间，在减字开头添加“泛起”；要结束泛音区间时，可在当前减字末尾添加“泛止”，也可在随后的延音行单独填写“泛止”。",
        "",
    ]
    if previous:
        previous_marker = section_marker(previous.get("notes") or [])
        lines.extend([
            f'【只读前一段 {previous.get("phrase_id")}｜{previous.get("status", "已确认")}】',
            *([previous_marker] if previous_marker else []),
            _render_phrase_lines(previous.get("notes") or [], previous.get("actions") or [], readonly=True),
            "注｜前一段已在此完整提供，直接使用；仅需更早段时才先 list_context 再 expand_context。",
            "",
        ])
    lines.extend([
        f'【当前段 {context.get("current_phrase_id")}｜待编辑】',
        *([section_marker(public["notes"])] if section_marker(public["notes"]) else []),
        _render_phrase_lines(
            public["notes"], public["baseline_plan"].get("actions") or [],
            pitch_warning_source_indices=(
                {int(index) for index in item.get("public_pitch_warning_source_indices") or []}
                if stage in {"guqinization", "single_stage"} else set()
            ),
        ),
    ])
    older = context.get("older_history") or {}
    if older.get("available"):
        lines.extend(["", f'更早段｜共 {older.get("phrase_count", 0)} 段；需要时先 list_context，再按需展开一段。'])
    return "\n".join(lines)


def synthesize_stage(
    item: dict, stage: str, patches: list[dict], *,
    summary_writer: Callable[[str, dict], str] | None = None,
) -> tuple[dict, dict]:
    """Create a teacher-forced but student-visible, executable conversation."""
    if stage not in STAGE_PATCH_TYPES:
        raise ValueError(stage)
    public_input = _public_input(item)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": STAGE_PROMPTS[stage]},
        {"role": "user", "content": json.dumps({
            "task": stage, "input": public_input,
            "instruction": "检查当前方案；需要时调用工具，最后提交结构化 patch。",
        }, ensure_ascii=False)},
    ]
    current = deepcopy(item["baseline_plan"])
    tool_log = []
    # One musical event is one local decision unit. Mode, position and finger
    # changes for that event must be observed and applied together; emitting a
    # fixed inspect/apply pair for every field teaches an artificial workflow
    # and wastes the online tool-round budget.
    grouped: OrderedDict[int, list[dict]] = OrderedDict()
    for patch in patches:
        grouped.setdefault(int(patch["source_index"]), []).append(patch)
    for number, (source_index, patch_group) in enumerate(grouped.items(), 1):
        inspect_id = f"call-{stage}-{number:04d}-inspect"
        inspect_args = {"source_index": source_index}
        baseline_action = next((action for action in current.get("actions", [])
                                if int(action["source_index"]) == source_index), None)
        inspect_result = {
            "ok": True, "source_index": source_index,
            "current_action": baseline_action,
            "note": next((note for note in public_input["notes"]
                          if int(note["index"]) == source_index), None),
        }
        default_summary = "先读取目标音及当前动作，确认需要调整的字段。"
        summary = summary_writer(default_summary, {
            "stage": stage, "tool": "inspect_plan_context", "result": inspect_result,
        }) if summary_writer else default_summary
        messages.append(_assistant_tool_call(inspect_id, "inspect_plan_context", inspect_args, summary))
        messages.append(_tool_message(inspect_id, "inspect_plan_context", inspect_result))
        tool_log.append({"name": "inspect_plan_context", "arguments": inspect_args,
                         "result": inspect_result})

        apply_id = f"call-{stage}-{number:04d}-apply"
        apply_args = {"patches": patch_group}
        replay = replay_patches(current, patch_group)
        apply_result = {"ok": replay.valid, "applied_patch_ids": replay.applied_patch_ids,
                        "errors": replay.errors,
                        "updated_action": next((action for action in replay.actions
                                                if int(action["source_index"]) == source_index), None)}
        summaries = {
            "CHANGE_MODE": "当前取音方式需要调整；提交局部模式修改并立即重放。",
            "REPOSITION": "当前弦徽位置需要调整；提交局部换位并立即重放。",
            "CHANGE_FINGER": "音高位置保持不变，提交局部指法修改。",
            "ADD_TECHNIQUE": "该处需要加入明确技法；提交局部技法 patch。",
            "CHANGE_ATTACK": "该处是无右手续音或恢复起音；明确修改 attack 与右手。",
            "SET_COMPOUND_GESTURE": "该处是不可拆分的多声复合动作；保留完整动作名。",
            "REMOVE_TECHNIQUE": "该技法不应保留；提交删除 patch。",
            "NO_OP": "当前动作已满足目标约束，提交不修改决定。",
        }
        patch_types = [patch["patch_type"] for patch in patch_group]
        default_summary = "同一事件的相关修改应原子提交：" + "；".join(
            summaries[patch_type] for patch_type in patch_types
        )
        summary = summary_writer(default_summary, {
            "stage": stage, "tool": "apply_plan_patch", "inspect_result": inspect_result,
            "patch_types": patch_types, "result": apply_result,
        }) if summary_writer else default_summary
        messages.append(_assistant_tool_call(apply_id, "apply_plan_patch", apply_args, summary))
        messages.append(_tool_message(apply_id, "apply_plan_patch", apply_result))
        tool_log.append({"name": "apply_plan_patch", "arguments": apply_args,
                         "result": apply_result})
        if not replay.valid:
            break
        current = {"actions": replay.actions}

    audit_id = f"call-{stage}-audit"
    audit_args = {"patch_ids": [patch["patch_id"] for patch in patches]}
    final_replay = replay_patches(item["baseline_plan"], patches)
    problems = [*final_replay.errors,
                *compare_replay_to_patch_targets(final_replay.actions, patches)]
    audit_result = {"ok": not problems, "replay_passed": final_replay.valid,
                    "target_fields_passed": not problems, "problems": problems}
    messages.append(_assistant_tool_call(
        audit_id, "audit_plan_patch", audit_args,
        "局部修改已完成，调用硬审计核对重放结果和目标字段。",
    ))
    messages.append(_tool_message(audit_id, "audit_plan_patch", audit_result))
    tool_log.append({"name": "audit_plan_patch", "arguments": audit_args,
                     "result": audit_result})
    messages.append({
        "role": "assistant",
        "content": json.dumps({
            "status": "completed" if not problems else "failed",
            "patches": patches if not problems else [],
            "verified_patch_ids": [patch["patch_id"] for patch in patches] if not problems else [],
        }, ensure_ascii=False),
    })
    public = {
        "schema_version": "agent-messages-1.1",
        "sample_id": f"{item['trajectory_id']}-{stage}", "agent_stage": stage,
        "generation_mode": "inverse_teacher_forced_grouped_replay", "messages": messages,
        "provenance": {"source_trajectory_id": item["trajectory_id"],
                       "score_key": item["score_key"],
                       "score_family_id": item["score_family_id"], "split": item["split"]},
        "verification": {"tool_calls_executed": True, "replay_passed": final_replay.valid,
                         "target_fields_passed": not problems},
    }
    private = {
        "sample_id": public["sample_id"], "teacher_private": {
            "reference_actions": item["reference_plan"]["actions"],
            "target_patches": patches, "tool_execution_log": tool_log,
        }, "verification": public["verification"],
    }
    return public, private


def synthesize_audit(item: dict, all_patches: list[dict]) -> tuple[dict, dict]:
    public_input = _public_input(item)
    call_id = "call-plan-audit-0001"
    replay = replay_patches(item["baseline_plan"], all_patches)
    problems = [*replay.errors, *compare_replay_to_patch_targets(replay.actions, all_patches)]
    result = {"ok": not problems, "replay_passed": replay.valid,
              "target_fields_passed": not problems, "problems": problems}
    messages = [
        {"role": "system", "content": STAGE_PROMPTS["plan_auditor"]},
        {"role": "user", "content": json.dumps({
            "task": "audit_plan", "input": public_input,
            "submitted_patches": all_patches,
        }, ensure_ascii=False)},
        _assistant_tool_call(call_id, "audit_plan_patch",
                             {"patch_ids": [p["patch_id"] for p in all_patches]},
                             "执行确定性重放与目标字段审核。"),
        _tool_message(call_id, "audit_plan_patch", result),
        {"role": "assistant", "content": json.dumps({
            "status": "passed" if not problems else "failed", "problems": problems,
        }, ensure_ascii=False)},
    ]
    public = {
        "schema_version": "agent-messages-1.1",
        "sample_id": f"{item['trajectory_id']}-plan_auditor",
        "agent_stage": "plan_auditor", "generation_mode": "deterministic_audit",
        "messages": messages,
        "provenance": {"source_trajectory_id": item["trajectory_id"],
                       "score_key": item["score_key"],
                       "score_family_id": item["score_family_id"], "split": item["split"]},
        "verification": {"tool_calls_executed": True, "replay_passed": replay.valid,
                         "target_fields_passed": not problems},
    }
    private = {"sample_id": public["sample_id"], "teacher_private": {
        "reference_actions": item["reference_plan"]["actions"],
        "target_patches": all_patches}, "verification": public["verification"]}
    return public, private
