"""Shared warning rendering."""
from __future__ import annotations

import json
from agents.ToolRuntime.warning_registry import ANCHOR_BLANK_TEXT

from agents.abc_to_jianzipu.inverse_baseline import AUDIT
from agents.abc_to_jianzipu.jianzi_renderer import hui_label, render_jianzi_surface
from agents.abc_to_jianzipu.repeat_materializer import OMITTED_PLACEHOLDER

from agents.ToolRuntime.warning_registry import COMPOUND_REPEAT_TEXT, INHERITED_GESTURE_WARNING_CODES, MISSING_ATTACK_TEXT, MISSING_REPEAT_ANCHOR_TEXT
from agents.ToolRuntime.warning_checks import event_index_for_source, pitch_label, validate_jianzi_only
from agents.ToolRuntime.hint_checks import repeated_pitch_hints

def anchor_repeat_mismatch_warning_text(warning: dict) -> str:
    historical = warning.get('history_event_index')
    detail = f'对应历史音序{historical}' if historical is not None else '超出标记至再作指令之间的历史序列'
    return (f'再作音高序列不匹配（{detail}；'
            f'请核对音序{warning.get("command_event_index")}的再作指令及省略范围）')


def open_reattack_warning_text(warning: dict) -> str:
    return (f'散音后紧接同弦普通拨弦（{warning["string"]}弦；可能打断散音余韵；'
            '考虑对前一音使用涓／勾剔／抹勾／抹挑／轮等同弦连续起音技法，或选择其他按音来增加丰富性）')

def inherited_gesture_warning_text(warning: dict) -> str:
    if warning.get('code') == 'tiao_on_first_string':
        return '一弦不能单独使用挑（请改用其他合适的右手指法）'
    code = warning['code']
    if code == 'slide_attack_missing_left_finger':
        return (f'{warning["technique"]}起音缺少左手按指'
                '（纯走手不附拨弦，起音请明确按指）')
    if code == 'same_position_finger_change':
        return (f'同弦同徽更换按指（{warning["string"]}弦{hui_label(warning["hui"])}，'
                f'{warning["previous_finger"]}换为{warning["finger"]}；'
                '若无换指需要，请沿用前序按指）')
    if code == 'jinfu_without_downward_walk':
        return '进复缺少向下走手前态（此前需先有退／下／注下等走手；可改用进／上／绰上）'
    if code == 'open_attack_ornament':
        return f'散音不能使用{warning["technique"]}'
    if code == 'li_nonadjacent_strings':
        strings = '、'.join(f'{s}弦' for s in warning['strings'])
        return f'历用弦不相邻（{strings}；历应使用同弦或连续相邻弦，请调整用弦）'
    if code == 'thumb_lower_left_position':
        return (f'左手按位不顺（前序{warning["previous_finger"]}{warning["from_string"]}弦{hui_label(warning["from_hui"])}，'
                f'当前大指{warning["to_string"]}弦{hui_label(warning["to_hui"])}；'
                '大指宜避免落在中指／名指按位的左下方，请考虑调整按指或弦徽）')
    if code == 'open_string_unplayable_gesture':
        return f'散音后不能承接{warning["technique"]}'
    if code == 'kneeling_taoqi_below_seventh_hui':
        return '跪指滔起只适用于七徽及更靠近岳山的徽位（请使用名指滔起）'
    if code == 'taoqi_without_thumb_origin':
        if warning.get('previous_string') is not None:
            return (f'滔起前置条件不满足（滔起要求此前在同弦({warning["string"]}弦)以左手大指按弦，'
                    f'但前段按音在{warning["previous_string"]}弦。请考虑调整前序减字的弦位与左手按指，'
                    '或为当前减字选择其他合适的技法）')
        return (f'滔起前置条件不满足（滔起要求同弦前一个按音使用左手大指按弦，'
                f'但前段实际使用的是{warning["previous_finger"]}。请考虑调整前序减字的左手按指，'
                '或为当前减字选择其他合适的技法）')
    if code == 'repeated_fu_without_walk':
        return '连续两次复之间没有新的走手，第二次复无可返回的起点'
    if code == 'fu_without_walk_origin':
        return '复缺少可返回的走手起点'
    if code == 'harmonic_ornament_without_stopped':
        return f'泛音区间内不能使用{warning["technique"]}（本音或前一音需显式为按音）'
    raise ValueError(code)

def pitch_mismatch_warning_text(warning: dict) -> str:
    def pitch(value):
        return f'{AUDIT.midi_name(value)}（MIDI {float(value):.2f}）'

    modes = {'harmonic': '泛音', 'open': '散音', 'stopped': '按音'}
    sounds = []
    for sound in warning.get('resolved_pitches') or []:
        mode = sound.get('mode')
        position = f'{sound["string"]}弦'
        if mode != 'open' and sound.get('hui') is not None:
            position += hui_label(sound['hui'])
        sounds.append(position + modes.get(mode, '') + '音高为' + pitch(sound['midi']))
    if not sounds:
        sounds = ['实得音高为' + '／'.join(pitch(v) for v in warning.get('actual_midis') or [])]
        if not warning.get('actual_midis'):
            sounds = []
    targets = warning.get('expected_midis') or []
    text = '音高不匹配'
    if sounds and targets and not warning.get('li_pitch_count_mismatch'):
        text += '（' + '；'.join(sounds) + '，与目标' + '／'.join(pitch(v) for v in targets) + '不符）'
    if warning.get('possible_harmonic_state_mismatch'):
        text += '（当前可能仍处于泛音状态；若此音应为按音，请明确写“按音”或先泛止）'
    if warning.get('li_pitch_count_mismatch'):
        text += (f'（“历”发出{len(warning.get("actual_midis") or [])}音，'
                 f'本处只有{len(warning.get("expected_midis") or [])}个连续目标起音；'
                 '多出的音没有对应目标，请修改本行减字）')
    origin = warning.get('origin_event_index', warning.get('origin_source_index'))
    if warning.get('origin_source_index') is not None and warning['origin_source_index'] != warning.get('source_index'):
        text += f'（本音由音序{origin}的多音减字覆盖；请修改音序{origin}的减字，而非只改本行）'
    return text

def harmonic_resonance_warning_text(warning: dict) -> str:
    strings = "、".join(f"{value}弦" for value in warning.get("strings") or [])
    if warning.get("same_pitch_different_hui"):
        return (
            f"泛音后紧接同弦不同徽泛音（{strings}；不适宜演奏；"
            "考虑对前一音使用锁／涓／勾剔／抹勾／抹挑／轮等同一徽位连续起音技法），或选择其他弦"
        )
    return f"泛音后紧接同弦起音（{strings}；会止住泛音，可能破坏余韵）"

def render_edit_preview(actions: list[dict], patches: list[dict], valid: bool,
                        errors: list[dict], item: dict | None = None,
                        warnings: list[dict] | None = None, *, include_hints: bool = True) -> str:
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
    walk_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "walk_span_too_large"
        and warning.get("source_index") is not None
    }
    walk_endpoint_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "walk_endpoint_same_as_start"
        and warning.get("source_index") is not None
    }
    walk_direction_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "walk_direction_reversed"
        and warning.get("source_index") is not None
    }
    consecutive_walk_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "consecutive_walks_without_attack"
        and warning.get("source_index") is not None
    }
    illegal_walk_blanks = {
        int(warning["source_index"])
        for warning in (warnings or [])
        if warning.get("code") == "illegal_blank_after_walk"
        and warning.get("source_index") is not None
    }
    resonance_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "open_to_stopped_same_string"
        and warning.get("source_index") is not None
    }
    harmonic_walk_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "walk_inside_harmonic_region"
        and warning.get("source_index") is not None
    }
    harmonic_resonance_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "harmonic_to_same_string_attack"
        and warning.get("source_index") is not None
    }
    fractional_harmonic_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "fractional_hui_in_harmonic"
        and warning.get("source_index") is not None
    }
    invalid_string_warnings = {
        int(warning['source_index']): warning
        for warning in (warnings or [])
        if warning.get('code') == 'invalid_string_number'
        and warning.get('source_index') is not None
    }
    pluck_pair_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "nonadjacent_pluck_pair"
        and warning.get("source_index") is not None
    }
    same_string_cuo_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "same_string_cuo_pair"
        and warning.get("source_index") is not None
    }
    same_finger_reach_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "same_finger_reach_too_large"
        and warning.get("source_index") is not None
    }
    low_hui_warnings = {
        int(warning["source_index"]): warning
        for warning in (warnings or [])
        if warning.get("code") == "low_hui_stopped_too_cramped"
        and warning.get("source_index") is not None
    }
    repeat_anchor_warnings_by_source: dict[int, list[dict]] = {}
    for warning in warnings or []:
        if (warning.get("code") in {"missing_repeat_anchor", "missing_repeat_instruction", 'repeat_anchor_followed_by_blanks',
                                   "repeat_anchor_instruction_overlap", "compound_repeat_instruction_overlap",
                                   "repeat_from_head_pitch_mismatch",
                                   "repeat_omission_marker_missing", 'repeat_numbered_start_missing',
                                   'repeat_numbered_pitch_mismatch', 'repeat_anchor_pitch_mismatch'}
                and warning.get("source_index") is not None):
            repeat_anchor_warnings_by_source.setdefault(int(warning['source_index']), []).append(warning)

    def warning_suffix(index: int) -> str:
        parts: list[str] = []
        extra_text = {
            'finite_gesture_uncovered_note': '前序复合技法已结束，此音未被覆盖（请补全减字或另起连续技法）',
            'repeat_plain_pitch_mismatch': '再作省略音高序列与前文不匹配（请明确重复起点，或补全减字）',
            'zhi_missing_continuous_gesture': '至某弦缺少可承接的连续技法（应搭配滚、拂、轮等；普通起音会中断承接）',
            'zhi_sweep_pitch_mismatch': '至某弦与连续弦序音高或声数不匹配（请核对起弦、中间省略音及终点弦）',
        }
        for warning in warnings or []:
            if warning.get('source_index') == index and warning.get('code') in extra_text:
                parts.append(':warning:' + extra_text[warning['code']])
        for warning in warnings or []:
            if warning.get('code') == 'ordinary_open_same_string_reattack' and warning.get('source_index') == index:
                parts.append(':warning:' + open_reattack_warning_text(warning))
            if warning.get('code') in INHERITED_GESTURE_WARNING_CODES and warning.get('source_index') == index:
                parts.append(':warning:' + inherited_gesture_warning_text(warning))
        pitch_warning = pitch_warnings.get(index)
        if pitch_warning:
            parts.append(':warning:' + pitch_mismatch_warning_text(pitch_warning))
        walk_warning = walk_warnings.get(index)
        if walk_warning:
            parts.append(
                ":warning:走手跨度过大（{technique}｜{start}→{end}｜{cents:g} cents；"
                "建议选择其他弦徽且此处不使用走手）".format(
                    technique=walk_warning.get("technique") or "走手",
                    start=hui_label(walk_warning.get("from_hui")),
                    end=hui_label(walk_warning.get("to_hui")),
                    cents=float(walk_warning.get("span_cents") or 0),
                )
            )
        endpoint_warning = walk_endpoint_warnings.get(index)
        if endpoint_warning:
            parts.append(":warning:走手终点与起点相同")
        direction_warning = walk_direction_warnings.get(index)
        if direction_warning:
            parts.append(":warning:走手方向错误")
        consecutive_walk_warning = consecutive_walk_warnings.get(index)
        if consecutive_walk_warning:
            count = consecutive_walk_warning.get("count") or 4
            parts.append(
                f":warning:连续{count}音无新音头（走手及置空音计入；建议补右手起音或改为按音）"
            )
        if index in illegal_walk_blanks:
            parts.append(":warning:非法置空")
        resonance_warning = resonance_warnings.get(index)
        if resonance_warning:
            strings = "、".join(
                f"{value}弦" for value in resonance_warning.get("strings") or []
            )
            parts.append(
                f":warning:散音后紧接同弦按音（{strings}；会止住散音，可能破坏韵味）"
            )
        harmonic_walk_warning = harmonic_walk_warnings.get(index)
        if index in fractional_harmonic_warnings:
            parts.append(":warning:泛音只能在徽位上弹出（请使用整数徽位，或先泛止并改用按音）")
        if index in invalid_string_warnings:
            numbers = '、'.join(str(n) for n in invalid_string_warnings[index]['strings'])
            parts.append(f':warning:非法弦号（{numbers}弦；古琴只有一至七弦，请修正弦号）')
        if harmonic_walk_warning:
            technique = harmonic_walk_warning.get("technique") or "走手"
            parts.append(f":warning:泛音区间内不能使用{technique}（请先泛止并重新起音）")
        harmonic_resonance_warning = harmonic_resonance_warnings.get(index)
        if harmonic_resonance_warning:
            parts.append(
                ":warning:" + harmonic_resonance_warning_text(harmonic_resonance_warning)
            )
        pluck_pair_warning = pluck_pair_warnings.get(index)
        if pluck_pair_warning:
            strings = "＋".join(
                f"{value}弦" for value in pluck_pair_warning.get("strings") or []
            )
            technique = pluck_pair_warning.get("technique") or "泼"
            parts.append(
                f":warning:{technique}双弦不相邻（{strings}；双弦{technique}应使用相邻弦）"
            )
        same_string_cuo_warning = same_string_cuo_warnings.get(index)
        if same_string_cuo_warning:
            string = (same_string_cuo_warning.get("strings") or ["?"])[0]
            parts.append(f":warning:撮的两根弦不能相同（{string}弦）")
        same_finger_warning = same_finger_reach_warnings.get(index)
        if same_finger_warning:
            parts.append(
                ":warning:同指跨位过大（{finger}｜{from_hui}{from_string}弦→"
                "{to_hui}{to_string}弦；建议考虑换指或手指组合）".format(
                    finger=same_finger_warning.get("finger") or "同指",
                    from_hui=hui_label(same_finger_warning.get("from_hui")),
                    from_string=same_finger_warning.get("from_string") or "?",
                    to_hui=hui_label(same_finger_warning.get("to_hui")),
                    to_string=same_finger_warning.get("to_string") or "?",
                )
            )
        if any(w.get('code') == 'missing_independent_attack'
               and w.get('source_index') == index for w in (warnings or [])):
            parts.append(':warning:' + MISSING_ATTACK_TEXT)
        low_hui_warning = low_hui_warnings.get(index)
        if low_hui_warning:
            string = low_hui_warning.get("string") or "?"
            boundary = (
                "1—4弦尽量避免四徽及以内按音"
                if isinstance(string, int) and string <= 4
                else "5弦尽量避免四徽以内按音"
            )
            parts.append(f":warning:低徽按音过紧（{boundary}）")
        for repeat_warning in repeat_anchor_warnings_by_source.get(index, []):
            if repeat_warning['code'] == 'repeat_anchor_followed_by_blanks':
                parts.append(':warning:' + ANCHOR_BLANK_TEXT)
                continue
            if repeat_warning['code'] == 'repeat_anchor_pitch_mismatch':
                parts.append(':warning:' + anchor_repeat_mismatch_warning_text(repeat_warning))
                continue
            if repeat_warning['code'] == 'repeat_numbered_start_missing':
                parts.append(f':warning:再作指定音序不存在或不在当前音之前（音序{repeat_warning["target_event_index"]}）')
            elif repeat_warning['code'] == 'repeat_numbered_pitch_mismatch':
                parts.append(f':warning:指定音序再作序列不匹配（从{repeat_warning["target_event_index"]}再作；按省略长度核验{repeat_warning["compared_events"]}音）')
            elif repeat_warning['code'] == 'repeat_from_head_pitch_mismatch':
                parts.append(':warning:从头再作序列与曲首或段首不匹配')
            elif repeat_warning['code'] == 'repeat_omission_marker_missing':
                affected = '、'.join(str(i) for i in repeat_warning.get('missing_event_indices', []))
                detail = f'（音序{affected}；请填写“{OMITTED_PLACEHOLDER}”）' if affected else ''
                parts.append(':warning:再作省略音缺少省略标记' + detail)
            elif repeat_warning["code"] == "repeat_anchor_instruction_overlap":
                parts.append(":warning:再作标记与再作指令不能重合")
            elif repeat_warning["code"] == "compound_repeat_instruction_overlap":
                parts.append(":warning:" + COMPOUND_REPEAT_TEXT)
            elif repeat_warning["code"] == "missing_repeat_instruction":
                parts.append(":warning:再作省略前缺少再作指令（前一个有效减字需含“再作”“从ㄱ再作”或“从头再作”）")
            else:
                parts.append(':warning:' + MISSING_REPEAT_ANCHOR_TEXT)
        return "".join(parts)

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
        lines.append(
            f'{visible_index}｜{jianpu}｜{display}｜{surface}{warning_suffix(index)}'
        )
    # ``edit_plan`` previews only the submitted rows above.  A cumulative
    # audit can nevertheless find pitch warnings inherited from the current
    # Base/earlier plan.  They must be shown to the model: otherwise the
    # acceptance gate keeps the turn alive for an issue it cannot identify.
    unmodified_warning_indices = sorted(
        ({int(w['source_index']) for w in (warnings or [])
          if w.get('code') in INHERITED_GESTURE_WARNING_CODES | {'ordinary_open_same_string_reattack'}
          and w.get('source_index') is not None}
         | set(pitch_warnings) | set(walk_warnings) | set(walk_endpoint_warnings)
         | set(walk_direction_warnings) | set(consecutive_walk_warnings)
         | illegal_walk_blanks
         | set(resonance_warnings)
         | set(harmonic_walk_warnings) | set(harmonic_resonance_warnings)
         | set(pluck_pair_warnings) | set(same_string_cuo_warnings)
         | set(same_finger_reach_warnings)
         | set(low_hui_warnings) | set(repeat_anchor_warnings_by_source)) - changed
    )
    if unmodified_warning_indices:
        unmodified_codes = {
            warning.get("code") for warning in (warnings or [])
            if warning.get("source_index") in unmodified_warning_indices
        }
        if unmodified_codes == {"jianzi_pitch_mismatch"}:
            lines.append("仍有未修改的音高警告｜以下行未包含在本轮提交；请核对或查询候选：")
        elif unmodified_codes == {"walk_span_too_large"}:
            lines.append("仍有未修改的走手警告｜以下行未包含在本轮提交；请核对前一按音与目标徽位：")
        elif unmodified_codes == {"walk_endpoint_same_as_start"}:
            lines.append("仍有未修改的走手警告｜以下行的走手终点与起点相同；请核对是否无需走手：")
        elif unmodified_codes == {"walk_direction_reversed"}:
            lines.append("仍有未修改的走手警告｜以下行的绰上/注下方向与徽位变化相反；请核对技法：")
        elif unmodified_codes == {"consecutive_walks_without_attack"}:
            lines.append("仍有未修改的走手警告｜以下行连续无新音头已超限；请补右手起音或改为按音：")
        elif unmodified_codes == {"illegal_blank_after_walk"}:
            lines.append("仍有未修改的非法置空警告｜以下行在单次走手后音高已变化，请填写减字：")
        elif unmodified_codes == {"open_to_stopped_same_string"}:
            lines.append("仍有未修改的余韵警告｜以下行紧接同弦散音后按音；请核对是否需要换弦或保留余韵：")
        elif unmodified_codes == {"walk_inside_harmonic_region"}:
            lines.append("仍有未修改的泛音警告｜以下行仍在泛音区间内使用走手；请先泛止并重新起音：")
        elif unmodified_codes == {"harmonic_to_same_string_attack"}:
            lines.append("仍有未修改的泛音余韵警告｜以下行紧接同弦泛音起音；请考虑换弦：")
        elif unmodified_codes == {"nonadjacent_pluck_pair"}:
            lines.append("仍有未修改的双弦警告｜以下行的拨/泼两弦不相邻；请核对右手可演奏性：")
        elif unmodified_codes == {"same_string_cuo_pair"}:
            lines.append("仍有未修改的撮警告｜以下行的两根弦相同；撮应由两根不同的弦组成：")
        elif unmodified_codes == {"same_finger_reach_too_large"}:
            lines.append("仍有未修改的同指跨位警告｜以下行与前一按音同指跨度过大；请考虑换指或手指组合：")
        elif unmodified_codes == {"low_hui_stopped_too_cramped"}:
            lines.append("仍有未修改的低徽按音警告｜以下行按音位置过紧；请考虑更高徽位、换弦或其他指法：")
        elif unmodified_codes == {"missing_repeat_anchor"}:
            lines.append('仍有未修改的再作警告｜' + MISSING_REPEAT_ANCHOR_TEXT + '：')
        else:
            lines.append("仍有未修改的警告｜以下行未包含在本轮提交；请核对：")
    for index in unmodified_warning_indices:
        action = by_index.get(index)
        visible_index = event_index_for_source(item, index) if item is not None else None
        visible_index = index if visible_index is None else visible_index
        if action is None:
            lines.append(
                f'{visible_index}｜{pitch_label(item, index) if item is not None else ""}｜'
                f"未找到动作｜-{warning_suffix(index)}"
            )
            continue
        surface = render_jianzi_surface(
            action, omitted_placeholder=OMITTED_PLACEHOLDER
        ) or "—"
        lines.append(
            f'{visible_index}｜{pitch_label(item, index) if item is not None else ""}｜'
            f'未修改｜{surface}{warning_suffix(index)}'
        )
    if errors:
        lines.append("错误｜" + json.dumps(errors, ensure_ascii=False, separators=(",", ":")))
    if valid and item is not None and include_hints:
        for hint in repeated_pitch_hints(item, actions):
            indices=hint['source_indices']
            first=event_index_for_source(item, indices[0])
            last=event_index_for_source(item, indices[-1])
            lines.append(f"音序{first if first is not None else indices[0]}–"
                         f"{last if last is not None else indices[-1]}｜:hint:{hint['text']}")
    return "\n".join(lines)

def public_plan_warning_messages(
    item: dict, *, historical: dict[tuple[str, str], dict] | None = None,
) -> dict[int, list[str]]:
    """Return public, row-local warning text for an existing plan.

    Guqinizer starts from a concrete Base plan.  Its first public score preview
    must expose the same advisory state that a no-op ``edit_plan`` would show,
    without adding a synthetic follow-up user turn.  Keep this mapping free of
    private/reference data so it is safe to render directly in the prompt.
    """
    _, report = validate_jianzi_only(
        item, [], toward_reference=False, require_complete=False,
        historical=historical,
    )
    messages: dict[int, list[str]] = {}
    for warning in report.get("warnings") or []:
        source_index = warning.get("source_index")
        if source_index is None:
            continue
        code = str(warning.get("code") or "")
        if code == "jianzi_pitch_mismatch":
            text = pitch_mismatch_warning_text(warning)
        elif code == 'ordinary_open_same_string_reattack':
            text = open_reattack_warning_text(warning)
        elif code in INHERITED_GESTURE_WARNING_CODES:
            text = inherited_gesture_warning_text(warning)
        elif code == "missing_independent_attack":
            text = MISSING_ATTACK_TEXT
        elif code == "illegal_blank_after_walk":
            text = "非法置空"
        elif code == "walk_span_too_large":
            text = "走手跨度过大"
        elif code == "walk_endpoint_same_as_start":
            text = "走手终点与起点相同"
        elif code == "walk_direction_reversed":
            text = "走手方向错误"
        elif code == "consecutive_walks_without_attack":
            count = warning.get("count") or 4
            text = f"连续{count}音无新音头"
        elif code == "open_to_stopped_same_string":
            strings = "、".join(f"{value}弦" for value in warning.get("strings") or [])
            text = f"散音后紧接同弦按音（{strings}；会止住散音，可能破坏韵味）"
        elif code == "walk_inside_harmonic_region":
            technique = warning.get("technique") or "走手"
            text = f"泛音区间内不能使用{technique}"
        elif code == "fractional_hui_in_harmonic":
            text = "泛音只能在徽位上弹出（请使用整数徽位，或先泛止并改用按音）"
        elif code == 'invalid_string_number':
            numbers = '、'.join(str(n) for n in warning['strings'])
            text = f'非法弦号（{numbers}弦；古琴只有一至七弦，请修正弦号）'
        elif code == "harmonic_to_same_string_attack":
            text = harmonic_resonance_warning_text(warning)
        elif code == "nonadjacent_pluck_pair":
            text = f"{warning.get('technique') or '泼'}双弦不相邻"
        elif code == "same_string_cuo_pair":
            text = "撮的两根弦相同"
        elif code == "same_finger_reach_too_large":
            text = "同指跨位过大"
        elif code == "low_hui_stopped_too_cramped":
            text = "低徽按音过紧"
        elif code == "missing_repeat_anchor":
            text = MISSING_REPEAT_ANCHOR_TEXT
        elif code == 'repeat_anchor_followed_by_blanks':
            text = ANCHOR_BLANK_TEXT
        elif code == "missing_repeat_instruction":
            text = "再作省略前缺少再作指令"
        elif code == "repeat_anchor_instruction_overlap":
            text = "再作标记与再作指令不能重合"
        elif code == "compound_repeat_instruction_overlap":
            text = COMPOUND_REPEAT_TEXT
        elif code == 'finite_gesture_uncovered_note':
            text = '前序复合技法已结束，此音未被覆盖（请补全减字或另起连续技法）'
        elif code == 'repeat_plain_pitch_mismatch':
            text = '再作省略音高序列与前文不匹配（请明确重复起点，或补全减字）'
        elif code == 'zhi_missing_continuous_gesture':
            text = '至某弦缺少可承接的连续技法（应搭配滚、拂、轮等；普通起音会中断承接）'
        elif code == 'zhi_sweep_pitch_mismatch':
            text = '至某弦与连续弦序音高或声数不匹配（请核对起弦、中间省略音及终点弦）'
        elif code == 'repeat_from_head_pitch_mismatch':
            text = '从头再作序列与曲首或段首不匹配'
        elif code == 'repeat_numbered_start_missing':
            text = f'再作指定音序不存在或不在当前音之前（音序{warning["target_event_index"]}）'
        elif code == 'repeat_numbered_pitch_mismatch':
            text = f'指定音序再作序列不匹配（从{warning["target_event_index"]}再作；按省略长度核验{warning["compared_events"]}音）'
        elif code == 'repeat_anchor_pitch_mismatch':
            text = anchor_repeat_mismatch_warning_text(warning)
        elif code == 'repeat_omission_marker_missing':
            text = '再作省略音缺少省略标记'
            affected = '、'.join(str(i) for i in warning.get('missing_event_indices', []))
            if affected:
                text += f'（音序{affected}；请填写“{OMITTED_PLACEHOLDER}”）'
        else:
            continue
        messages.setdefault(int(source_index), []).append(text)
    return messages
