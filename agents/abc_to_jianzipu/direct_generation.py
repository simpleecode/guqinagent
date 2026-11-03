"""Tool-free final-notation protocol shared by Direct SFT and inference."""
from __future__ import annotations

import json
import re
from copy import deepcopy

PROTOCOL = 'guqin-direct-1.0'
SYSTEM = (
    '你是古琴减字谱编配助手。根据简谱、ABC、时值、调弦和只读上下文，'
    '直接为当前段生成完整的最终减字谱，兼顾音高、可演奏性、指法衔接和古琴韵味。'
    '结合按音、散音、泛音以及合适的走手与装饰技法；徽位、按指和音色的省略按局部状态继承。'
    '泛起进入泛音区间，泛止退出；明确休止可结束泛音状态。空显示不代表删除声音，'
    '仅在延音、休止、复合动作覆盖或重复省略等合适场合使用。'
    '输出对象必须且只能有 jianzi_rows 字段，其值为当前段全部事件的[序号,减字文本]数组。'
    '序号使用输入的演奏事件序号，不输出小节线；需要空显示的事件也输出空字符串。'
    '只输出最终 JSON，不输出推理过程、解释、Markdown或工具调用。'
)


def user_from_public_prompt(text: str) -> str:
    """Retain the original public musical context, removing tool-use directions."""
    lines = [re.split(r':(?:warning|hint):', line, maxsplit=1)[0] for line in text.splitlines()
             if not any(name in line for name in ('list_context', 'expand_context', 'get_pitch_candidates', 'edit_plan'))]
    return '\n'.join(lines).rstrip() + '\n\n输出｜仅输出当前段完整的 {"jianzi_rows":[[序号,"减字文本"],...]}。'


def event_sources(item: dict) -> dict[int, int]:
    result = {}
    for note in item['input']['notes_without_jianzi']:
        event = note.get('event_index')
        if event is None:
            continue
        if isinstance(event, bool) or not isinstance(event, int) or event in result:
            raise ValueError('missing/duplicate/noninteger event index')
        result[event] = int(note['index'])
    if not result:
        raise ValueError('phrase has no event rows')
    return result


def parse_answer(text: str, item: dict) -> tuple[list[list], dict]:
    """Strict full-artifact parsing: reject missing, duplicated or foreign events."""
    text = re.sub(r'^\s*<think>\s*</think>\s*', '', text, count=1).strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {'jianzi_rows'} or not isinstance(value['jianzi_rows'], list):
        raise ValueError('expected only a jianzi_rows array')
    mapping = event_sources(item)
    found = {}
    for row in value['jianzi_rows']:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError('invalid row shape')
        index, notation = row
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(notation, str):
            raise ValueError('rows require integer event index and string notation')
        if index in found or index not in mapping:
            raise ValueError('duplicated or foreign event index')
        found[index] = notation
    if found.keys() != mapping.keys():
        raise ValueError('incomplete final artifact')
    normalized = [[index, found[index]] for index in mapping]
    plan = {'actions': [{'source_index': mapping[index], 'jianzi_text': found[index]} for index in mapping]}
    return normalized, plan


def inference_messages(item: dict, previous: dict | None, historical: dict) -> list[dict]:
    from agents.abc_to_jianzipu.teacher_trajectory import render_public_prompt
    from ABC_J.scripts.generate_teacher_tool_trajectories import blank_plan_from_item, harmonic_region_at_phrase_start
    item = deepcopy(item)
    item.pop('reference_plan', None)
    item.pop('related_phrases', None)
    item.pop('public_pitch_warning_source_indices', None)
    item.pop('public_plan_warning_messages_by_source', None)
    item['baseline_plan'] = blank_plan_from_item(item)
    handoff = item['input'].setdefault('phrase_handoff', {})
    handoff.pop('previous_phrase', None)
    if previous:
        handoff['previous_phrase'] = {'phrase_id': previous['phrase_id'], 'status': 'confirmed_readonly',
            'notes': previous['input']['notes_without_jianzi'], 'actions': previous['reference_plan']['actions']}
    item['harmonic_region_at_start'] = harmonic_region_at_phrase_start(item, historical)
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user_from_public_prompt(render_public_prompt(item, 'single_stage'))}]
