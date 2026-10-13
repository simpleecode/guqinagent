"""Shared recognition and display of non-sounding score structure rows."""
import re

_BARS = {
    '|': ('bar', '小节线', '|'),
    '||': ('double_bar', '双小节线', '||'),
    '|||': ('multiple_bar', '多重小节线', '|]'),
    '|]': ('end_bar', '结束线', '|]'),
    '||:': ('repeat_start', '反复开始', '|:'),
    '|:': ('repeat_start', '反复开始', '|:'),
    ':||': ('repeat_end', '反复结束', ':|'),
    ':|': ('repeat_end', '反复结束', ':|'),
    '::': ('repeat_both', '反复结束／开始', '::'),
}


def structural_token_info(token):
    raw = str(token or '').strip()
    if raw in _BARS:
        kind, label, abc = _BARS[raw]
        return {'token': raw, 'kind': kind, 'label': label, 'abc': abc}
    # A closing volta bracket can precede a bar, and a new ending follows it.
    match = re.fullmatch(r'(\])?([|:]+)(?:\[([1-9][0-9,\-]*))?', raw)
    if not match or match.group(2) not in _BARS:
        return None
    closing, base, ending = match.groups()
    if not closing and not ending:
        return None
    kind, label, abc = _BARS[base]
    labels = (['反复结尾结束'] if closing else []) + [label]
    if ending:
        numbers = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五',
                   '6': '六', '7': '七', '8': '八', '9': '九', '10': '十'}
        display_ending = re.sub(r'\d+', lambda m: numbers.get(m[0], m[0]), ending)
        labels.append(f'第{display_ending}遍结尾')
        kind = 'repeat_ending'
        abc += '[' + ending
    return {'token': raw, 'kind': kind, 'label': '／'.join(labels), 'abc': abc}


def is_structural_note(note):
    return bool(note.get('structure_token') or structural_token_info(note.get('abc'))
                or str(note.get('jianpu') or '').strip() == '|'
                or note.get('duration') == '小节线')


def render_structure(note):
    info = structural_token_info(note.get('structure_token') or note.get('abc'))
    if info is None:
        return '小节线'
    if info['token'] == '|':
        return '小节线'
    if info['kind'] in {'repeat_end', 'repeat_ending'}:
        return info['token']
    return f"{info['label']}（{info['token']}）"
