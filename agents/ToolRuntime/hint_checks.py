"""Optional musical suggestions; never contribute to warning/acceptance gates."""
from __future__ import annotations
import re
from agents.abc_to_jianzipu.inverse_baseline import AUDIT
from agents.abc_to_jianzipu.jianzi_renderer import render_jianzi_surface
from agents.abc_to_jianzipu.score_structure import is_structural_note

SAME_PITCH_HINT = ('连续同音反复，可考虑使用轮／长锁／短锁等技法'
                   '（请结合时值与分组选择）')

def basic_or_blank(action: dict | None) -> bool:
    if not action:
        return True
    text = action.get('jianzi_text')
    if text is None:
        text = action.get('text') or render_jianzi_surface(action)
    text = str(text or '').strip().strip('[]')
    if not text or text in {'空', '待填写', '减字待填写', '—'}:
        return True
    # Only a single ordinary stroke with position/mode modifiers qualifies.
    # Any ornament, compound gesture, repeat instruction or unknown text
    # prevents this suggestion, instead of guessing its coverage.
    stripped = re.sub(r'泛起|泛止|按音|泛音|散音|散|大指|中指|名指|食指|跪指', '', text)
    stripped = re.sub(r'(?:[一二三四五六七八九十百0-9.]+徽(?:[一二三四五六七八九0-9]+分)?|徽外|[一二三四五六七1-7]弦)', '', stripped)
    return bool(re.fullmatch(r'[抹挑勾剔托擘打摘搓劈]', stripped))

def repeated_pitch_hints(item: dict, actions: list[dict]) -> list[dict]:
    by_index={int(a['source_index']):a for a in actions}
    notes=item.get('input',{}).get('notes_without_jianzi') or []
    try:
        tonic=AUDIT.parse_tonic_midi(item['input']['metadata'])
    except (KeyError,ValueError,TypeError):
        return []
    result=[]; run=[]; previous=None
    def finish():
        if len(run)>=5 and all(basic_or_blank(by_index.get(int(n['index']))) for n in run):
            result.append({'source_indices':[int(n['index']) for n in run],
                           'text':SAME_PITCH_HINT})
    for note in notes:
        if is_structural_note(note):
            continue
        abc=str(note.get('abc') or '')
        # Rest, tie continuation, chord and grace events do not qualify.
        scalar=bool(re.fullmatch(r'[\^_=]*[A-Ga-g][,\x27]*[0-9/]*',abc))
        pitch=AUDIT.parse_jianpu(note.get('jianpu'),tonic) if scalar else None
        if pitch is None or (run and pitch!=previous):
            finish();run=[]
        if pitch is not None:run.append(note)
        previous=pitch
    finish()
    return result
