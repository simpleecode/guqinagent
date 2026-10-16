"""Canonical public repeat label; old labels remain readable as aliases."""
import re

REPEAT_ANCHOR_LABEL = '[再作标记ㄱ]'
REPEAT_ANCHOR_NAMES = ('再作标记ㄱ', '再作标记', '再作起点ㄱ', '再作起点',
                       '再做标记ㄱ', '再做标记', '再做起点ㄱ', '再做起点')
_LABEL = re.compile(r'[〔【\[]再[作做](?:起点|标记)(?:ㄱ)?[〕】\]]')
_NAMED = re.compile(r'再[作做](?:起点|标记)ㄱ')


def normalize_repeat_anchor_text(text):
    text = str(text)
    text = text.replace('缺少再作起点标记', '缺少再作标记')
    text = text.replace('再作起点标记与再作指令不能重合', '再作标记与再作指令不能重合')
    text = _NAMED.sub('再作标记ㄱ', _LABEL.sub(REPEAT_ANCHOR_LABEL, text))
    return text.replace('再作起点', '再作标记').replace('再做起点', '再作标记').replace('再做标记', '再作标记')
