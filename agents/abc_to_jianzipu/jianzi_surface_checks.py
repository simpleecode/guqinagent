"""Narrow surface checks, independent of pitch and musical judgement."""
import re

_DUPLICATE_STRING = re.compile(
    r'([一二三四五六七1-7])弦(?:徽外|[一二三四五六七八九十0-9]+徽)'
    r'[^弦\s＋+（）()，,；;]{0,15}?([一二三四五六七1-7])弦'
)
_NUMBERS = dict(zip('一二三四五六七', '1234567'))


def duplicate_string_prefixes(text):
    """Find same-string repetitions without crossing compound delimiters."""
    return [m.group() for m in _DUPLICATE_STRING.finditer(str(text or ''))
            if _NUMBERS.get(m[1], m[1]) == _NUMBERS.get(m[2], m[2])]
