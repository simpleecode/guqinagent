from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ZH = {
    1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
    8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二", 13: "十三",
}

SLIDE_DIRECTION = {"绰": "上", "注": "下"}

# 独立成行时只书写技法字，不重复上一音已经交代的弦徽位置。
# 同一次拨弦上的“吟”仍在 attack=true 分支中作为后缀渲染。
POSITIONLESS_CONTINUATIONS = {"吟"}


def hui_label(hui: float | None) -> str:
    if hui is None:
        return ""
    whole = int(hui)
    fraction = round((hui - whole) * 10)
    label = f"{ZH.get(whole, str(whole))}徽"
    if fraction:
        label += f"{ZH.get(fraction, str(fraction))}分"
    return label


def _get(action: Any, field: str, default: Any = None) -> Any:
    if isinstance(action, Mapping):
        return action.get(field, default)
    return getattr(action, field, default)


def render_jianzi_text(
    action: Any,
    *,
    prefer_text: bool = False,
    omitted_placeholder: str | None = None,
) -> str:
    """Render one action with the repository's single减字 surface convention.

    The output is plain text without square brackets. Callers that display a
    score token add brackets themselves.  Technique column placeholders such as
    ``无`` never enter this surface.
    """
    if action is None:
        return "减字待填写"
    explicit_text = _get(action, "jianzi_text")
    if explicit_text is None:
        return "减字待填写"
    return str(explicit_text).strip()


def render_jianzi_surface(action: Any, **kwargs: Any) -> str:
    text = render_jianzi_text(action, **kwargs)
    return f"[{text}]" if text else ""
