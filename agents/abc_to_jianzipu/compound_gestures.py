from __future__ import annotations

import re
from typing import Any, Mapping


# These glyphs describe a multi-sound or multi-string gesture.  They must not
# be reduced to the scalar meaning of a substring such as 撮 or 剌.
_COMPOUND_GESTURE_RE = re.compile(
    r"掐(?:拨剌|拂[歷历]|撮)[一二三]声|[歷历]拂|拂[歷历]|拨"
)


def find_compound_gesture(text: str | None) -> str | None:
    """Return the exact compound-gesture spelling embedded in ``text``."""
    match = _COMPOUND_GESTURE_RE.search(str(text or ""))
    return match.group(0) if match else None


def remove_compound_gesture(text: str | None) -> str:
    """Remove the first recognized atomic gesture, retaining any base pluck."""
    return _COMPOUND_GESTURE_RE.sub("", str(text or ""), count=1)


def action_compound_gesture(action: Any) -> str | None:
    if isinstance(action, Mapping):
        value = action.get("compound_gesture")
    else:
        value = getattr(action, "compound_gesture", None)
    value = str(value or "").strip()
    return value or None
