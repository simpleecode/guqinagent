from __future__ import annotations

from collections.abc import Callable


def split_phrase_ranges(
    notes: list[dict], *, max_sounding: int,
    is_sounding: Callable[[dict], bool],
) -> list[tuple[int, int]]:
    """Split with natural boundaries first, barlines as the cut point.

    Frozen rule (2026-08-20): a section change (如 <一> → <二>) always starts a
    new phrase; once a phrase exceeds ``max_sounding`` sounding notes, it ends
    at the NEXT barline (小节线) instead of backing up, so every phrase carries
    the complete bar it overflows into.  Sections that never exceed the limit
    stay whole.  Fallback for barline-free stretches (潇湘水云 等源数据无
    小节线)：if no barline appears by ``max_sounding + 8`` sounding notes, the
    phrase is force-cut there.
    """
    if max_sounding < 1:
        raise ValueError("max_sounding must be positive")
    hard_limit = max_sounding + 8
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(notes):
        current_section = None
        sounding_count = 0
        end = len(notes)
        for index in range(start, len(notes)):
            note = notes[index]
            section = (note.get("section") or {}).get("number")
            if index > start and section is not None and section != current_section:
                end = index
                break
            if section is not None:
                current_section = section
            if is_sounding(note):
                sounding_count += 1
                if sounding_count >= hard_limit:
                    # 无小节线兜底：超限 8 音仍无线，强制截断。
                    end = index + 1
                    break
            # 小节线行本身不发音，判断必须在发音分支之外。
            if sounding_count > max_sounding and \
                    str(note.get("abc", "")).strip() == "|":
                end = index + 1
                break
        if end <= start:
            raise RuntimeError(f"phrase splitter made no progress at index {start}")
        ranges.append((start, end))
        start = end
    return ranges
