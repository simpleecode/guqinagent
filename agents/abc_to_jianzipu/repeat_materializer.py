"""Mark already-present 再作 copies in the sounding note stream.

The source score contains the complete pitch/timing stream, including the
notes that are repeated.  What is omitted in the printed tablature is the
*jianzi text* of the copied passage.  Therefore materialisation must not
insert synthetic notes: it records an anchor (``再作`` or
``〔再作起点〕``), then marks the existing notes beginning at
``从ㄱ再作`` as ``notation_omitted=True``.  This distinction is important:
inserting copies shifts phrase indices and makes the agent see a different
piece from the annotated score.

``从ㄱ再作`` starts a copy from the ``再作起点`` anchor.  A plain ``再作``
starts a copy of the immediately preceding material.  In both cases the
source stream already contains the copied rows; the omitted region is the
contiguous run of rows with empty ``jianzi`` up to the next non-empty jianzi.
This deliberately avoids guessing a length from timing or pitch.  Unpaired
markers are retained and reported, but their visible empty tail is still
marked because the score explicitly indicates a repeat there.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Callable

MARKER_PRIORITY = ("从头再作", "从ㄱ再作", "再作三声", "再作二声", "再作起点", "再作")
REPEAT_COPY_COUNT = {"再作": 1, "从ㄱ再作": 1, "从头再作": 1, "再作二声": 2, "再作三声": 3}
OMITTED_PLACEHOLDER = "无（由于是再作部分，省略）"


def extract_repeat_markers(jianzi) -> list[str]:
    """Ordered repeat markers embedded in a jianzi string, longest first."""
    text = str(jianzi or "")
    hits: list[tuple[int, int, str]] = []
    for marker in MARKER_PRIORITY:
        for match in re.finditer(re.escape(marker), text):
            hits.append((match.start(), -len(marker), marker))
    hits.sort()
    chosen: list[tuple[int, str]] = []
    result = []
    for position, _, marker in hits:
        if any(start <= position < start + len(name) for start, name in chosen):
            continue
        chosen.append((position, marker))
        result.append(marker)
    return result


def is_marker_only(jianzi) -> bool:
    """True when the jianzi contains nothing but repeat markers/separators."""
    text = str(jianzi or "")
    for marker in MARKER_PRIORITY:
        text = text.replace(marker, "")
    return text.replace("。", "").strip() == "" and bool(str(jianzi or "").strip())


def materialize_repeats(notes: list[dict], *,
                        is_sounding: Callable[[dict], bool]) -> tuple[list[dict], dict]:
    """Expand repeat instructions; returns (new_notes, report)."""
    # Keep the original stream and indices.  Repeat materialisation annotates
    # existing rows; it never appends synthetic clones.
    output: list[dict] = [deepcopy(note) for note in notes]
    marker_counts: Counter = Counter()
    span_lengths: list[int] = []
    marker_only_notes = 0
    anchors: list[dict] = []
    copy_markers: list[dict] = []
    unpaired_markers: list[dict] = []

    def mark_empty_tail(copy_start: int, marker: str,
                        anchor_index: int | None, *, include_marker: bool) -> int:
        """Flag an indicated repeat and its following empty-jianzi rows."""
        first = copy_start if include_marker else copy_start + 1
        marked = 0
        if include_marker:
            row = output[copy_start]
            row["notation_omitted"] = True
            row["materialized"] = {
                "marker": marker,
                "source_index": notes[copy_start].get("index"),
                "anchor_index": anchor_index,
            }
            marked = 1
            first = copy_start + 1
        for target in range(first, len(output)):
            text = str(notes[target].get("jianzi") or "").strip()
            if text:
                break
            row = output[target]
            row["notation_omitted"] = True
            row["materialized"] = {
                "marker": marker,
                "source_index": notes[target].get("index"),
                "anchor_index": anchor_index,
            }
            marked += 1
        if marked:
            span_lengths.append(marked)
        return marked

    # ``〔再作起点〕`` identifies ㄱ.  A plain ``再作`` is itself a copy
    # marker; ``从ㄱ再作`` refers back to the most recent explicit anchor.
    pending_anchor: tuple[int, str, int] | None = None
    for position, note in enumerate(notes):
        markers = extract_repeat_markers(note.get("jianzi"))
        for marker in markers:
            marker_counts[marker] += 1
        only_markers = is_marker_only(note.get("jianzi"))
        if only_markers:
            marker_only_notes += 1
        if "再作起点" in markers:
            pending_anchor = (position, markers[0], int(note.get("index", position)))
            anchors.append({"position": position, "index": note.get("index"),
                            "marker": markers[0]})

        for marker in markers:
            if marker == "再作起点":
                continue
            if marker not in REPEAT_COPY_COUNT:
                continue
            anchor_index = pending_anchor[2] if pending_anchor is not None else None
            if marker == "从ㄱ再作" and pending_anchor is None:
                unpaired_markers.append({"position": position,
                                         "index": note.get("index"),
                                         "marker": marker})
                continue
            include_marker = marker != "再作"
            copy_markers.append({"position": position, "index": note.get("index"),
                                 "marker": marker, "anchor_index": anchor_index})
            mark_empty_tail(position, marker, anchor_index,
                            include_marker=include_marker)

    report = {
        "markers": dict(marker_counts),
        "marker_only_notes": marker_only_notes,
        "anchors": anchors,
        "copy_markers": copy_markers,
        "unpaired_markers": unpaired_markers,
        "spans_materialized": len(span_lengths),
        "copies_added": 0,
        "span_length_min_median_max": (
            [min(span_lengths), sorted(span_lengths)[len(span_lengths) // 2],
             max(span_lengths)] if span_lengths else []),
        "notes_before": len(notes),
        "notes_after": len(output),
    }
    return output, report
