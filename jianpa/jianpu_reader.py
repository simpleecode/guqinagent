#!/usr/bin/env python3
"""简谱文本 → 旋律事件（键盘友好格式）。

这是把流行曲送进管线的最快入口：网上随手可得的简谱照着敲一遍即可，
不依赖任何站点下载权限。记谱法与 to_runtime 的输出互为镜像：

  T:像风一样            标题（可选）
  1=F 4/4              调号（必需）+ 拍号（可选，默认 4/4）
  L:1/8                单位时值（可选，默认四分音符）

  3' 3' 5 6 | 1' - 7 6      音级 1-7；' 升八度、, 降八度（可叠加）；
  #4 b3 b7 0/2 |             # / b 变化音；0 休止；
  6 2'&5 | 1' 2'/2 -2        时值 /2 /4 /8 八分十六分三十二分、
                             2 4 二分全分；. 附点；- 延音（可带倍数）；
  |                          小节线（不写则按拍号自动生成）
                             2'&5 撮（主音&副音）

八度标记只是相对的——管线随后会整体归中，所以照着简谱敲原调即可。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from to_runtime import anchor_tonic_midi

MAJOR_SCALE = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
TONIC_RE = re.compile(r"1=([A-G])([#b♯♭]?)")
METER_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
NOTE_RE = re.compile(
    r"^(?P<acc>[#b♯♭]?)(?P<deg>[1-7])(?P<oct>[',]*)(?P<dur>/\d+|\d+)?(?P<dot>\.)?$")
REST_RE = re.compile(r"^0(?P<dur>/\d+|\d+)?(?P<dot>\.)?$")


def _duration(suffix: str | None, dot: str | None, unit: float) -> float:
    if suffix is None:
        length = unit
    elif suffix.startswith("/"):
        length = unit / int(suffix[1:])
    else:
        length = unit * int(suffix)
    return length * (1.5 if dot else 1.0)


def _midi(tonic_pc: int, acc: str, deg: str, octs: str) -> int:
    semitone = MAJOR_SCALE[int(deg)]
    if acc in ("#", "♯"):
        semitone += 1
    elif acc in ("b", "♭"):
        semitone -= 1
    semitone += 12 * (octs.count("'") - octs.count(","))
    return anchor_tonic_midi(tonic_pc) + semitone


def parse_jianpu_text(text: str) -> dict:
    """Return {title, events, bars, beats_per_bar, tonic_pc, mode}.

    events = [(quarter_pos, quarter_dur, midi)]; dyads carry the partner as
    an extra zero-length event at the same position (build_readable pairs
    them back into ``jianpu_alt``).
    """
    title = ""
    tonic_pc = None
    tonic_label = ""
    numerator, denominator = 4, 4
    unit = 1.0
    body: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        if line.startswith("T:"):
            title = line[2:].strip()
            continue
        if line.startswith("L:"):
            match = re.fullmatch(r"L:\s*(\d+)\s*/\s*(\d+)", line)
            if match:
                unit = 4.0 * int(match.group(1)) / int(match.group(2))
            continue
        tonic = TONIC_RE.search(line)
        if tonic and len(line) <= 16:
            tonic_pc = (LETTERS[tonic.group(1)]
                        + (1 if tonic.group(2) in ("#", "♯") else -1 if tonic.group(2) in ("b", "♭") else 0)) % 12
            tonic_label = tonic.group(1) + ("" if not tonic.group(2) else "b" if tonic.group(2) in ("b", "♭") else "#")
            meter = METER_RE.search(line)
            if meter and not line.startswith("L:"):
                numerator, denominator = int(meter.group(1)), int(meter.group(2))
            continue
        body.append(line)
    if tonic_pc is None:
        raise ValueError("missing tonic header, e.g. '1=F 4/4'")

    beats_per_bar = 4.0 * numerator / denominator
    events: list[tuple[float, float, int]] = []
    bars: list[float] = []
    pos = 0.0
    last_attack: tuple[float, float] | None = None
    for token in " ".join(body).split():
        clean = token.strip(":|") or token
        if set(clean) <= {"|", ":", " "}:  # barline / repeat marks
            bars.append(pos)
            continue
        if clean.startswith("-"):
            if last_attack is None:
                raise ValueError(f"tie '-' before any note: {token}")
            hold = _duration(clean[1:] or None, None, unit)
            start, _ = last_attack
            for i, (s, d, m) in enumerate(events):
                if s == start and d > 0 and m >= 0:
                    events[i] = (s, d + hold, m)
            pos += hold
            continue
        if "&" in clean:
            first, second = clean.split("&", 1)
            head = NOTE_RE.match(first)
            tail = NOTE_RE.match(second)
            if not head or not tail:
                raise ValueError(f"bad dyad token: {token}")
            length = _duration(head.group("dur"), head.group("dot"), unit)
            events.append((pos, length, _midi(tonic_pc, head.group("acc"), head.group("deg"), head.group("oct"))))
            # partner carries the same length so build_readable pairs it as
            # jianpu_alt instead of treating it as a zero-length grace
            events.append((pos, length, _midi(tonic_pc, tail.group("acc"), tail.group("deg"), tail.group("oct"))))
            last_attack = (pos, length)
            pos += length
            continue
        rest = REST_RE.match(clean)
        if rest:
            pos += _duration(rest.group("dur"), rest.group("dot"), unit)
            last_attack = None
            continue
        match = NOTE_RE.match(clean)
        if not match:
            raise ValueError(f"bad token: {token}")
        length = _duration(match.group("dur"), match.group("dot"), unit)
        events.append((pos, length, _midi(tonic_pc, match.group("acc"), match.group("deg"), match.group("oct"))))
        last_attack = (pos, length)
        pos += length

    events.sort(key=lambda e: (round(e[0], 4), -e[2]))
    if not any(d > 0 for _, d, _ in events):
        raise ValueError("no notes")
    if not bars:
        total = max(s + d for s, d, _ in events if d >= 0)
        pos_b = beats_per_bar
        while pos_b < total - 1e-6:
            bars.append(pos_b)
            pos_b += beats_per_bar
    return {
        "title": title, "events": events, "bars": bars,
        "beats_per_bar": beats_per_bar, "tonic_pc": tonic_pc,
        "tonic_label": tonic_label, "mode": None,
    }
