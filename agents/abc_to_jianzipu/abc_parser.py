from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from .models import CanonicalScore, Phrase, ScoreEvent, Section


TICKS_PER_QUARTER = 960
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER = ("B", "E", "A", "D", "G", "C", "F")
MAJOR_SIGNATURES = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "BB": -2, "EB": -3, "AB": -4, "DB": -5, "GB": -6, "CB": -7,
}
MINOR_SIGNATURES = {
    "A": 0, "E": 1, "B": 2, "F#": 3, "C#": 4, "G#": 5, "D#": 6, "A#": 7,
    "D": -1, "G": -2, "C": -3, "F": -4, "BB": -5, "EB": -6, "AB": -7,
}
SECTION_RE = re.compile(r"^%+\s*(<([^>]+)>)\s*$")
HEADER_RE = re.compile(r"^([A-Za-z]):\s*(.*)$")


class AbcParseError(ValueError):
    pass


@dataclass(slots=True)
class _ParsedToken:
    kind: str
    abc: str
    pitches: list[int]
    duration: Fraction
    tie_out: bool = False
    diagnostics: list[str] | None = None


def _fraction(text: str, default: Fraction) -> Fraction:
    if not text:
        return default
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        numerator_value = int(numerator) if numerator else 1
        denominator_value = int(denominator) if denominator else 2
        return default * Fraction(numerator_value, denominator_value)
    return default * int(text)


def _parse_meter(value: str) -> tuple[int, int]:
    value = value.strip()
    if value == "C":
        return 4, 4
    if value == "C|":
        return 2, 2
    match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
    if not match:
        raise AbcParseError(f"unsupported meter: {value}")
    return int(match.group(1)), int(match.group(2))


def _key_signature(key: str) -> dict[str, int]:
    compact = key.strip().replace("♭", "b").replace("♯", "#")
    match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", compact)
    if not match:
        raise AbcParseError(f"unsupported key: {key}")
    root = (match.group(1).upper() + match.group(2)).upper()
    mode_text = match.group(3).strip().lower()
    is_minor = mode_text.startswith(("m", "min", "aeo"))
    count = (MINOR_SIGNATURES if is_minor else MAJOR_SIGNATURES).get(root)
    if count is None:
        raise AbcParseError(f"unsupported key signature: {key}")
    signature = {letter: 0 for letter in NOTE_PC}
    order = SHARP_ORDER if count >= 0 else FLAT_ORDER
    accidental = 1 if count >= 0 else -1
    for letter in order[:abs(count)]:
        signature[letter] = accidental
    return signature


def _pitch(letter: str, accidental_text: str, octave_marks: str,
           signature: dict[str, int], bar_accidentals: dict[tuple[str, int], int]) -> int:
    upper = letter.upper()
    octave = 5 if letter.islower() else 4
    octave += octave_marks.count("'")
    octave -= octave_marks.count(",")
    key = (upper, octave)
    if accidental_text:
        if accidental_text == "=":
            accidental = 0
        elif accidental_text.startswith("^"):
            accidental = len(accidental_text)
        else:
            accidental = -len(accidental_text)
        bar_accidentals[key] = accidental
    else:
        accidental = bar_accidentals.get(key, signature[upper])
    return 12 * (octave + 1) + NOTE_PC[upper] + accidental


def _read_note(text: str, pos: int, signature: dict[str, int],
               bar_accidentals: dict[tuple[str, int], int],
               unit: Fraction) -> tuple[_ParsedToken, int]:
    start = pos
    accidental = ""
    while pos < len(text) and text[pos] in "^_":
        accidental += text[pos]
        pos += 1
    if pos < len(text) and text[pos] == "=":
        accidental = "="
        pos += 1
    if pos >= len(text) or text[pos] not in "ABCDEFGabcdefgzZxX":
        raise AbcParseError(f"expected note at offset {start}")
    letter = text[pos]
    pos += 1
    octave_marks = ""
    while pos < len(text) and text[pos] in "',":
        octave_marks += text[pos]
        pos += 1
    duration_text = ""
    while pos < len(text) and (text[pos].isdigit() or text[pos] == "/"):
        duration_text += text[pos]
        pos += 1
    tie_out = pos < len(text) and text[pos] == "-"
    if tie_out:
        pos += 1
    duration = _fraction(duration_text, unit)
    abc = text[start:pos]
    if letter in "zZxX":
        return _ParsedToken("rest", abc, [], duration, tie_out), pos
    midi = _pitch(letter, accidental, octave_marks, signature, bar_accidentals)
    return _ParsedToken("note", abc, [midi], duration, tie_out), pos


def _read_chord(text: str, pos: int, signature: dict[str, int],
                bar_accidentals: dict[tuple[str, int], int],
                unit: Fraction) -> tuple[_ParsedToken, int]:
    start = pos
    pos += 1
    pitches: list[int] = []
    while pos < len(text) and text[pos] != "]":
        if text[pos].isspace():
            pos += 1
            continue
        token, pos = _read_note(text, pos, signature, bar_accidentals, Fraction(1, 1))
        if token.pitches:
            pitches.extend(token.pitches)
    if pos >= len(text):
        raise AbcParseError(f"unclosed chord at offset {start}")
    pos += 1
    duration_text = ""
    while pos < len(text) and (text[pos].isdigit() or text[pos] == "/"):
        duration_text += text[pos]
        pos += 1
    tie_out = pos < len(text) and text[pos] == "-"
    if tie_out:
        pos += 1
    duration = _fraction(duration_text, unit)
    return _ParsedToken(
        "chord", text[start:pos], sorted(set(pitches)), duration, tie_out,
        ["chord_reduced_to_top_melody_pitch"],
    ), pos


def _tokenize_body(body_lines: list[tuple[str, str | None]], key: str,
                   unit: Fraction) -> list[tuple[_ParsedToken | str, str | None]]:
    signature = _key_signature(key)
    bar_accidentals: dict[tuple[str, int], int] = {}
    result: list[tuple[_ParsedToken | str, str | None]] = []
    for text, pending_section in body_lines:
        pos = 0
        marker_for_next = pending_section
        while pos < len(text):
            char = text[pos]
            if char.isspace():
                pos += 1
                continue
            if char == "%":
                break
            if char in "|:":
                start = pos
                while pos < len(text) and text[pos] in "|:[]123456789":
                    pos += 1
                result.append(("BAR:" + text[start:pos], marker_for_next))
                marker_for_next = None
                bar_accidentals.clear()
                continue
            if char == "-":
                # The repository's extractor writes held notes as ``C4 -C4``:
                # a leading tie marker before the continuation event.  Accept
                # this alongside standard ABC's ``C4-C4`` spelling.
                result.append(("TIE:-", marker_for_next))
                marker_for_next = None
                pos += 1
                continue
            if char == "{":
                end = text.find("}", pos + 1)
                if end < 0:
                    raise AbcParseError(f"unclosed grace group at offset {pos}")
                result.append(("GRACE:" + text[pos:end + 1], marker_for_next))
                marker_for_next = None
                pos = end + 1
                continue
            if char == "[" and pos + 2 < len(text) and text[pos + 2] == ":":
                end = text.find("]", pos + 1)
                if end < 0:
                    raise AbcParseError(f"unclosed inline field at offset {pos}")
                result.append(("FIELD:" + text[pos + 1:end], marker_for_next))
                marker_for_next = None
                pos = end + 1
                continue
            if char == "[":
                token, pos = _read_chord(text, pos, signature, bar_accidentals, unit)
            elif char in "^_=ABCDEFGabcdefgzZxX":
                token, pos = _read_note(text, pos, signature, bar_accidentals, unit)
            elif char in "()<>.!+~":
                # Decorations/slur punctuation are preserved as diagnostics in
                # the source ABC, but do not create an attack in this MVP.
                pos += 1
                continue
            else:
                raise AbcParseError(f"unsupported ABC token {char!r} at offset {pos}")
            result.append((token, marker_for_next))
            marker_for_next = None
    return result


def parse_abc(abc_text: str, *, tuning_name: str = "正调",
              open_midi: list[float] | None = None,
              phrase_bars: int = 8) -> CanonicalScore:
    """Parse an ABC tune into a deterministic event timeline.

    The parser intentionally targets the monophonic subset emitted by this
    repository while preserving chords and reporting their melody reduction.
    It does not silently expand repeats; repeat symbols are retained as
    diagnostics so callers can decide whether to reject or preprocess them.
    """
    headers: dict[str, str] = {}
    body_lines: list[tuple[str, str | None]] = []
    diagnostics: list[str] = []
    pending_section: str | None = None
    body_started = False
    for raw_line in abc_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        section_match = SECTION_RE.match(line)
        if section_match:
            pending_section = section_match.group(2).strip()
            continue
        header_match = HEADER_RE.match(line)
        if header_match and not body_started:
            field, value = header_match.groups()
            headers[field.upper()] = value
            if field.upper() == "K":
                body_started = True
            continue
        if line.startswith("%"):
            continue
        body_started = True
        body_lines.append((line, pending_section))
        pending_section = None

    key = headers.get("K", "C")
    meter = headers.get("M", "4/4")
    unit_text = headers.get("L", "1/8")
    try:
        unit = Fraction(unit_text)
    except (ValueError, ZeroDivisionError) as exc:
        raise AbcParseError(f"invalid L field: {unit_text}") from exc
    meter_num, meter_den = _parse_meter(meter)
    quarter_per_bar = Fraction(meter_num * 4, meter_den)
    tempo_match = re.search(r"(\d+)\s*$", headers.get("Q", ""))
    tempo = int(tempo_match.group(1)) if tempo_match else None
    tokens = _tokenize_body(body_lines, key, unit)

    events: list[ScoreEvent] = []
    sections: list[Section] = []
    current_section_id: str | None = None
    onset_quarters = Fraction(0)
    bar = 1
    bar_start = Fraction(0)
    pending_tie_index: int | None = None

    def ensure_section(marker_text: str | None, event_id: str) -> None:
        nonlocal current_section_id
        if marker_text is None and sections:
            return
        raw = marker_text or "一"
        pieces = raw.split(maxsplit=1)
        label = pieces[0]
        title = pieces[1] if len(pieces) > 1 else ""
        current_section_id = f"sec-{len(sections) + 1}"
        sections.append(Section(
            id=current_section_id, label=label, title=title,
            start_event=event_id, marker=f"<{raw}>",
        ))

    for item, marker in tokens:
        if isinstance(item, str):
            if item.startswith("BAR:"):
                bar_token = item[4:]
                if any(symbol in bar_token for symbol in (":", "[1", "[2")):
                    diagnostics.append(f"repeat_not_expanded:{bar_token}")
                if onset_quarters > bar_start:
                    bar += 1
                    bar_start = onset_quarters
                continue
            if item.startswith("GRACE:"):
                diagnostics.append(f"grace_preserved_not_timed:{item[6:]}")
                continue
            if item.startswith("FIELD:"):
                diagnostics.append(f"inline_field_preserved:{item[6:]}")
                continue
            if item.startswith("TIE:"):
                if events and events[-1].pitches_midi:
                    pending_tie_index = len(events) - 1
                    events[-1].abc += "-"
                else:
                    diagnostics.append("leading_tie_without_previous_pitch")
                continue
        event_id = f"n{len(events) + 1:05d}"
        ensure_section(marker, event_id)
        duration_quarters = item.duration * 4
        duration_ticks = round(float(duration_quarters) * TICKS_PER_QUARTER)
        onset_ticks = round(float(onset_quarters) * TICKS_PER_QUARTER)
        beat = float(onset_quarters - bar_start) + 1.0

        if pending_tie_index is not None and item.pitches:
            previous = events[pending_tie_index]
            if previous.pitches_midi == item.pitches:
                previous.duration_ticks += duration_ticks
                previous.abc += item.abc
                pending_tie_index = pending_tie_index if item.tie_out else None
                onset_quarters += duration_quarters
                continue
            diagnostics.append(f"tie_pitch_mismatch:{previous.id}:{event_id}")
            pending_tie_index = None

        event = ScoreEvent(
            id=event_id,
            kind=item.kind,
            abc=item.abc,
            pitches_midi=item.pitches,
            onset_ticks=onset_ticks,
            duration_ticks=duration_ticks,
            bar=bar,
            beat=beat,
            attack=item.kind != "rest",
            section_id=current_section_id,
            diagnostics=list(item.diagnostics or []),
        )
        events.append(event)
        if item.tie_out and item.pitches:
            pending_tie_index = len(events) - 1
        onset_quarters += duration_quarters

    if pending_tie_index is not None:
        diagnostics.append(f"dangling_tie:{events[pending_tie_index].id}")
    if not sections and events:
        sections.append(Section("sec-1", "一", "", events[0].id, "<一>"))
        for event in events:
            event.section_id = "sec-1"

    phrases: list[Phrase] = []
    phrase_events: list[ScoreEvent] = []
    current_section: str | None = None

    def flush_phrase() -> None:
        if not phrase_events:
            return
        phrase_id = f"p{len(phrases) + 1:04d}"
        for phrase_event in phrase_events:
            phrase_event.phrase_id = phrase_id
        phrases.append(Phrase(
            id=phrase_id,
            section_id=phrase_events[0].section_id,
            event_ids=[event.id for event in phrase_events],
            start_bar=phrase_events[0].bar,
            end_bar=phrase_events[-1].bar,
        ))
        phrase_events.clear()

    for event in events:
        if current_section is not None and event.section_id != current_section:
            flush_phrase()
        if phrase_events and event.bar - phrase_events[0].bar >= phrase_bars:
            flush_phrase()
        phrase_events.append(event)
        current_section = event.section_id
        if event.kind == "rest" and event.duration_ticks >= TICKS_PER_QUARTER:
            flush_phrase()
    flush_phrase()

    return CanonicalScore(
        title=headers.get("T", "Untitled"),
        key=key,
        meter=meter,
        unit_note_length=unit_text,
        tempo=tempo,
        tuning_name=tuning_name,
        open_midi=list(open_midi or [48, 50, 53, 55, 57, 60, 62]),
        sections=sections,
        phrases=phrases,
        events=events,
        source_abc=abc_text,
        diagnostics=diagnostics,
    )
