#!/usr/bin/env python3
"""Audit numbered-notation pitches against readable guqin jianzipu JSON."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.abc_to_jianzipu.compound_gestures import find_compound_gesture  # noqa: E402
from agents.abc_to_jianzipu.score_structure import is_structural_note

MAPPER_PATH = (
    ROOT / "skills" / "guqin-pitch-mapper" / "scripts"
    / "guqin_pitch_mapper.py"
)
_SPEC = importlib.util.spec_from_file_location("guqin_pitch_mapper", MAPPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load pitch mapper: {MAPPER_PATH}")
MAPPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MAPPER)

ZH_NUMBERS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13,
}
NOTE_PC = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
    "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11,
}
MAJOR = (0, 2, 4, 5, 7, 9, 11)

# Degree-1 ("do") absolute MIDI for each numbered-notation tonic, i.e. the
# pitch an unmarked degree 1 maps to. Values are empirically derived from the
# captured corpus and cross-checked against the App's own algorithm
# (docs/PITCH_ALGORITHM_REVERSE_ENGINEERED.md). Key points:
#   * The App reuses one physical tuning (zhengdiao C D F G A C D) for every
#     tonic; "1=X" is a movable-do label, not a re-tuning.
#   * Sitongli's "1=B" actually means B-flat (A#): the do is reached at the
#     1st string pressed at hui 7.6, which is Bb3 (MIDI 58). Writing "1=B"
#     for a B-flat do is the App's own convention, confirmed both by the
#     reverse-engineered doc ("1=Bb") and by the median degree-1 pitch across
#     all captured 1=B scores.
#   * 1=F and 1=C anchor degree 1 at F4 / C4 (MIDI 65 / 60), matching the
#     old 60+pc behaviour for those two tonics.
# Only the three tonics that appear in the captured corpus are pinned. Any
# other tonic falls back to the letter-based heuristic below.
TONIC_DEGREE1_MIDI = {
    "F": 65,   # F4
    "C": 60,   # C4
    "B": 58,   # Bb3 -- App's legacy "1=B" label is a B-flat do, not B natural
}
NOTE_NAMES = ("C", "C♯", "D", "E♭", "E", "F",
              "F♯", "G", "A♭", "A", "B♭", "B")
RIGHT_HAND_TECHNIQUES = (
    "挑", "勾", "抹", "剔", "托", "擘", "打", "摘", "撮", "泼", "剌",
    "历", "歷", "厉", "轮", "輪",
)
LEFT_HAND_FINGERS = ("大指", "名指", "中指", "食指", "跪指")
ORNAMENTS = {"吟", "猱", "撞", "逗", "往来", "复", "掐起", "带起", "抓起", "爪起", "滔起"}

# ``绰上`` / ``注下`` are connected left-hand movements, not arbitrary
# repositioning instructions.  Spans beyond 500 cents should be reviewed;
# 498-cent whole-step-like moves are permitted.
WALK_SPAN_WARNING_CENTS = 501.0
# A single left-hand finger making a large simultaneous change in both string
# and hui is mechanically demanding.  This is an advisory rather than an
# impossibility rule: expert fingering can sometimes justify it, but the
# notation agent should consciously consider a finger combination instead of
# carrying the same finger by default.
SAME_FINGER_REACH_WARNING_DISTANCE = 4.0

# Low numbered hui are physically cramped on the treble/middle strings.  This
# is an ergonomic advisory, not an invalidation rule: an experienced player
# can occasionally use one, but the notation agent should choose it
# deliberately.  Hui numbering grows toward the bridge, hence "fourth hui or
# above" in the playing sense below means a numeric position <= 4.
LOW_HUI_STOPPED_LIMITS = {
    # Strings 1--4: fourth hui, its fractions, and all lower-numbered hui.
    "strings_1_to_4_max_hui": 4.0,
    # String 5: only positions strictly lower than fourth hui; fourth hui is
    # intentionally allowed, matching the requested boundary.
    "string_5_max_exclusive_hui": 4.0,
}


def is_pitch_ambiguous_sweep(text: str) -> bool:
    """Whether a swept ``拂`` glyph has no single pitch to audit.

    A notation such as ``泛音名指五徽拂一至六弦`` is one compound sweep,
    not an assertion that a single numbered-note event sounds at the endpoint
    string.  Comparing one jianpu target to the parser's representative
    member of that sweep produces artificial octave warnings.
    """
    return "拂" in str(text or "")


def is_continuity_exempt_compound(text: str) -> bool:
    """Whether an articulated gesture should not carry a one-note sustain.

    Sweeps and locks deliberately articulate or damp a group of strings.
    They are therefore unsuitable predecessors for the narrow advisory that
    says the *next* same-string note stops a residual open/harmonic sound.
    """
    return any(token in str(text or "") for token in ("拂", "锁", "鎖"))


def has_explicit_resonance_stop(text: str) -> bool:
    """A written stop or 少息 ends the next-note resonance advisory."""
    return any(marker in str(text or "") for marker in ("停", "少息"))


def parse_string_numbers(text: str) -> list[int]:
    """Parse ordinary and compact string-number spellings.

    Besides ``四弦七弦``, corpus shorthand such as ``四七弦`` names both
    strings, with only the final 弦 shared.  Parse that compact sequence as
    two separate strings rather than silently dropping the first one.
    """
    # “如一” is a named right-hand technique meaning two strings sound as
    # one, not the string sequence “一”.  In forms such as
    # “剔五弦散如一四弦”, parsing the 一 in 如一 as part of “一四弦” invents
    # a spurious first-string note and causes a false pitch mismatch.
    text = text.replace("如一", "")
    matches = re.finditer(
        r"(?P<compact>[一二三四五六七1-7]{2,})弦"
        r"|(?P<number>十(?:一|二|三)?|[一二三四五六七1-7])弦",
        text,
    )
    values = []
    for match in matches:
        tokens = list(match.group("compact")) if match.group("compact") else [match.group("number")]
        for token in tokens:
            value = ZH_NUMBERS.get(token)
            if value is None:
                value = int(token)
            if 1 <= value <= 7:
                values.append(value)
    return values


def invalid_string_numbers(text: str) -> list[int]:
    """Detect out-of-range explicit strings without mistaking hui for strings.

    Preserve compact valid spellings such as 四七弦 / 47弦.
    """
    invalid = set()
    for token in re.findall(r'([零〇一二三四五六七八九十百\d]+)\s*弦', str(text)):
        if token.isdigit():
            values = [int(c) for c in token] if all(c in '1234567' for c in token) else [int(token)]
        elif '百' in token:
            invalid.add(100)
            continue
        elif '十' in token:
            left, right = token.split('十', 1)
            values = [(ZH_NUMBERS.get(left, 1) if left else 1) * 10 + ZH_NUMBERS.get(right, 0)]
        else:
            values = [int(c) if c.isdigit() else ZH_NUMBERS.get(c, 0) for c in token]
        invalid.update(value for value in values if not 1 <= value <= 7)
    return sorted(invalid)


def invalid_string_warnings(data: dict) -> list[dict]:
    return [
        {'index': note.get('index'), 'code': 'invalid_string_number', 'strings': numbers}
        for note in data.get('notes', [])
        if (numbers := invalid_string_numbers(note.get('jianzi') or ''))
    ]


def string_number(token: str) -> int:
    value = ZH_NUMBERS.get(token)
    return int(value) if value is not None else int(token)


def parse_ruyi_open_string(text: str) -> int | None:
    """Return the string explicitly opened by the ``散如一N弦`` suffix."""
    match = re.search(r"散如一\s*([一二三四五六七1-7])弦", text)
    return string_number(match.group(1)) if match else None


def parse_li_string_sequence(text: str) -> list[int]:
    """Return the ordered strings sounded by 历/歷/厉.

    历 is a sequential index-finger sweep, not a simultaneous chord.  Corpus
    surfaces occur both compactly (历四三) and with separators/弦 (历四、三弦).
    """
    span = re.search(r'[历歷厉]([一二三四五六七1-7])弦?至([一二三四五六七1-7])弦',text)
    if span:
        first,last = map(string_number,span.groups())
        step = 1 if last >= first else -1
        return list(range(first,last+step,step))
    match = re.search(r"[历歷厉]([一二三四五六七1-7](?:[、，,＋+]?\s*[一二三四五六七1-7])+)(?:弦)?", text)
    if not match:
        return []
    return [string_number(token) for token in re.findall(
        r"[一二三四五六七1-7]", match.group(1)
    )]


def new_context() -> dict:
    """Return the pitch-relevant subset of the inheritance state."""
    return {
        "sound_mode": None,
        # A bare rest ends the sounding state, but an abbreviated right-hand
        # attack after it must not silently reuse a held position if the last
        # sounded attack before the rest was open.
        "abbreviated_open_after_rest": False,
        "harmonic": False,
        # Persistent span state.  Unlike ``sound_mode``, an explicit open
        # string inside 泛起…泛止 is a per-note override and must not end the
        # harmonic region for later notes.
        "harmonic_scope": False,
        "harmonic_hui": None,
        "stopped_string": None,
        "stopped_hui": None,
        # Kept separately from ``sound_mode``.  A score can contain a
        # non-sounding slide/control sign, or a compound pluck, while the
        # left hand is still at a usable stopped position.
        "active_left_hui": None,
        "active_left_string": None,
        # A standalone 复 returns to the position before the preceding walk.
        "walk_return_position": None,
        "left_finger": None,
        # Per-finger stopped positions are retained separately from the
        # immediately active position.  They are needed for releases such as
        # 带起同声, where a left-hand finger releases while the right hand
        # sounds a different string.
        "left_positions": {},
        "right_hand": None,
    }


def normalize_context(context: dict | None) -> dict:
    """Accept old caller-provided contexts while adding new state fields."""
    if context is None:
        return new_context()
    for key, value in new_context().items():
        context.setdefault(key, value)
    if context["harmonic"]:
        context["sound_mode"] = "harmonic"
    context["harmonic_scope"] = bool(
        context.get("harmonic_scope") or context["harmonic"]
    )
    if not isinstance(context.get("left_positions"), dict):
        context["left_positions"] = {}
    return context


def is_bare_rest_note(note: dict, text: str | None = None) -> bool:
    """Whether this row is a rest with no notation that continues a state."""
    surface = str(note.get("jianzi") or "").strip() if text is None else text.strip()
    return not is_structural_note(note) and "休止" in str(note.get("jianpu") or "") and not surface


def advance_context_for_bare_rest(context: dict, note: dict, text: str | None = None) -> bool:
    """Apply the canonical state transition for an unnotated rest.

    A bare rest ends the sounding/harmonic state while preserving a held
    stopped position for explicit walks. If the last sounded attack was open,
    ordinary abbreviated plucks after the rest remain open. All stateful audit
    passes and runtime phrase-boundary replay use this helper.
    """
    if not is_bare_rest_note(note, text):
        return False
    previous = dict(context)
    context.clear()
    context.update(new_context())
    context["abbreviated_open_after_rest"] = (
        previous.get("sound_mode") == "open"
        or (previous.get("sound_mode") is None
            and bool(previous.get("abbreviated_open_after_rest")))
    )
    context["active_left_hui"] = previous.get("active_left_hui")
    context["active_left_string"] = previous.get("active_left_string")
    context["left_finger"] = previous.get("left_finger")
    context["left_positions"] = dict(previous.get("left_positions") or {})
    return True


def replay_harmonic_scope(active: bool, note: dict, text: str | None = None) -> bool:
    """Advance phrase-boundary harmonic scope with the canonical rest rule."""
    surface = str(note.get("jianzi") or "") if text is None else text
    for marker in re.findall(r"泛起|泛止", surface):
        active = marker == "泛起"
    return False if is_bare_rest_note(note, surface) else active


def first_token(text: str, choices: tuple[str, ...]) -> str | None:
    return next((token for token in choices if token in text), None)


def remember_left_position(
    context: dict, string: int, hui: float | str | None,
    finger: str | None = None,
) -> None:
    """Record a concrete stopped position for later left-hand releases."""
    if hui is None:
        return
    resolved_finger = finger or context.get("left_finger")
    if resolved_finger in LEFT_HAND_FINGERS:
        # Dict order is the fallback recency order, not first-ever use order.
        context['left_positions'].pop(str(resolved_finger), None)
        context["left_positions"][str(resolved_finger)] = {
            "string": int(string), "hui": hui,
        }


def invalidate_active_left_hui(context: dict) -> None:
    """Forget an unknown movement endpoint, including per-finger caches."""
    active_string = context.get("active_left_string")
    context["active_left_hui"] = None
    context["stopped_hui"] = None
    context["left_positions"] = {
        finger: position for finger, position in context["left_positions"].items()
        if position.get("string") != active_string
    }


def release_position_for_daiqi(context: dict, text: str) -> tuple[str, int] | None:
    """Resolve the previously stopped string released by 带起.

    An explicitly named left finger wins.  Otherwise the current stopped
    position is the musically least speculative reading; if it has already
    been superseded, use the most recently recorded finger position.
    """
    requested = first_token(text, LEFT_HAND_FINGERS)
    positions = context["left_positions"]
    if requested:
        return (requested, int(positions[requested]['string'])) if requested in positions else None
    active_finger = context.get("left_finger")
    active_string = context.get("active_left_string")
    if active_finger in LEFT_HAND_FINGERS and active_string is not None:
        return str(active_finger), int(active_string)
    if active_finger in positions:
        # An intervening open-string pluck clears melodic inheritance, but
        # need not release a finger still pressing a DIFFERENT string.
        return str(active_finger), int(positions[active_finger]['string'])
    if positions:
        finger = next(reversed(positions))
        return str(finger), int(positions[finger]["string"])
    return None


def midi_name(value: float) -> str:
    nearest = round(value)
    return f"{NOTE_NAMES[nearest % 12]}{nearest // 12 - 1}"


def parse_tonic_midi(metadata: dict) -> float:
    # A few legacy App scores use a non-standard register for their tonic.
    # Keep that capture-specific fact in metadata instead of redefining the
    # ordinary spelling ``1=bB`` for every score.
    captured = metadata.get("tonic_degree1_midi")
    if isinstance(captured, (int, float)):
        return float(captured)
    tonic_text = str(metadata.get("tonic", "1=C")).split("=")[-1].strip()
    if re.fullmatch(r"b[A-Ga-g]", tonic_text):
        tonic_text = tonic_text[1].upper() + "b"
    key = tonic_text.upper().replace("♯", "#").replace("♭", "B")
    if key not in NOTE_PC:
        raise ValueError(f"unrecognized tonic: {tonic_text}")
    # Degree 1 ("do") anchor for this tonic. Prefer the empirically pinned
    # table (see TONIC_DEGREE1_MIDI); fall back to a letter-based octave-4
    # heuristic for tonics not yet measured in the corpus. The fallback is
    # approximate because Sitongli's "1=B" convention (B-flat do at octave 3,
    # not B natural at octave 4) shows the letter alone is unreliable.
    if key in TONIC_DEGREE1_MIDI:
        return TONIC_DEGREE1_MIDI[key]
    return 60 + NOTE_PC[key]


def parse_open_midi(metadata: dict) -> list[float]:
    rows = metadata.get("tuning", {}).get("open_strings") or []
    # Standalone captures produced by extract_jianpu_jianzi preserve the
    # compact runtime tuning under `tuning` and put its audited expansion in
    # `tuning_interpretation`.  Both describe the same captured pitches.
    if not rows:
        rows = metadata.get("tuning_interpretation", {}).get("open_strings") or []
    if len(rows) != 7:
        raise ValueError("metadata.tuning.open_strings must contain 7 strings")
    result = []
    for expected, row in enumerate(rows, 1):
        if int(row.get("string", expected)) != expected:
            raise ValueError("open_strings must be ordered from string 1 to 7")
        pitch = str(row["pitch"]).upper().replace("♯", "#").replace("♭", "B")
        if pitch not in NOTE_PC:
            raise ValueError(f"unrecognized open-string pitch: {pitch}")
        octave = int(row["octave"])
        # ``open_strings.pitch`` / ``octave`` are the *sounding* open-string
        # pitch emitted by ``interpret_tuning``.  ``semitone_offset`` remains
        # alongside them solely as provenance for the named tuning; adding it
        # here would apply every non-zero tuning adjustment twice (for example
        # 紧五弦 A3 -> B♭3 would incorrectly become B3).
        result.append(12 * (octave + 1) + NOTE_PC[pitch])
    return result


def parse_jianpu(text: str | None, tonic_midi: float) -> float | None:
    if not text:
        return None
    if "休止" in text or "延音" in text:
        return None
    match = re.search(r"([1-7])", text)
    if not match:
        return None
    degree = int(match.group(1))
    octave = text.count("\u0307") - text.count("\u0323")
    accidental = text.count("♯") + text.count("#")
    accidental -= text.count("♭") + text.count("b")
    return tonic_midi + MAJOR[degree - 1] + 12 * octave + accidental


OUTSIDE_HUI_SEMITONE_DROPS = {"徽外": 1.0, "徽外半": 2.0}
OUTSIDE_HUI_ANCHOR = 12.3


def parse_hui(text: str) -> float | str | None:
    # Check the longer spelling first: 徽外 is a substring of 徽外半.
    if "徽外半" in text:
        return "徽外半"
    if "徽外" in text:
        return "徽外"
    match = re.search(
        r"(十三|十二|十一|十|九|八|七|六|五|四|三|二|一)徽"
        r"(?:(一|二|三|四|五|六|七|八|九)分)?",
        text,
    )
    if not match:
        return None
    hui = ZH_NUMBERS[match.group(1)]
    if match.group(2):
        hui += ZH_NUMBERS[match.group(2)] / 10
    return float(hui)


def position_pitch(
    string: int, hui: float | str | None, open_midi: list[float],
    mode: str = "stopped",
) -> tuple[float | None, str | None]:
    if not 1 <= string <= 7:
        return None, "string_out_of_range"
    if hui is None:
        return open_midi[string - 1], None
    if isinstance(hui, str):
        if mode == "harmonic":
            return None, "hui_outside_not_valid_for_harmonic"
        semitone_drop = OUTSIDE_HUI_SEMITONE_DROPS.get(hui)
        if semitone_drop is None:
            return None, "unknown_hui_label"
        coordinate = MAPPER.hui_coordinate(OUTSIDE_HUI_ANCHOR)
        value = (open_midi[string - 1] + 12 * math.log2(1 / coordinate)
                 - semitone_drop)
        return value, None
    try:
        coordinate = MAPPER.hui_coordinate(hui)
    except ValueError as exc:
        return None, str(exc)
    if mode == "harmonic":
        if hui != int(hui):
            return None, "fractional_hui_not_valid_for_harmonic"
        return open_midi[string - 1] + MAPPER.HARMONIC_SEMITONES[int(hui) - 1], None
    return open_midi[string - 1] + 12 * math.log2(1 / coordinate), None


def _parse_jianzi(
    text: str, open_midi: list[float], context: dict | None = None,
) -> tuple[list[float], str | None]:
    context = normalize_context(context)
    if not text:
        return [], "empty_jianzi"

    # Multi-sound gestures are meaningful, but a scalar pitch check cannot
    # represent them.  Stop before substring parsing mistakes 掐撮/掐拨剌 for
    # an ordinary 撮/剌 double stop.
    if find_compound_gesture(text):
        return [], "context_dependent_compound_gesture"

    # 撮、泼、剌也可以由两个按音构成。每个括号组件都必须独立带有
    # 弦位和按音标识；不能借用普通“按音＋散音”的简写分支，否则会把
    # 两根按弦误判成“多个弦且徽位不明”。左手可达性另由
    # audit_cuo_fingering_feasibility.py 判断，此处只还原两个声音。
    double_stopped = re.search(
        r"(?:小|大|反)?[撮泼剌]\s*[（(]([^）)]*)[）)]", text
    )
    if double_stopped:
        # Some teacher outputs use the enumeration comma between the two
        # voices. Accept it for pitch/state replay without rewriting the
        # original text or treating a readable variant as an audit warning.
        components = [part.strip() for part in re.split(
            r"[＋+、]", double_stopped.group(1)
        ) if part.strip()]
        # Two explicitly harmonic components are independent pitches, not
        # an ambiguous scalar glyph. Do not treat their touch points as
        # stopped left-hand positions or implicitly open a 泛起 region.
        if len(components) == 2 and all('泛音' in c for c in components):
            harmonic_parts = []
            for component in components:
                strings = parse_string_numbers(component)
                hui = parse_hui(component)
                if len(strings) != 1 or hui is None:
                    return [], 'double_harmonic_position_missing'
                if '按音' in component or '散音' in component:
                    return [], 'double_harmonic_mode_conflict'
                pitch, error = position_pitch(strings[0], hui, open_midi, 'harmonic')
                if error or pitch is None:
                    return [], error or 'double_harmonic_pitch_unavailable'
                harmonic_parts.append((hui, pitch))
                context['current_note_attacks'].append({
                    'string': strings[0], 'hui': hui, 'mode': 'harmonic',
                })
            huis = {hui for hui, _pitch in harmonic_parts}
            context['sound_mode'] = 'harmonic'
            context['harmonic_hui'] = next(iter(huis)) if len(huis) == 1 else None
            context['stopped_string'] = None
            context['stopped_hui'] = None
            context['active_left_string'] = None
            context['active_left_hui'] = None
            context['left_finger'] = None
            return [pitch for _hui, pitch in harmonic_parts], None
        parsed_components: list[tuple[int, float | str, float, str | None]] = []
        for component in components:
            if "按音" not in component:
                parsed_components = []
                break
            component_strings = parse_string_numbers(component)
            component_hui = parse_hui(component)
            if len(component_strings) != 1 or component_hui is None:
                parsed_components = []
                break
            pitch, error = position_pitch(
                component_strings[0], component_hui, open_midi, "stopped"
            )
            if error or pitch is None:
                return [], error or "double_stopped_pitch_unavailable"
            parsed_components.append((
                component_strings[0], component_hui, pitch,
                first_token(component, LEFT_HAND_FINGERS),
            ))
        if len(parsed_components) == 2:
            # There is no single unambiguous active left-hand location after a
            # double stop. Retaining either component here would make a later
            # abbreviated glyph inherit a position selected arbitrarily.
            context["sound_mode"] = "stopped"
            context["stopped_string"] = None
            context["stopped_hui"] = None
            context["active_left_string"] = None
            context["active_left_hui"] = None
            context["left_finger"] = None
            for string, hui, _pitch, finger in parsed_components:
                remember_left_position(context, string, hui, finger)
            # In a 泛起…泛止 span, a two-string compound can explicitly move
            # the harmonic touch point.  It has no single stopped-string
            # position to retain, but subsequent shorthand such as “挑2弦”
            # must inherit its shared hui.  Previously this early-return
            # branch retained the preceding harmonic_hui (e.g. 五徽 after a
            # new 四徽撮), producing a spurious 500-cent mismatch.
            component_huis = {hui for _string, hui, _pitch, _finger in parsed_components}
            if context.get("harmonic_scope") and len(component_huis) == 1:
                context["harmonic_hui"] = next(iter(component_huis))
            component_fingers = {
                finger for _string, _hui, _pitch, finger in parsed_components
                if finger
            }
            # A same-finger compound has no unique active string, but it does
            # establish a usable common hui for a following shorthand such as
            # “勾2弦”.  Retaining that shared position avoids a false octave
            # mismatch while still leaving active_left_string unresolved.
            if len(component_huis) == 1 and len(component_fingers) == 1:
                context["active_left_hui"] = next(iter(component_huis))
                context["left_finger"] = next(iter(component_fingers))
            return [part[2] for part in parsed_components], None

    # 爪起（亦写抓起）不是保持按位再次发声，而是承接前一个由大指
    # 按弦发出的按音：大指甲尖拨起并放开同一弦，使它转为散音。
    # 它可与当前另一根弦的右手取声同时出现，例如“散挑七弦爪起”。
    if "爪起" in text or "抓起" in text:
        released_string = context.get("active_left_string")
        if released_string is None or context.get("left_finger") != "大指":
            return [], "zhuaqi_requires_previous_thumb_stopped_note"
        ordinary = text.replace("爪起", "").replace("抓起", "").strip()
        pitches: list[float] = []
        if ordinary:
            ordinary_context = dict(context)
            ordinary_pitches, ordinary_reason = parse_jianzi(
                ordinary, open_midi, ordinary_context
            )
            if ordinary_reason is not None:
                return [], ordinary_reason
            pitches.extend(ordinary_pitches)
            context["right_hand"] = ordinary_context.get("right_hand")
        released_pitch = open_midi[int(released_string) - 1]
        if released_pitch not in pitches:
            pitches.append(released_pitch)
        context["sound_mode"] = "open"
        context["stopped_string"] = None
        context["stopped_hui"] = None
        context["active_left_string"] = None
        context["active_left_hui"] = None
        context["left_finger"] = None
        context["left_positions"].pop("大指", None)
        return pitches, None

    # 带起 releases a previously stopped string by its established left-hand
    # finger.  With 同声/同起, that open-string release sounds together with
    # the ordinary right-hand pluck in this glyph.  The releasing finger is
    # not restricted to 名指: an explicitly stated 大指、中指等 takes
    # precedence, otherwise the current left-hand position is inherited.
    if "带起" in text:
        release = release_position_for_daiqi(context, text)
        if release is None:
            return [], "daiqi_requires_previous_stopped_note"
        released_finger, released_string = release
        ordinary = (text.replace("带起", "").replace("同声", "")
                    .replace("同起", "").strip())
        pitches: list[float] = []
        if ordinary:
            ordinary_context = dict(context)
            ordinary_context["left_positions"] = dict(context["left_positions"])
            ordinary_pitches, ordinary_reason = parse_jianzi(
                ordinary, open_midi, ordinary_context
            )
            if ordinary_reason is not None:
                return [], ordinary_reason
            pitches.extend(ordinary_pitches)
            context["right_hand"] = ordinary_context.get("right_hand")
        released_pitch = open_midi[released_string - 1]
        if released_pitch not in pitches:
            pitches.append(released_pitch)
        context["left_positions"].pop(released_finger, None)
        if context.get("left_finger") == released_finger:
            context["sound_mode"] = "open"
            context["stopped_string"] = None
            context["stopped_hui"] = None
            context["active_left_string"] = None
            context["active_left_hui"] = None
            context["left_finger"] = None
        return pitches, None

    ends_harmonic = "泛止" in text
    # The placement of 泛止 is meaningful.  A prefix closes the harmonic
    # region before this note (泛止名指九徽勾五弦); a suffix closes it after
    # this note (名指九徽勾五弦泛止).  A standalone 泛止 remains a control
    # marker handled below.
    compact_text = text.strip()
    ends_harmonic_before_note = (
        compact_text.startswith("泛止") and compact_text != "泛止"
    )
    ends_harmonic_after_note = ends_harmonic and not ends_harmonic_before_note
    if ends_harmonic_before_note:
        context["harmonic"] = False
        context["harmonic_scope"] = False
        context["harmonic_hui"] = None
        context["sound_mode"] = None

    # 历 is an ordered sweep, not a simultaneous chord. It still needs the
    # normal pitch-state resolution: an explicit left-hand position, or a
    # surrounding 泛起…泛止 state, applies to each declared string. The former
    # early-return branch incorrectly forced every 历 into open-string mode.
    li_strings = parse_li_string_sequence(text)
    strings = li_strings or parse_string_numbers(text)
    ruyi_open_string = parse_ruyi_open_string(text)
    # Prefix labels such as ``〔再作起点〕`` may appear before 泛起.  The
    # state transition is attached to the marker, not to byte zero.
    starts_harmonic = (
        "泛起" in text
        and ("泛止" not in text or text.find("泛起") < text.find("泛止"))
    )
    # A control glyph may carry a leading qualifier, e.g. ``停泛止``.  It
    # has no string/pitch of its own, but it still closes the persistent
    # 泛起…泛止 span.  Previously only the exact bare glyph ``泛止`` reached
    # the cleanup below; ``停泛止`` returned as an unparseable no-string row
    # and left the auditor in harmonic mode.  That disagreed with the
    # prompt-facing marker replay, which correctly sees the embedded 泛止.
    if ends_harmonic and not strings:
        context["harmonic"] = False
        context["harmonic_scope"] = False
        context["harmonic_hui"] = None
        context["sound_mode"] = None
        return [], "ornament_or_control"
    # 泛起 may be a standalone control glyph.  It still opens a persistent
    # harmonic span for the notes that follow, even though this glyph itself
    # names no string and therefore produces no scalar pitch.
    if starts_harmonic and not strings:
        context["harmonic"] = True
        context["harmonic_scope"] = True
        context["sound_mode"] = "harmonic"
        hui = parse_hui(text)
        if hui is not None:
            context["harmonic_hui"] = hui
        context["active_left_string"] = None
        context["active_left_hui"] = None
        return [], "ornament_or_control"
    # ``至X弦`` is a transition/control glyph, not an instruction to pluck
    # X as an open string.  Treating it as a new pitch creates a false
    # mismatch in an otherwise continuous phrase.
    if text.startswith("至") and first_token(text, RIGHT_HAND_TECHNIQUES) is None:
        return [], "context_dependent_transition"
    # Position-changing or residual-tone actions inherit the previous pressed
    # string.  A movement with an explicit destination (e.g. 绰上五徽六分,
    # 浒上七徽九分, 淌九徽, or 引上七徽六分) has no new right-hand attack, but its endpoint is still a
    # definite sounding melody pitch and should be compared with the aligned
    # jianpu note.  Destination-less ornaments remain non-comparable controls.
    # ``掐起/滔起`` may be written with a new left-hand finger and hui but
    # without an explicit string (e.g. ``跪指五徽滔起``).  It is nevertheless
    # a positioned continuation on the previously stopped string: its sound
    # and the following abbreviated plucks must use that newly stated hui.
    # Treating only the bare word ``滔起`` as a control used to leave the
    # older hui active, producing false warnings immediately afterwards.
    positioned_qiaqi = (
        not strings
        and any(gesture in text for gesture in ("掐起", "滔起"))
        and parse_hui(text) is not None
    )
    direction_only = (
        not strings
        and (
            text.startswith(
                ("上", "下", "进", "退", "绰", "注", "浒", "淌", "引上")
            )
            or re.fullmatch(r"(?:急|缓|少)?(?:进|退|上|下)", text) is not None
            or text in ORNAMENTS
            or positioned_qiaqi
        )
    )
    if direction_only:
        if text == "复":
            origin = context.get("walk_return_position")
            if origin and origin.get("string") == context.get("active_left_string"):
                string, destination = int(origin["string"]), origin["hui"]
                context["stopped_string"] = string
                context["stopped_hui"] = destination
                context["active_left_hui"] = destination
                context["sound_mode"] = "stopped"
                context["walk_return_position"] = None
                remember_left_position(context, string, destination)
                pitch, error = position_pitch(string, destination, open_midi, "stopped")
                return ([pitch] if error is None else []), error
            # Without a recorded origin, 复 cannot prove that the old hui
            # remains current. Do not invent a return destination.
            invalidate_active_left_hui(context)
            return [], "context_dependent_slide"
        destination = parse_hui(text)
        if destination is not None and context.get("active_left_string") is not None:
            active_string = int(context["active_left_string"])
            if (not positioned_qiaqi
                    and context.get("active_left_hui") is not None):
                context["walk_return_position"] = {
                    "string": active_string, "hui": context["active_left_hui"],
                }
            else:
                context["walk_return_position"] = None
            left_finger = first_token(text, LEFT_HAND_FINGERS)
            if left_finger is not None:
                context["left_finger"] = left_finger
            context["stopped_hui"] = destination
            context["active_left_hui"] = destination
            context["sound_mode"] = "stopped"
            remember_left_position(context, active_string, destination, left_finger)
            pitch, error = position_pitch(
                active_string, destination, open_midi, "stopped"
            )
            if error:
                return [], error
            return [pitch], None
        if re.fullmatch(r"(?:急|缓|少)?(?:进|退|上|下)", text):
            # A bare movement has no deterministic distance. Retain its
            # origin for a later 复, but invalidate the unobserved endpoint.
            if context.get("active_left_hui") is not None:
                context["walk_return_position"] = {
                    "string": context.get("active_left_string"),
                    "hui": context["active_left_hui"],
                }
            invalidate_active_left_hui(context)
        reason = (
            "ornament_or_control" if text in ORNAMENTS
            else "context_dependent_slide"
        )
        return [], reason
    if text == "泛止":
        context["harmonic"] = False
        context["harmonic_scope"] = False
        context["harmonic_hui"] = None
        context["sound_mode"] = None
        return [], "ornament_or_control"

    compound = re.search(
        r"[撮泼剌]\（?（?[^）]*?"
        r"(?P<hui>(?:十三|十二|十一|十|九|八|七|六|五|四|三|二|一)徽"
        r"(?:(?:一|二|三|四|五|六|七|八|九)分)?|徽外半|徽外)"
        r"(?P<stopped>[一二三四五六七1-7])弦(?:按音)?[＋+、](?P<open>[一二三四五六七1-7])弦散音",
        text,
    )
    # The corpus also uses the musically equivalent order “散音＋按音”,
    # especially for 撮.  Keep the same named captures so downstream pitch
    # and left-hand-state handling is identical to the canonical order.
    if compound is None:
        compound = re.search(
            r"[撮泼剌]\（?（?[^）]*?"
            r"(?P<open>[一二三四五六七1-7])弦散音[＋+、]"
            r"(?:[^）]*?)"
            r"(?P<hui>(?:十三|十二|十一|十|九|八|七|六|五|四|三|二|一)徽"
            r"(?:(?:一|二|三|四|五|六|七|八|九)分)?|徽外半|徽外)"
            r"(?P<stopped>[一二三四五六七1-7])弦(?:按音)?",
            text,
        )
    # Some generated and historical rows put the stopped string before its
    # hui: “一弦七徽按音”.  Accept it in either compound-component order.
    if compound is None:
        compound = re.search(
            r"[撮泼剌]\（?（?[^）]*?"
            r"(?P<stopped>[一二三四五六七1-7])弦"
            r"(?P<hui>(?:十三|十二|十一|十|九|八|七|六|五|四|三|二|一)徽"
            r"(?:(?:一|二|三|四|五|六|七|八|九)分)?|徽外半|徽外)(?:按音)?"
            r"[＋+、](?P<open>[一二三四五六七1-7])弦散音",
            text,
        )
    if compound is None:
        compound = re.search(
            r"[撮泼剌]\（?（?[^）]*?"
            r"(?P<open>[一二三四五六七1-7])弦散音[＋+、]"
            r"(?:[^）]*?)"
            r"(?P<stopped>[一二三四五六七1-7])弦"
            r"(?P<hui>(?:十三|十二|十一|十|九|八|七|六|五|四|三|二|一)徽"
            r"(?:(?:一|二|三|四|五|六|七|八|九)分)?|徽外半|徽外)(?:按音)?",
            text,
        )
    if compound:
        hui = parse_hui(compound.group("hui"))
        if hui is None:
            return [], "missing_hui"
        stopped, error = position_pitch(
            string_number(compound.group("stopped")), hui, open_midi
        )
        if error:
            return [], error
        opened, error = position_pitch(
            string_number(compound.group("open")), None, open_midi, "open"
        )
        # Compound surfaces in the corpus use both Arabic and Chinese string
        # designators.  Reuse the same normalizer as the pitch calculation;
        # calling ``int("五")`` here previously aborted the *entire* phrase
        # audit and silently suppressed every edit-plan warning in that batch.
        stopped_string = string_number(compound.group("stopped"))
        context["stopped_string"] = stopped_string
        context["stopped_hui"] = hui
        context["active_left_string"] = stopped_string
        context["active_left_hui"] = hui
        stopped_component = next((part for part in re.split(r'[＋+、]', double_stopped.group(1))
                                  if '散音' not in part and stopped_string in parse_string_numbers(part)), '') if double_stopped else text
        stopped_finger = first_token(stopped_component, LEFT_HAND_FINGERS)
        if stopped_finger is not None:
            context['left_finger'] = stopped_finger
        remember_left_position(context, stopped_string, hui, stopped_finger)
        context['current_note_attacks'] = [
            {'string': stopped_string, 'hui': hui, 'mode': 'stopped'},
            {'string': string_number(compound.group('open')), 'hui': None, 'mode': 'open'},
        ]
        context["sound_mode"] = None
        return ([stopped, opened] if error is None else []), error

    # A positioned right-hand pluck may omit its string while continuing a
    # known stopped-string contact: 挑四徽六分 inherits the held string,
    # but updates its hui just like 挑六弦四徽六分. Do not guess after an
    # open/harmonic attack or when a rest has cleared that contact.
    if (not strings and parse_hui(text) is not None
            and first_token(text, RIGHT_HAND_TECHNIQUES) is not None
            and context.get('sound_mode') == 'stopped'
            and isinstance(context.get('active_left_string'), int)
            and not any(token in text for token in ('散', '泛音', '泛起'))):
        strings = [context['active_left_string']]
    if not strings:
        return [], "no_explicit_string"
    hui = parse_hui(text)
    # A hui plus several strings outside an explicit compound is ambiguous.
    if hui is not None and len(strings) != 1 and not li_strings:
        return [], "ambiguous_multiple_strings"
    if starts_harmonic:
        context["harmonic"] = True
        context["harmonic_scope"] = True
        context["sound_mode"] = "harmonic"
        # Harmonics are produced after releasing the stopped-string state.
        context["active_left_string"] = None
        context["active_left_hui"] = None
        if hui is not None:
            context["harmonic_hui"] = hui
    # In “剔五弦散如一四弦”, 散 modifies only the 如一 partner (4弦).
    # The first string keeps the inherited stopped position.
    explicit_open = (
        "散" in text and not starts_harmonic and ruyi_open_string is None
    )
    explicit_left = hui is not None or any(
        finger in text for finger in LEFT_HAND_FINGERS
    )
    uses_current_position = "就" in text
    right_hand = first_token(text, RIGHT_HAND_TECHNIQUES)
    if right_hand is not None:
        context["right_hand"] = right_hand
    left_finger = first_token(text, LEFT_HAND_FINGERS)
    if left_finger is not None:
        context["left_finger"] = left_finger

    # Explicit 散音 / 按音 are per-note overrides even inside a 泛起…泛止
    # span.  Keep the surrounding harmonic region active for following
    # notes: neither spelling is a substitute for 泛止.
    if explicit_open:
        mode = "open"
        context["sound_mode"] = "open"
        # An open attack releases a stopped position only on the sounded
        # string. A left hand pressing another string can remain in place for
        # a subsequent standalone walk; keep that position while the last
        # *sounded* mode stays open for ordinary abbreviated plucks.
        held_on_other_string = (
            context.get('active_left_string') is not None
            and context.get('active_left_hui') is not None
            and context['active_left_string'] not in strings
        )
        context['left_positions'] = {
            finger: position for finger, position in context['left_positions'].items()
            if position.get('string') not in strings
        }
        if context.get('left_finger') not in context['left_positions']:
            context['left_finger'] = None
        if not held_on_other_string:
            context["active_left_string"] = None
            context["active_left_hui"] = None
    elif "按音" in text:
        mode = "stopped"
        context["sound_mode"] = "stopped"
    elif "泛音" in text or context["harmonic_scope"]:
        mode = "harmonic"
    elif explicit_left or uses_current_position:
        mode = "stopped"
    elif (context["sound_mode"] == "open"
          or (context["sound_mode"] is None
              and context["abbreviated_open_after_rest"])):
        # An abbreviated pluck after an explicit 散 remains open, even when
        # an intervening bare rest has ended the previous sounding event.
        mode = "open"
        context["sound_mode"] = "open"
        context["abbreviated_open_after_rest"] = False
    else:
        mode = "stopped"

    pitches = []
    for string in strings:
        string_mode = "open" if string == ruyi_open_string else mode
        effective_hui = None if string_mode == "open" else hui
        if effective_hui is None and string_mode == "harmonic":
            effective_hui = context["harmonic_hui"]
        elif effective_hui is None and string_mode == "stopped":
            # Jianzipu commonly omits the left-hand position while it remains
            # on the same string. “就” explicitly permits the current position
            # to be reused when the newly plucked string is different.
            if context.get("active_left_hui") is not None:
                effective_hui = context.get("active_left_hui")
            if uses_current_position and effective_hui is None:
                return [], "current_stopped_position_missing"
        if string_mode == "harmonic" and effective_hui is None:
            return [], "harmonic_hui_missing"
        pitch, error = position_pitch(
            string, effective_hui, open_midi, string_mode
        )
        if error:
            return [], error
        pitches.append(pitch)
        context['current_note_attacks'].append({
            'string': string, 'hui': effective_hui, 'mode': string_mode,
        })
        if string_mode == "harmonic":
            context["harmonic_hui"] = effective_hui
        elif effective_hui is not None:
            context["sound_mode"] = "stopped"
            context["stopped_string"] = string
            context["stopped_hui"] = effective_hui
            context["active_left_string"] = string
            context["active_left_hui"] = effective_hui
            remember_left_position(context, string, effective_hui)
    # An attack can contain a subsequent left-hand movement, e.g.
    # 名指十徽勾2弦绰上九徽. The sounding attack is at ten hui, while the
    # position inherited by later gestures is nine hui. An intervening open
    # attack on another string (散挑5弦应合) does not release that position.
    if (len(strings) == 1 and mode == "stopped" and right_hand is not None
            and pitches):
        suffix_walk = re.search(
            r"(?:绰上|注下)(?:十三|十二|十一|十|九|八|七|六|五|四|三|二|一)徽"
            r"(?:(?:一|二|三|四|五|六|七|八|九)分)?",
            text,
        )
        if suffix_walk and suffix_walk.start() > text.find(right_hand):
            destination = parse_hui(suffix_walk.group())
            if destination is not None:
                context["walk_return_position"] = {
                    "string": strings[0], "hui": context["active_left_hui"],
                }
                context["stopped_hui"] = destination
                context["active_left_hui"] = destination
                remember_left_position(context, strings[0], destination)
    if ends_harmonic_after_note:
        context["harmonic"] = False
        context["harmonic_scope"] = False
        context["harmonic_hui"] = None
        context["sound_mode"] = None
    return pitches, None


def parse_jianzi(
    text: str, open_midi: list[float], context: dict | None = None,
) -> tuple[list[float], str | None]:
    """Parse one glyph and always commit its trailing harmonic transition.

    Several specialised branches (撮双音、带起、爪起等) return before the
    ordinary single-string tail.  A trailing ``泛止`` is nevertheless a state
    transition after that glyph, so make it a wrapper-level invariant instead
    of relying on every branch to remember the same cleanup.
    """
    normalized = normalize_context(context)
    # Per-note evidence is recorded where pitch mode and inherited position
    # are resolved, before a trailing control marker clears persistent state.
    normalized['current_note_attacks'] = []
    if invalid_string_numbers(text):
        return [], 'invalid_string_number'
    compact = str(text or "").strip()
    prior_walk_origin = normalized.get('walk_return_position')
    prior_left_string = normalized.get('active_left_string')
    prior_left_hui = normalized.get('active_left_hui')
    attack_token = first_token(compact, RIGHT_HAND_TECHNIQUES)
    embedded_walk = re.search(r'绰上|注下|退|进|引上|淌下', compact)
    if (parse_string_numbers(compact)
            or first_token(compact, RIGHT_HAND_TECHNIQUES) is not None
            or "泛起" in compact or "泛止" in compact):
        normalized["walk_return_position"] = None
    # Commit a leading 泛起 before delegating to specialised parsers.  In
    # particular, the double-stopped branch returns before the normal
    # single-string state update, while corpus glyphs such as
    # ``泛起撮（…）`` still open a harmonic span.  Bracketed control labels
    # may precede 泛起, so rely on marker order rather than startswith().
    start_at = compact.find("泛起")
    stop_at = compact.find("泛止")
    if start_at >= 0 and (stop_at < 0 or start_at < stop_at):
        normalized["harmonic"] = True
        normalized["harmonic_scope"] = True
        normalized["sound_mode"] = "harmonic"
        marker_hui = parse_hui(compact)
        if marker_hui is not None:
            normalized["harmonic_hui"] = marker_hui
        normalized["active_left_string"] = None
        normalized["active_left_hui"] = None
    normalized['current_note_harmonic_scope'] = bool(
        normalized.get('harmonic_scope') and not compact.startswith('泛止')
    )
    pitches, reason = _parse_jianzi(text, open_midi, normalized)
    # A same-string attack introduced with a slide may continue an existing
    # walk. Its preceding position remains a valid destination for 复.
    # Require a proven walk origin and a resolved stopped attack; an ordinary
    # new attack or a change of string must still clear the return state.
    if (prior_walk_origin and embedded_walk and attack_token
            and embedded_walk.start() < compact.find(attack_token)
            and prior_walk_origin.get('string') == prior_left_string
            and prior_left_hui is not None and reason is None
            and len(normalized.get('current_note_attacks') or []) == 1):
        attack = normalized['current_note_attacks'][0]
        if attack.get('mode') == 'stopped' and attack.get('string') == prior_left_string:
            normalized['walk_return_position'] = {
                'string': prior_left_string, 'hui': prior_left_hui,
            }
    if normalized["sound_mode"] in {"stopped", "harmonic"}:
        normalized["abbreviated_open_after_rest"] = False
    if "泛止" in compact and not compact.startswith("泛止"):
        normalized["harmonic"] = False
        normalized["harmonic_scope"] = False
        normalized["harmonic_hui"] = None
        normalized["sound_mode"] = None
    return pitches, reason


def _standalone_walk_transitions(data: dict):
    """Yield parseable inherited 绰上/注下 start/end state transitions.

    This follows the *same* state machine as ``audit``.  Only a standalone
    walk with an explicit destination and an inherited stopped string is
    yielded. A prefixed technique on a newly articulated note (for example
    ``绰上五徽挑六弦``) is intentionally excluded because it is not the
    state-inheriting ``direction_only`` form parsed below.  Both pitch-span
    and direction validators must use this single replay so they cannot
    disagree about what the walk inherited.
    """
    metadata = data.get("metadata") or {}
    supplied_open_midi = data.get("open_midi")
    if (isinstance(supplied_open_midi, list)
            and len(supplied_open_midi) == 7
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in supplied_open_midi)):
        open_midi = [float(value) for value in supplied_open_midi]
    else:
        open_midi = parse_open_midi(metadata)

    context = new_context()
    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        if advance_context_for_bare_rest(context, note, text):
            continue
        is_standalone_walk = (
            text.startswith(("绰上", "注下"))
            and not parse_string_numbers(text)
            and first_token(text, RIGHT_HAND_TECHNIQUES) is None
        )
        previous_string = context.get("active_left_string")
        previous_hui = context.get("active_left_hui")
        destination = parse_hui(text) if is_standalone_walk else None

        # Always advance the shared parser state, including on a row that is
        # not comparable.  The next walk must inherit exactly what pitch audit
        # would inherit.
        actual, reason = parse_jianzi(text, open_midi, context)
        if (not is_standalone_walk or destination is None
                or previous_string is None or previous_hui is None
                or reason is not None or not actual):
            continue
        start_pitch, start_error = position_pitch(
            int(previous_string), previous_hui, open_midi, "stopped"
        )
        end_pitch, end_error = position_pitch(
            int(previous_string), destination, open_midi, "stopped"
        )
        if start_error or end_error or start_pitch is None or end_pitch is None:
            continue
        yield {
            "index": note.get("index"),
            "technique": "绰上" if text.startswith("绰上") else "注下",
            "string": int(previous_string),
            "from_hui": previous_hui,
            "to_hui": destination,
            "span_cents": round(abs(float(end_pitch) - float(start_pitch)) * 100.0, 1),
        }


def walk_span_warnings(data: dict, threshold_cents: float = WALK_SPAN_WARNING_CENTS) -> list[dict]:
    """Return deterministic warnings for implausibly large 绰上/注下 walks."""
    warnings: list[dict] = []
    for transition in _standalone_walk_transitions(data):
        span_cents = float(transition["span_cents"])
        if span_cents >= threshold_cents:
            warnings.append({
                **transition,
                "code": "walk_span_too_large",
                "span_cents": round(span_cents, 1),
                "threshold_cents": float(threshold_cents),
            })
    return warnings


def walk_motion_warnings(data: dict) -> list[dict]:
    """Warn on no-op and direction-reversed inherited 绰上/注下 movements.

    Numerically increasing hui positions are downward (``注下``); decreasing
    positions are upward (``绰上``).  This check is deliberately independent
    of pitch matching: a same-pitch or short movement can still be a
    musically meaningless no-op or use the wrong named technique.
    """
    warnings: list[dict] = []
    for transition in _standalone_walk_transitions(data):
        # ``徽外`` is a valid pitch position but has no ordered numeric hui
        # coordinate, so it cannot support a deterministic up/down judgment.
        try:
            start = float(transition["from_hui"])
            end = float(transition["to_hui"])
        except (TypeError, ValueError):
            continue
        technique = transition["technique"]
        if abs(end - start) < 1e-9:
            warnings.append({**transition, "code": "walk_endpoint_same_as_start"})
        elif ((technique == "绰上" and end > start)
              or (technique == "注下" and end < start)):
            warnings.append({
                **transition,
                "code": "walk_direction_reversed",
                "suggested_technique": "注下" if technique == "绰上" else "绰上",
            })
    return warnings


def zhuang_direction_warnings(data: dict) -> list[dict]:
    """Flag a bare 撞 whose written target falls below its pressed attack.

    撞 rises briefly and returns; it cannot account for a *lower* next note.
    Only compare a directly preceding, single-pitch stopped pluck.  Missing
    notation, ornaments, chords, rests and 反撞 are deliberately undecided.
    """
    metadata = data.get("metadata") or {}
    tonic = parse_tonic_midi(metadata)
    supplied = data.get("open_midi")
    open_midi = ([float(value) for value in supplied]
                 if isinstance(supplied, list) and len(supplied) == 7
                 and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                         for value in supplied)
                 else parse_open_midi(metadata))
    context = new_context()
    previous_attack_pitch = None
    warnings = []
    for note in data.get("notes", []):
        if is_structural_note(note):
            continue
        text = str(note.get("jianzi") or "").strip()
        targets = [value for value in (
            parse_jianpu(note.get("jianpu"), tonic),
            parse_jianpu(note.get("jianpu_alt"), tonic),
        ) if value is not None]
        if advance_context_for_bare_rest(context, note, text):
            previous_attack_pitch = None
            continue
        before = dict(context)
        actual, reason = parse_jianzi(text, open_midi, context)
        if (text == "撞" and len(targets) == 1
                and previous_attack_pitch is not None
                and before.get("sound_mode") == "stopped"
                and before.get("active_left_string") is not None
                and before.get("active_left_hui") is not None
                and targets[0] < previous_attack_pitch - 0.5):
            warnings.append({"index": note.get("index"),
                             "code": "jianzi_pitch_mismatch",
                             "reason": "zhuang_target_below_previous_attack"})
        attacks = context.get("current_note_attacks") or []
        previous_attack_pitch = (
            targets[0] if len(targets) == len(actual) == len(attacks) == 1
            and reason is None and attacks[0].get("mode") == "stopped"
            and abs(actual[0] - targets[0]) <= 0.5
            and first_token(text, RIGHT_HAND_TECHNIQUES) is not None
            else None
        )
    return warnings


def consecutive_walk_warnings(data: dict, threshold: int = 4) -> list[dict]:
    """Count sounding events without a new attack, including walk-covered blanks.

    The ordinary fourth event warns as before. A run containing 往来 may
    cover five events (the glyph itself included), so it warns on the sixth.
    """
    metadata = data.get("metadata") or {}
    supplied_open_midi = data.get("open_midi")
    if (isinstance(supplied_open_midi, list)
            and len(supplied_open_midi) == 7
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in supplied_open_midi)):
        open_midi = [float(value) for value in supplied_open_midi]
    else:
        open_midi = parse_open_midi(metadata)

    context = new_context()
    run_length = 0
    has_wanglai = False
    warnings: list[dict] = []
    tonic = parse_tonic_midi(metadata)
    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        is_bar = is_structural_note(note)
        if is_bar:
            continue
        if advance_context_for_bare_rest(context, note, text):
            run_length = 0
            has_wanglai = False
            continue
        sounding = parse_jianpu(note.get("jianpu"), tonic) is not None
        if not text:
            if run_length and sounding:
                run_length += 1
            else:
                run_length = 0
                has_wanglai = False
        else:
            no_right_hand = first_token(text, RIGHT_HAND_TECHNIQUES) is None
            is_wanglai = no_right_hand and "往来" in text
            is_standalone_walk = no_right_hand and (
                text.startswith(("绰上", "注下"))
                or text in {"复", "进复", "退复"}
            )
            actual, reason = parse_jianzi(text, open_midi, context)
            if sounding and (is_wanglai or (is_standalone_walk and actual and reason is None)):
                run_length += 1
                has_wanglai = has_wanglai or is_wanglai
            else:
                run_length = 0
                has_wanglai = False
        warn_at = 6 if has_wanglai else threshold
        if run_length == warn_at:
            warnings.append({
                "index": note.get("index"),
                "code": "consecutive_walks_without_attack",
                "count": run_length,
            })
    return warnings


def open_to_stopped_transition_warnings(data: dict) -> list[dict]:
    """Warn when the next sounded row stops the same string just opened.

    This is deliberately a notation/continuity advisory rather than a pitch
    error: pressing the formerly open string immediately damps its resonance
    and can spoil the intended lingering sound.  Only explicit, parseable
    ``散`` → same-string positioned notes are considered; rest, sustain, and
    otherwise ambiguous rows clear the pending open-string state.
    """
    metadata = data.get("metadata") or {}
    supplied_open_midi = data.get("open_midi")
    if (isinstance(supplied_open_midi, list)
            and len(supplied_open_midi) == 7
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in supplied_open_midi)):
        open_midi = [float(value) for value in supplied_open_midi]
    else:
        open_midi = parse_open_midi(metadata)

    context = new_context()
    pending_open_strings: set[int] = set()
    pending_open_is_compound = False
    pending_open_stopped = False
    warnings: list[dict] = []
    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        is_bar = is_structural_note(note)
        if is_bar:
            # A barline is structural, not an intervening sound.
            continue
        if advance_context_for_bare_rest(context, note, text):
            pending_open_strings = set()
            pending_open_is_compound = False
            pending_open_stopped = False
            continue

        actual, reason = parse_jianzi(text, open_midi, context)
        stopped_here = has_explicit_resonance_stop(text)
        strings = set(parse_string_numbers(text))
        # In a compound 撮/泼/剌, ``散音`` applies only to its explicitly
        # named open-string component.  The other named string is stopped;
        # treating all strings in the surface as open (or stopped) creates a
        # false warning on the following note, e.g. 撮(三弦按音＋六弦散音)
        # followed by 大指五徽挑三弦.
        compound_open_match = re.search(
            r"[撮泼剌][^）)]*?([一二三四五六七1-7])弦散音", text
        )
        compound_open_strings = (
            {string_number(compound_open_match.group(1))}
            if compound_open_match else set()
        )
        explicit_open_strings = (
            compound_open_strings
            if compound_open_match else (strings if "散" in text else set())
        )
        explicit_stopped_strings = (
            strings - compound_open_strings
            if compound_open_match and parse_hui(text) is not None
            else (strings if parse_hui(text) is not None and "散" not in text else set())
        )
        # A positioned harmonic is not a stopped-string note, even though it
        # has a hui in its surface form. Use the parser's sounding mode.
        parsed_stopped_strings = {
            attack["string"] for attack in context.get("current_note_attacks") or []
            if attack.get("mode") == "stopped"
        }
        explicit_stopped_strings &= parsed_stopped_strings
        shared = sorted(pending_open_strings & explicit_stopped_strings)
        # 罨／虚罨 deliberately strikes the newly stopped string to sound a
        # positioned note. Its contact with a preceding open string is the
        # technique itself, not an unwanted interruption of an ordinary pluck.
        if (shared and actual and reason is None and not pending_open_is_compound
                and not pending_open_stopped and "罨" not in text):
            warnings.append({
                "index": note.get("index"),
                "code": "open_to_stopped_same_string",
                "strings": shared,
            })

        # Only a plainly notated open-string sound can carry into the very
        # next event. Any other row, including a rest or an unparseable glyph,
        # ends this one-step continuity check.
        pending_open_strings = (
            explicit_open_strings
            if (actual and reason is None and not stopped_here
                and not is_continuity_exempt_compound(text))
            else set()
        )
        pending_open_is_compound = is_continuity_exempt_compound(text)
        pending_open_stopped = stopped_here
    return warnings


def ordinary_open_reattack_warnings(data: dict) -> list[dict]:
    """Adjacent ordinary single plucks of the same open string only."""
    open_midi = data.get('open_midi') or parse_open_midi(data.get('metadata') or {})
    context = new_context()
    previous = None
    warnings = []
    ordinary = re.compile(r'(?:散音|散)?[抹挑勾剔擘托劈打摘搓][一二三四五六七1-7]弦')
    for note in data.get('notes', []):
        if is_structural_note(note):
            continue
        text = str(note.get('jianzi') or '').strip()
        if advance_context_for_bare_rest(context, note, text):
            previous = None
            continue
        actual, reason = parse_jianzi(text, open_midi, context)
        attacks = context.get('current_note_attacks') or []
        current = (attacks[0]['string'] if ordinary.fullmatch(text)
                   and reason is None and len(actual) == 1 and len(attacks) == 1
                   and attacks[0]['mode'] == 'open' else None)
        if current is not None and previous is not None and current == previous[1]:
            warnings.append({'index': note.get('index'), 'previous_index': previous[0],
                             'code': 'ordinary_open_same_string_reattack', 'string': current})
        previous = (note.get('index'), current) if current is not None else None
    return warnings


def inherited_gesture_warnings(data: dict) -> list[dict]:
    """Check gestures against the same inherited state used by pitch audit.

    Unknown phrase-boundary finger/string state is not guessed.  Structural
    rows do not change state; a bare rest resets it through the shared replay.
    """
    open_midi = data.get('open_midi') or parse_open_midi(data.get('metadata') or {})
    context = new_context()
    previous_text = ''
    previous_explicit_stopped = False
    previous_walk_direction = None
    last_attack_strings = set()
    warnings: list[dict] = []
    for note in data.get('notes', []):
        if is_structural_note(note):
            continue
        text = str(note.get('jianzi') or '').strip()
        if advance_context_for_bare_rest(context, note, text):
            previous_text = ''
            previous_explicit_stopped = False
            previous_walk_direction = None
            last_attack_strings.clear()
            continue
        before = dict(context)
        has_attack_string = bool(parse_string_numbers(text))
        slide_prefix = re.match(r'^(?:(?:泛起|泛止|按音|泛音|急|缓|少))*([绰注])', text)
        slide_attack = first_token(text, RIGHT_HAND_TECHNIQUES)
        if (slide_prefix and slide_attack is not None
                and first_token(text[:text.find(slide_attack)], LEFT_HAND_FINGERS) is None):
            warnings.append({'index': note.get('index'),
                             'code': 'slide_attack_missing_left_finger',
                             'technique': slide_prefix.group(1)})
        invalid_open_ornament = re.fullmatch(
            r'(?:散音|散)(吟|猱|撞)(?:[一二三四五六七1-7]弦)?', text)
        if invalid_open_ornament:
            warnings.append({'index': note.get('index'),
                             'code': 'open_attack_ornament',
                             'technique': invalid_open_ornament.group(1)})
        if '进复' in text and before.get('active_left_string') is not None:
            if previous_walk_direction != 'down':
                warnings.append({'index': note.get('index'),
                                 'code': 'jinfu_without_downward_walk'})
        for li_match in re.finditer(r'历[一二三四五六七1234567]+弦', text):
            li_strings = parse_string_numbers(li_match.group(0))
            if any(abs(a - b) > 1 for a, b in zip(li_strings, li_strings[1:])):
                warnings.append({'index': note.get('index'),
                                 'code': 'li_nonadjacent_strings',
                                 'strings': li_strings})
        standalone_open_dependent = bool(
            text and not has_attack_string
            and first_token(text, RIGHT_HAND_TECHNIQUES) is None
            and (re.match(r'^(?:急|缓|少)?(?:绰|注|淌|浒|引上|上|下|进|退)', text)
                 or text in {'吟', '猱'})
        )
        if (before.get('sound_mode') == 'open' and standalone_open_dependent
                and before.get('active_left_string') is None):
            warnings.append({'index': note.get('index'),
                             'code': 'open_string_unplayable_gesture',
                             'technique': text})
        if text == '复' and previous_text == '复':
            warnings.append({'index': note.get('index'),
                             'code': 'repeated_fu_without_walk'})
        elif text == '复':
            origin = before.get('walk_return_position')
            if not origin or origin.get('string') != before.get('active_left_string'):
                warnings.append({'index': note.get('index'),
                                 'code': 'fu_without_walk_origin'})
        if '滔起' in text:
            explicit_strings = parse_string_numbers(text)
            string = (explicit_strings[0] if len(explicit_strings) == 1
                      else before.get('active_left_string'))
            # 徽位数字越小越靠近岳山；「七徽及以上」包含七徽、六徽、
            # 五徽及其更靠近岳山的分数徽位，而不包含七徽几分。
            tao_hui = parse_hui(text)
            if tao_hui is None:
                tao_hui = before.get('active_left_hui')
            if ('跪指' in text and tao_hui is not None
                    and (isinstance(tao_hui, str) or tao_hui > 7.0)):
                warnings.append({'index': note.get('index'),
                                 'code': 'kneeling_taoqi_below_seventh_hui',
                                 'hui': tao_hui})
            positions = before.get('left_positions') or {}
            thumb = positions.get('大指') or {}
            thumb_on_string = thumb.get('string') == string
            if thumb_on_string:
                # A newer finger on the same string supersedes the thumb;
                # a newer finger on another string does not release it.
                seen_thumb = False
                for finger, position in positions.items():
                    if finger == '大指':
                        seen_thumb = True
                    elif seen_thumb and position.get('string') == string:
                        thumb_on_string = False
                        break
            previous_string = before.get('active_left_string')
            previous_finger = before.get('left_finger')
            if (string is not None and previous_finger is not None
                    and not thumb_on_string):
                if previous_string is not None and previous_string != string:
                    warnings.append({'index': note.get('index'),
                                     'code': 'taoqi_without_thumb_origin',
                                     'string': string,
                                     'previous_string': previous_string,
                                     'previous_finger': previous_finger})
                elif previous_finger != '大指':
                    warnings.append({'index': note.get('index'),
                                     'code': 'taoqi_without_thumb_origin',
                                     'string': string,
                                     'previous_finger': previous_finger})
        harmonic_on_this_note = bool(
            (before.get('harmonic_scope') and not text.startswith('泛止'))
            or ('泛起' in text and
                ('泛止' not in text or text.find('泛起') < text.find('泛止')))
        )
        harmonic_ornaments = [name for name in ('吟', '猱', '撞') if name in text]
        if (harmonic_ornaments
                and harmonic_on_this_note
                and '按音' not in text and not previous_explicit_stopped):
            warnings.append({'index': note.get('index'),
                             'code': 'harmonic_ornament_without_stopped',
                             'technique': '／'.join(harmonic_ornaments)})
        actual, reason = parse_jianzi(text, open_midi, context)
        previous_explicit_stopped = bool(
            '按音' in text and actual and reason is None
            and any(attack['mode'] == 'stopped'
                    for attack in context.get('current_note_attacks') or [])
        )
        attacks = context.get('current_note_attacks') or []
        # Use resolved attack strings, including inherited and compound attacks.
        resolved_strings = {attack.get('string') for attack in attacks}
        if not resolved_strings and not has_attack_string:
            resolved_strings = last_attack_strings
        if (slide_attack == '挑' and '抹挑' not in text and 1 in resolved_strings):
            warnings.append({'index': note.get('index'),
                             'code': 'tiao_on_first_string', 'string': 1})
        if attacks:
            last_attack_strings = {attack.get('string') for attack in attacks}
        if attacks or '复' in text:
            previous_walk_direction = None
        elif re.match(r'^(?:急|缓|少)?(?:退|下|注|淌)', text):
            previous_walk_direction = 'down'
        elif re.match(r'^(?:急|缓|少)?(?:进|上|绰|浒|引上)', text):
            previous_walk_direction = 'up'
        if (harmonic_ornaments and attacks and not invalid_open_ornament
                and all(attack.get('mode') == 'open' for attack in attacks)):
            warnings.append({'index': note.get('index'),
                             'code': 'open_attack_ornament',
                             'technique': '／'.join(harmonic_ornaments)})
        # Compare resolved contact positions, including a preceding walk.
        # Judge only a single stopped attack by the thumb; compound attacks
        # and open/harmonic modes do not establish this hand-shape relation.
        if (reason is None and len(actual) == 1
                and len(context.get('current_note_attacks') or []) == 1
                and context['current_note_attacks'][0].get('mode') == 'stopped'
                and context.get('left_finger') == '大指'
                and before.get('sound_mode') == 'stopped'
                and before.get('left_finger') in {'中指', '名指'}
                and isinstance(before.get('active_left_string'), int)
                and isinstance(before.get('active_left_hui'), (float, int))
                and isinstance(context.get('active_left_string'), int)
                and isinstance(context.get('active_left_hui'), (float, int))
                and context['active_left_string'] > before['active_left_string']
                and context['active_left_hui'] > before['active_left_hui'] + 1e-6):
            warnings.append({'index': note.get('index'),
                             'code': 'thumb_lower_left_position',
                             'previous_finger': before['left_finger'],
                             'from_string': before['active_left_string'],
                             'from_hui': before['active_left_hui'],
                             'to_string': context['active_left_string'],
                             'to_hui': context['active_left_hui']})
        if (reason is None and len(actual) == 1 and len(attacks) == 1
                and attacks[0].get('mode') == 'stopped'
                and before.get('sound_mode') == 'stopped'
                and before.get('left_finger') in LEFT_HAND_FINGERS
                and context.get('left_finger') in LEFT_HAND_FINGERS
                and before['left_finger'] != context['left_finger']
                and first_token(text, RIGHT_HAND_TECHNIQUES) is not None
                and not any(t in text for t in ('换指', '替指', '滔起', '带起', '掐起', '抓起', '爪起', '罨'))
                and isinstance(before.get('active_left_string'), int)
                and context.get('active_left_string') == before['active_left_string']
                and isinstance(before.get('active_left_hui'), (float, int))
                and isinstance(context.get('active_left_hui'), (float, int))
                and abs(before['active_left_hui'] - context['active_left_hui']) < 1e-6):
            warnings.append({'index': note.get('index'),
                             'code': 'same_position_finger_change',
                             'string': context['active_left_string'],
                             'hui': context['active_left_hui'],
                             'previous_finger': before['left_finger'],
                             'finger': context['left_finger']})
        previous_text = text
    return warnings


def harmonic_continuity_warnings(data: dict) -> list[dict]:
    """Return advisory warnings for incompatible harmonic continuations.

    ``泛起…泛止`` is a harmonic *region*, rather than a stopped-note
    position.  Consequently a standalone ``绰上``/``注下`` cannot inherit a
    playable pressed-string origin inside that region.  Also, immediately
    sounding the same string again damps the preceding harmonic just as an
    immediate stopped note damps a preceding open string.  These are musical
    continuity advisories, not pitch-validity failures.
    """
    metadata = data.get("metadata") or {}
    supplied_open_midi = data.get("open_midi")
    if (isinstance(supplied_open_midi, list)
            and len(supplied_open_midi) == 7
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in supplied_open_midi)):
        open_midi = [float(value) for value in supplied_open_midi]
    else:
        open_midi = parse_open_midi(metadata)

    context = new_context()
    pending_harmonic_strings: set[int] = set()
    pending_harmonic_is_compound = False
    pending_harmonic_stopped = False
    pending_harmonic_hui = None
    pending_harmonic_target = None
    tonic_midi = parse_tonic_midi(metadata)
    warnings: list[dict] = []
    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        is_bar = is_structural_note(note)
        if is_bar:
            continue

        # Match ``audit`` and phrase-boundary replay: a score-level bare
        # rest terminates a 泛起…泛止 region even when the printed score
        # omitted an explicit 泛止.  This warning pass previously advanced
        # only the jianzi parser, which cannot see that an empty row is a
        # rest; a historical 泛起 could therefore leak across the rest and
        # manufacture a later same-string harmonic warning.
        if advance_context_for_bare_rest(context, note, text):
            pending_harmonic_strings = set()
            pending_harmonic_is_compound = False
            pending_harmonic_stopped = False
            pending_harmonic_hui = None
            pending_harmonic_target = None
            continue

        context_before = dict(context)
        strings = set(parse_string_numbers(text))
        actual, reason = parse_jianzi(text, open_midi, context)
        stopped_here = has_explicit_resonance_stop(text)

        # Invalid harmonic positions cannot produce an audited pitch. Report
        # the physical constraint explicitly instead of silently skipping it.
        if reason == "fractional_hui_not_valid_for_harmonic":
            warnings.append({
                "index": note.get("index"),
                "code": "fractional_hui_in_harmonic",
            })

        attacks = context.get('current_note_attacks') or []
        is_harmonic_attack = bool(
            actual and reason is None and strings
            and attacks and all(a['mode'] == 'harmonic' for a in attacks)
        )

        harmonic_walk = re.match(
            r"^(?:急|缓|少)?(绰上|注下|引上|上|下|进|退|淌|浒)", text
        )
        if (context.get('current_note_harmonic_scope')
                and harmonic_walk and not strings
                and first_token(text, RIGHT_HAND_TECHNIQUES) is None):
            warnings.append({
                "index": note.get("index"),
                "code": "walk_inside_harmonic_region",
                "technique": harmonic_walk.group(1),
            })

        # 历 articulates its strings in order. Only its first attack touches
        # the immediately preceding sound; the later attacks follow that
        # first string rather than the previous score row.
        ordered_li = bool(parse_li_string_sequence(text) and attacks)
        initial_strings = {attacks[0]['string']} if ordered_li else strings
        shared = sorted(pending_harmonic_strings & initial_strings)
        current_hui = attacks[0]['hui'] if len(attacks) == 1 else None
        # Rearticulating a harmonic at the same effective position does not
        # move the contact point. Technique names (including compound or
        # abbreviated spellings) are irrelevant to this continuity advisory.
        same_harmonic_position = bool(
            is_harmonic_attack and len(actual) == 1
            and len(strings) == 1 and strings == pending_harmonic_strings
            and current_hui is not None and current_hui == pending_harmonic_hui
        )
        if (shared and actual and reason is None and not pending_harmonic_is_compound
                and not pending_harmonic_stopped
                and not same_harmonic_position):
            warning = {
                "index": note.get("index"),
                "code": "harmonic_to_same_string_attack",
                "strings": shared,
            }
            current_target = parse_jianpu(note.get("jianpu"), tonic_midi)
            if (is_harmonic_attack and len(actual) == 1 and len(strings) == 1
                    and current_hui is not None and pending_harmonic_hui is not None
                    and current_hui != pending_harmonic_hui
                    and current_target is not None and pending_harmonic_target is not None
                    and current_target == pending_harmonic_target):
                warning["same_pitch_different_hui"] = True
            warnings.append(warning)

        # The advisory only spans the immediately following sounded event.
        # A bare rest, sustain, malformed glyph, or non-harmonic note ends it.
        pending_harmonic_strings = (
            ({attacks[-1]['string']} if ordered_li else strings)
            if (is_harmonic_attack and not stopped_here
                        and not is_continuity_exempt_compound(text))
            else set()
        )
        pending_harmonic_is_compound = is_continuity_exempt_compound(text)
        pending_harmonic_stopped = stopped_here
        single_harmonic = is_harmonic_attack and len(strings) == 1 and len(actual) == 1
        pending_harmonic_hui = (attacks[-1]['hui'] if ordered_li and is_harmonic_attack
                                else current_hui if single_harmonic else None)
        pending_harmonic_target = (
            parse_jianpu(note.get("jianpu"), tonic_midi)
            if single_harmonic and not note.get("jianpu_alt") else None
        )
    return warnings


def nonadjacent_pluck_pair_warnings(data: dict) -> list[dict]:
    """Warn when a two-string 拨/泼 gesture names non-adjacent strings.

    ``拨``/``泼`` (including 拨剌/泼剌) are neighboring-string gestures.
    Do not infer missing strings or judge compound gestures with more than two
    declared strings; this only catches the explicit two-string notation that
    is mechanically impossible as a single paired stroke.
    """
    warnings: list[dict] = []
    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        match = re.search(r"(拨|泼)(剌)?", text)
        if not match:
            continue
        strings = list(dict.fromkeys(parse_string_numbers(text)))
        if len(strings) != 2 or abs(strings[0] - strings[1]) <= 1:
            continue
        warnings.append({
            "index": note.get("index"),
            "code": "nonadjacent_pluck_pair",
            "technique": match.group(0),
            "strings": strings,
        })
    return warnings


def same_string_cuo_warnings(data: dict) -> list[dict]:
    """Warn when both explicitly written components of 撮 use one string."""
    warnings: list[dict] = []
    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        if "撮" not in text:
            continue
        # Inspect only the parenthesized component description. Other string
        # references elsewhere in the row (e.g. a preceding instruction) are
        # not part of the two-string 撮.
        match = re.search(r"撮\s*[（(]([^）)]*)[）)]", text)
        if not match:
            continue
        components = parse_string_numbers(match.group(1))
        if len(components) == 2 and components[0] == components[1]:
            warnings.append({
                "index": note.get("index"),
                "code": "same_string_cuo_pair",
                "strings": [components[0]],
            })
    return warnings


def same_finger_reach_warnings(
    data: dict,
    threshold: float = SAME_FINGER_REACH_WARNING_DISTANCE,
) -> list[dict]:
    """Warn when adjacent stopped attacks make an implausibly large same-finger reach.

    The reach measure gives string travel half the weight of hui travel:
    ``abs(current_hui - previous_hui) + abs(current_string - previous_string) / 2``.
    We only judge two adjacent, singly parseable stopped *attacks* with a
    resolved named left-hand finger.  Bars are structural and do not break the
    sequence; rests, sustains, walks, compounds, open strings, harmonics, and
    ambiguous/unparseable rows do.  This keeps the warning advisory and avoids
    guessing at complex multi-stop ergonomics.
    """
    metadata = data.get("metadata") or {}
    supplied_open_midi = data.get("open_midi")
    if (isinstance(supplied_open_midi, list)
            and len(supplied_open_midi) == 7
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in supplied_open_midi)):
        open_midi = [float(value) for value in supplied_open_midi]
    else:
        open_midi = parse_open_midi(metadata)

    context = new_context()
    previous: dict | None = None
    warnings: list[dict] = []
    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        is_bar = (str(note.get("abc") or "").strip() == "|"
                  or str(note.get("jianpu") or "").strip() == "|"
                  or note.get("duration") == "小节线")
        if is_bar:
            continue
        if advance_context_for_bare_rest(context, note, text):
            previous = None
            continue

        actual, reason = parse_jianzi(text, open_midi, context)
        strings = parse_string_numbers(text)
        finger = context.get("left_finger")
        hui = context.get("active_left_hui")
        string = context.get("active_left_string")
        is_stopped_attack = (
            reason is None
            and len(actual) == 1
            and len(strings) == 1
            and first_token(text, RIGHT_HAND_TECHNIQUES) is not None
            and context.get("sound_mode") == "stopped"
            and finger in LEFT_HAND_FINGERS
            and isinstance(string, int)
            and isinstance(hui, (int, float))
        )
        if not is_stopped_attack:
            previous = None
            continue

        current = {
            "index": note.get("index"),
            "finger": str(finger),
            "string": int(string),
            "hui": float(hui),
        }
        if previous is not None and current["finger"] == previous["finger"]:
            hui_delta = abs(current["hui"] - previous["hui"])
            string_delta = abs(current["string"] - previous["string"])
            distance = hui_delta + string_delta / 2.0
            if distance >= threshold:
                warnings.append({
                    "index": current["index"],
                    "code": "same_finger_reach_too_large",
                    "finger": current["finger"],
                    "from_string": previous["string"],
                    "from_hui": previous["hui"],
                    "to_string": current["string"],
                    "to_hui": current["hui"],
                    "hui_delta": round(hui_delta, 3),
                    "string_delta": int(string_delta),
                    "distance": round(distance, 3),
                    "threshold": float(threshold),
                })
        previous = current
    return warnings


def low_hui_stopped_warnings(data: dict) -> list[dict]:
    """Warn about mechanically cramped low-hui stopped notes.

    This only evaluates a parseable, explicitly attacked stopped note.  Open
    strings, harmonics, rests/sustains, and ambiguous/compound surfaces are
    not guessed at.  The requested ergonomic boundaries are:

    * strings 1--4: hui <= 4 (including fractional fourth-hui positions);
    * string 5: hui < 4;
    * strings 6--7: no warning from this rule.
    """
    metadata = data.get("metadata") or {}
    supplied_open_midi = data.get("open_midi")
    if (isinstance(supplied_open_midi, list)
            and len(supplied_open_midi) == 7
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in supplied_open_midi)):
        open_midi = [float(value) for value in supplied_open_midi]
    else:
        open_midi = parse_open_midi(metadata)

    context = new_context()
    warnings: list[dict] = []

    def append_if_cramped(index: object, string: int, hui: float | str) -> None:
        if not isinstance(hui, (int, float)):
            return
        hui_value = float(hui)
        is_cramped = (
            (1 <= string <= 4 and hui_value <= LOW_HUI_STOPPED_LIMITS["strings_1_to_4_max_hui"])
            or (string == 5 and hui_value < LOW_HUI_STOPPED_LIMITS["string_5_max_exclusive_hui"])
        )
        if is_cramped:
            warnings.append({
                "index": index,
                "code": "low_hui_stopped_too_cramped",
                "string": int(string),
                "hui": hui_value,
                "strings_1_to_4_max_hui": LOW_HUI_STOPPED_LIMITS["strings_1_to_4_max_hui"],
                "string_5_max_exclusive_hui": LOW_HUI_STOPPED_LIMITS["string_5_max_exclusive_hui"],
            })

    for note in data.get("notes", []):
        text = str(note.get("jianzi") or "").strip()
        if advance_context_for_bare_rest(context, note, text):
            continue
        actual, reason = parse_jianzi(text, open_midi, context)
        strings = parse_string_numbers(text)
        hui = context.get("active_left_hui")
        string = context.get("active_left_string")

        # A 撮/泼/剌 written as explicit parenthesized components can contain
        # one stopped component plus one open component. It has no scalar
        # sound_mode after parsing by design, but its stopped component is
        # perfectly unambiguous for this local ergonomic check.
        compound = re.search(r"[撮泼剌]\s*[（(]([^）)]*)[）)]", text)
        if compound and "泛" not in text:
            for component in re.split(r"[＋+]", compound.group(1)):
                if "按音" not in component:
                    continue
                component_strings = parse_string_numbers(component)
                component_hui = parse_hui(component)
                if len(component_strings) == 1 and component_hui is not None:
                    append_if_cramped(note.get("index"), component_strings[0], component_hui)
            continue

        is_explicit_stopped_attack = (
            reason is None
            and len(actual) == 1
            and len(strings) == 1
            and first_token(text, RIGHT_HAND_TECHNIQUES) is not None
            and "泛" not in text
            and context.get("sound_mode") == "stopped"
            and isinstance(string, int)
            and int(string) in strings
            and isinstance(hui, (int, float))
        )
        if not is_explicit_stopped_attack:
            continue
        append_if_cramped(note.get("index"), int(string), hui)
    return warnings


def best_pairing(expected: list[float], actual: list[float]) -> list[dict]:
    best = None
    for perm in itertools.permutations(actual):
        pairs = [
            {
                "jianpu_midi": target,
                "jianpu_name": midi_name(target),
                "jianzi_midi": value,
                "jianzi_name": midi_name(value),
                "delta_cents": round((value - target) * 100, 3),
                "absolute_cents": round(abs(value - target) * 100, 3),
            }
            for target, value in zip(expected, perm)
        ]
        score = sum(pair["absolute_cents"] for pair in pairs)
        if best is None or score < best[0]:
            best = (score, pairs)
    return best[1] if best else []


def audit(data: dict, tolerance_cents: float) -> dict:
    metadata = data.get("metadata") or {}
    tonic_midi = parse_tonic_midi(metadata)
    # Runtime callers that already normalized the score tuning should pass it
    # explicitly.  This avoids silently falling back to (or reconstructing)
    # another tuning from descriptive metadata.
    supplied_open_midi = data.get("open_midi")
    if (isinstance(supplied_open_midi, list)
            and len(supplied_open_midi) == 7
            and all(isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    for value in supplied_open_midi)):
        open_midi = [float(value) for value in supplied_open_midi]
    else:
        open_midi = parse_open_midi(metadata)
    details = []
    pair_errors = []
    counts = {"matched": 0, "mismatched": 0, "skipped": 0}
    context = new_context()

    notes = list(data.get("notes", []))
    for note_position, note in enumerate(notes):
        context_before_note = dict(context)
        expected = [
            value for value in (
                parse_jianpu(note.get("jianpu"), tonic_midi),
                parse_jianpu(note.get("jianpu_alt"), tonic_midi),
            )
            if value is not None
        ]
        actual, reason = parse_jianzi(
            note.get("jianzi", ""), open_midi, context
        )
        text = str(note.get("jianzi", ""))
        li_strings = parse_li_string_sequence(text)
        paired_strokes = bool(re.search(r'勾剔|抹挑|抹勾|挑勾|涓',text)
                              and len(parse_string_numbers(text)) == 1
                              and len(actual) == 1 and reason is None)
        if paired_strokes:
            following = next((n for n in notes[note_position+1:] if not is_structural_note(n)), None)
            paired_strokes = bool(following and not str(following.get('jianzi') or '').strip())
        if paired_strokes:
            actual = actual * 2
        sequential = bool(li_strings or paired_strokes)
        covered_indices = []
        if sequential and len(actual) > 1:
            sequential_expected = []
            for covered_note in notes[note_position:]:
                # A barline separates measures, not successive sounded notes.
                # Rests and sustains still end the sweep's target sequence.
                if is_structural_note(covered_note):
                    continue
                value = parse_jianpu(covered_note.get("jianpu"), tonic_midi)
                if value is None:
                    break
                sequential_expected.append(value)
                covered_indices.append(covered_note.get("index"))
                if len(sequential_expected) == len(actual):
                    break
            if sequential_expected:
                expected = sequential_expected
        right = first_token(text, RIGHT_HAND_TECHNIQUES)
        if right and text.find("绰") != -1 and text.find("绰") < text.find(right):
            # A prefix 绰 can describe a context-dependent contour.  When the
            # glyph also states one string, one hui, and one articulated pitch,
            # however, that endpoint is fully auditable (e.g.
            # “绰中指十徽勾一弦” has the same sounding endpoint as
            # “中指十徽勾一弦”).  Skip only when the endpoint is not explicit
            # and uniquely resolved; do not discard a pitch the parser proved.
            explicit_single_endpoint = (
                len(parse_string_numbers(text)) == 1
                and parse_hui(text) is not None
                and len(actual) == 1
                and reason is None
            )
            if not explicit_single_endpoint:
                actual, reason = [], "context_dependent_pre_attack_technique"
        elif (right is None
              and text.startswith(("绰", "注", "淌", "浒", "引上", "上", "下", "进", "退"))
              and not actual):
            # Without an inherited stopped string or an explicit endpoint,
            # this is still only a symbolic movement.  Conversely,
            # ``parse_jianzi`` returns an endpoint pitch for forms such as
            # “注下七徽三分” and “绰上七徽” once the preceding stopped string
            # is known.  Preserve that deterministic result for both the
            # offline audit and edit_plan warnings.
            reason = reason or "context_dependent_slide"
        row = {
            "index": note.get("index"),
            "jianpu": note.get("jianpu"),
            "jianpu_alt": note.get("jianpu_alt"),
            "jianzi": note.get("jianzi", ""),
            "lyric": note.get("lyric", ""),
        }
        # Preserve the exact per-note resolution before subsequent notes (or
        # a trailing 泛止) change persistent state. Consumers must not infer
        # sounding positions independently from surface text.
        resolved = []
        for attack in context.get('current_note_attacks') or []:
            value, error = position_pitch(attack['string'], attack.get('hui'),
                                          open_midi, attack['mode'])
            if error is None and any(abs(value - v) < 1e-6 for v in actual):
                resolved.append({**attack, 'midi': value})
        if not resolved and len(actual) == 1 and reason is None:
            # Standalone walks and positioned 滔起 resolve through the active
            # left-hand position, rather than the ordinary attack branch.
            string = context.get('active_left_string')
            hui = context.get('active_left_hui')
            if isinstance(string, int) and hui is not None:
                value, error = position_pitch(string, hui, open_midi, 'stopped')
                if error is None and abs(value - actual[0]) < 1e-6:
                    resolved.append({'string': string, 'hui': hui,
                                     'mode': 'stopped', 'midi': value})
        row['resolved_pitches'] = resolved
        row['expected_midis'] = expected
        row['actual_midis'] = actual
        if covered_indices:
            row["covered_indices"] = covered_indices
        if not expected:
            row.update(status="skipped", reason="no_sounding_jianpu")
        elif is_pitch_ambiguous_sweep(text):
            row.update(status="skipped", reason="compound_sweep_pitch_ambiguous")
        elif reason:
            row.update(status="skipped", reason=reason)
        elif any(gesture in text for gesture in ("应合", "放合")) and len(expected) == 2:
            # These compound gestures can encode a coordinated two-note event
            # while the surface parse only exposes one independently verifiable
            # pitch (or the two strings are intended to merge as one sound).
            # Accept when any parsed member matches either written pitch.
            candidates = [
                {
                    "expected": target,
                    "actual": value,
                    "delta_cents": round((value - target) * 100, 3),
                    "absolute_cents": round(abs(value - target) * 100, 3),
                }
                for target in expected for value in actual
            ]
            best = min(candidates, key=lambda pair: pair["absolute_cents"])
            matched = best["absolute_cents"] <= tolerance_cents
            gesture = next(name for name in ("应合", "放合") if name in text)
            reason_prefix = {"应合": "yinghe", "放合": "fanghe"}[gesture]
            row.update(
                status="matched" if matched else "mismatched",
                reason=f"{reason_prefix}_single_pitch_match" if matched
                else f"{reason_prefix}_no_pitch_match",
                pairs=[best],
                expected_pitch_count=len(expected),
                actual_pitch_count=len(actual),
            )
            pair_errors.append(best["absolute_cents"])
        elif len(expected) != len(actual):
            # Single-symbol rows whose jianzi sounds several strings are the
            # corpus's "main note + implied partner" shorthand, not added
            # voices: 83% of single-symbol 撮 annotations are unison
            # doublings and most of the rest octaves.  The row passes when
            # at least one sounded member matches the notated pitch; the
            # reverse (two notated pitches, one sounded) stays a mismatch.
            main_note_pairs = [
                {"expected": expected[0], "actual": value,
                 "delta_cents": round((value - expected[0]) * 100, 3),
                 "absolute_cents": round(abs(value - expected[0]) * 100, 3)}
                for value in actual
            ]
            best = min(main_note_pairs, key=lambda pair: pair["absolute_cents"])
            if (not sequential and len(expected) == 1 and len(actual) > 1
                    and best["absolute_cents"] <= tolerance_cents):
                row.update(
                    status="matched",
                    reason="single_symbol_multivoice_main_note",
                    pairs=[best],
                )
            else:
                row.update(
                    status="mismatched",
                    reason="li_pitch_count_mismatch" if li_strings else "pitch_count_mismatch",
                    expected_pitch_count=len(expected),
                    actual_pitch_count=len(actual),
                )
        else:
            # 历 sounds its strings in written order; unlike a chord, its
            # pitches must not be permuted to find a better match.
            pairs = ([{
                'jianpu_midi': target,
                'jianpu_name': midi_name(target),
                'jianzi_midi': value,
                'jianzi_name': midi_name(value),
                'delta_cents': round((value - target) * 100, 3),
                'absolute_cents': round(abs(value - target) * 100, 3),
            } for target, value in zip(expected, actual)]
                     if sequential else best_pairing(expected, actual))
            matched = all(
                pair["absolute_cents"] <= tolerance_cents for pair in pairs
            )
            row.update(
                status="matched" if matched else "mismatched",
                pairs=pairs,
                mean_absolute_cents=round(
                    sum(pair["absolute_cents"] for pair in pairs) / len(pairs), 3
                ),
            )
            # A candidate lookup can legitimately offer a stopped-string
            # position even while the score is inside 泛起…泛止.  If the
            # submitted shorthand inherits that harmonic state but the same
            # surface becomes exact after explicitly leaving it, expose this
            # narrow diagnostic to the editing tool rather than a bare,
            # misleading pitch warning.
            inherited_harmonic = bool(
                context_before_note.get("harmonic_scope")
                or context_before_note.get("harmonic")
            )
            explicit_mode = any(token in text for token in ("按音", "散", "泛起", "泛止"))
            if not matched and inherited_harmonic and not explicit_mode:
                stopped_context = dict(context_before_note)
                stopped_context.update({
                    "harmonic": False,
                    "harmonic_scope": False,
                    "harmonic_hui": None,
                    "sound_mode": None,
                })
                stopped_actual, stopped_reason = parse_jianzi(
                    text, open_midi, stopped_context
                )
                if stopped_reason is None and len(stopped_actual) == len(expected):
                    stopped_pairs = best_pairing(expected, stopped_actual)
                    if all(pair["absolute_cents"] <= tolerance_cents
                           for pair in stopped_pairs):
                        row["possible_harmonic_state_mismatch"] = True
            pair_errors.extend(pair["absolute_cents"] for pair in pairs)
        counts[row["status"]] += 1
        details.append(row)
        advance_context_for_bare_rest(context, note, text)

    compared = counts["matched"] + counts["mismatched"]
    return {
        "summary": {
            "total_notes": len(details),
            "compared_notes": compared,
            **counts,
            "match_rate": round(counts["matched"] / compared, 6) if compared else None,
            "compared_pitch_pairs": len(pair_errors),
            "mean_absolute_cents": (
                round(sum(pair_errors) / len(pair_errors), 3)
                if pair_errors else None
            ),
            "tolerance_cents": tolerance_cents,
        },
        "evidence": {
            "tonic_midi": tonic_midi,
            "open_midi": open_midi,
            "compound_pitch_order": "unordered_minimum_error_pairing",
            "skipped_context_dependent_techniques": True,
        },
        "details": details,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="jianpu_jianzi_readable.json")
    parser.add_argument("-o", "--output", type=Path, help="write full JSON report")
    parser.add_argument("--tolerance-cents", type=float, default=50.0)
    parser.add_argument(
        "--show", choices=("summary", "mismatches", "all"), default="summary"
    )
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        report = audit(json.load(handle), args.tolerance_cents)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    if args.show == "summary":
        shown = report["summary"]
    elif args.show == "mismatches":
        shown = {
            "summary": report["summary"],
            "mismatches": [
                row for row in report["details"] if row["status"] == "mismatched"
            ],
        }
    else:
        shown = report
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
