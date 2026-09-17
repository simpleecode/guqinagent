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
    "历", "歷", "厉",
)
LEFT_HAND_FINGERS = ("大指", "名指", "中指", "食指", "跪指")
ORNAMENTS = {"吟", "猱", "撞", "逗", "往来", "复", "掐起", "带起", "抓起", "爪起", "滔起"}


def parse_string_numbers(text: str) -> list[int]:
    """Parse Arabic or Chinese string numbers from a jianzi surface text."""
    tokens = re.findall(r"(十(?:一|二|三)?|[一二三四五六七]|[1-7])弦", text)
    values = []
    for token in tokens:
        value = ZH_NUMBERS.get(token)
        if value is None:
            value = int(token)
        if 1 <= value <= 7:
            values.append(value)
    return values


def string_number(token: str) -> int:
    value = ZH_NUMBERS.get(token)
    return int(value) if value is not None else int(token)


def parse_li_string_sequence(text: str) -> list[int]:
    """Return the ordered strings sounded by 历/歷/厉.

    历 is a sequential index-finger sweep, not a simultaneous chord.  Corpus
    surfaces occur both compactly (历四三) and with separators/弦 (历四、三弦).
    """
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
        "left_finger": None,
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
    return context


def first_token(text: str, choices: tuple[str, ...]) -> str | None:
    return next((token for token in choices if token in text), None)


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


def parse_jianzi(
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
    starts_harmonic = text.startswith("泛起")
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
    direction_only = (
        not strings
        and (
            text.startswith(
                ("上", "下", "进", "退", "绰", "注", "浒", "淌", "引上")
            )
            or text in ORNAMENTS
        )
    )
    if direction_only:
        destination = parse_hui(text)
        if destination is not None and context.get("active_left_string") is not None:
            active_string = int(context["active_left_string"])
            context["stopped_hui"] = destination
            context["active_left_hui"] = destination
            context["sound_mode"] = "stopped"
            pitch, error = position_pitch(
                active_string, destination, open_midi, "stopped"
            )
            if error:
                return [], error
            return [pitch], None
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
        r"(?P<stopped>[一二三四五六七1-7])弦按音[＋+](?P<open>[一二三四五六七1-7])弦散音",
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
        context["stopped_string"] = int(compound.group("stopped"))
        context["stopped_hui"] = hui
        context["active_left_string"] = int(compound.group("stopped"))
        context["active_left_hui"] = hui
        context["sound_mode"] = None
        return ([stopped, opened] if error is None else []), error

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
    explicit_open = "散" in text and not starts_harmonic
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
        # “散” is explicit left-hand information: it releases any former
        # stopped position, so a following abbreviated pluck cannot inherit
        # an obsolete hui.
        context["active_left_string"] = None
        context["active_left_hui"] = None
    elif "按音" in text:
        mode = "stopped"
        context["sound_mode"] = "stopped"
    elif "泛音" in text or context["harmonic_scope"]:
        mode = "harmonic"
    elif explicit_left or uses_current_position:
        mode = "stopped"
    elif context["sound_mode"] == "open":
        # A neighbouring abbreviated pluck after an explicit 散 remains open.
        mode = "open"
    else:
        mode = "stopped"

    pitches = []
    for string in strings:
        effective_hui = hui
        if effective_hui is None and mode == "harmonic":
            effective_hui = context["harmonic_hui"]
        elif effective_hui is None and mode == "stopped":
            # Jianzipu commonly omits the left-hand position while it remains
            # on the same string. “就” explicitly permits the current position
            # to be reused when the newly plucked string is different.
            if context.get("active_left_hui") is not None:
                effective_hui = context.get("active_left_hui")
            if uses_current_position and effective_hui is None:
                return [], "current_stopped_position_missing"
        if mode == "harmonic" and effective_hui is None:
            return [], "harmonic_hui_missing"
        pitch, error = position_pitch(string, effective_hui, open_midi, mode)
        if error:
            return [], error
        pitches.append(pitch)
        if mode == "harmonic":
            context["harmonic_hui"] = effective_hui
        elif effective_hui is not None:
            context["sound_mode"] = "stopped"
            context["stopped_string"] = string
            context["stopped_hui"] = effective_hui
            context["active_left_string"] = string
            context["active_left_hui"] = effective_hui
    if ends_harmonic_after_note:
        context["harmonic"] = False
        context["harmonic_scope"] = False
        context["harmonic_hui"] = None
        context["sound_mode"] = None
    return pitches, None


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
        covered_indices = []
        if li_strings and len(actual) > 1:
            sequential_expected = []
            for covered_note in notes[note_position:note_position + len(actual)]:
                value = parse_jianpu(covered_note.get("jianpu"), tonic_midi)
                if value is None:
                    sequential_expected = []
                    break
                sequential_expected.append(value)
                covered_indices.append(covered_note.get("index"))
            if len(sequential_expected) == len(actual):
                expected = sequential_expected
        right = first_token(text, RIGHT_HAND_TECHNIQUES)
        if right and text.find("绰") != -1 and text.find("绰") < text.find(right):
            # A prefix 绰 describes a contour into a newly articulated note.
            # parse_jianzi has already updated position context, but its one
            # endpoint must not be presented as a complete pitch proof.
            actual, reason = [], "context_dependent_pre_attack_technique"
        elif right is None and text.startswith(
            ("绰", "注", "淌", "浒", "引上", "上", "下", "进", "退")
        ):
            # A standalone left-hand movement continues an existing sounding
            # string. Its written destination updates performance context, but
            # octave/register conventions and the full contour are not safely
            # reducible to one aligned MIDI value. Keep the symbolic action as
            # supervision and make scalar pitch evidence advisory only.
            actual, reason = [], "context_dependent_slide"
        row = {
            "index": note.get("index"),
            "jianpu": note.get("jianpu"),
            "jianpu_alt": note.get("jianpu_alt"),
            "jianzi": note.get("jianzi", ""),
            "lyric": note.get("lyric", ""),
        }
        if covered_indices:
            row["covered_indices"] = covered_indices
        if not expected:
            row.update(status="skipped", reason="no_sounding_jianpu")
        elif reason:
            row.update(status="skipped", reason=reason)
        elif len(expected) != len(actual):
            row.update(
                status="mismatched",
                reason="pitch_count_mismatch",
                expected_pitch_count=len(expected),
                actual_pitch_count=len(actual),
            )
        else:
            pairs = best_pairing(expected, actual)
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
        if "休止" in str(note.get("jianpu", "")) and not note.get("jianzi"):
            # A bare rest releases stopped/open-string hand positions, but it
            # does not end 泛起…泛止.  Preserve the harmonic span and its hui;
            # only an explicit 泛止 may close it.
            harmonic_scope = bool(context.get("harmonic_scope") or context.get("harmonic"))
            harmonic_hui = context.get("harmonic_hui")
            context.clear()
            context.update(new_context())
            if harmonic_scope:
                context["harmonic"] = True
                context["harmonic_scope"] = True
                context["sound_mode"] = "harmonic"
                context["harmonic_hui"] = harmonic_hui

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
