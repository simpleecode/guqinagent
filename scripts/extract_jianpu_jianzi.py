# -*- coding: utf-8 -*-
"""
Extract & map the Jianpu (简谱, numbered notation) and Jianzipu (减字谱, guqin
tab) sequences from raw_data.json.

raw_data.json is a Frida-captured runtime object from the 丝桐里 (Sitongli)
guqin Android app (GET /v2/scores/161789/data).  The structure is:

    raw_score_data.notes[] = [
        {
          "note": "q1'",            # raw jianpu token
          "note_decoded": {          # decoded jianpu
              "duration": {...}, "pitch": {...}, "slurs": [...]
          },
          "jians": [ {...}, ... ],   # list of jianzipu glyphs
          "lyric": "...",            # lyrics
        },
        ...
    ]

Jianpu token grammar (prefix duration, digit pitch, suffix octave):
    duration prefix:  ''  = crotchet (1 beat)
                      'q' = quaver    (1/2 beat)
                      's' = semiquaver(1/4 beat)
                      'd' = demisemiquaver (1/8 beat)
                      '-' = prolongation / tie
    pitch:  '0' (rest) .. '7'
    octave suffix: ',' = one octave down,  "'" = one octave up
                   (repeat for multiple octaves)

Jianzipu glyph "std" dict.  The "tp" field tags the glyph kind; the letter
fields a/b/c/d/e/f/g/h/z/t/y/x hold the component slots.  Mapping table below.
"""
import importlib.util
import json
import os
import re
import argparse
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Mapping tables (丝桐里 internal pinyin-letter codes  ->  human readable)
# ---------------------------------------------------------------------------

# Left-hand fingers  (field a / z).  Pinyin-initial codes confirmed against
# the guqin tradition and the app's own internal naming.
LEFT_FINGER = {
    "d": "大指",   # dà zhǐ   (thumb)      -- 1st most common in this score
    "s": "食指",   # shí zhǐ  (index)
    "z": "中指",   # zhōng zhǐ (middle)
    "m": "名指",   # míng zhǐ (ring)       -- 2nd most common
    "g": "跪指",   # guì zhǐ  (kneel)      -- rare
}

# Right-hand plucking techniques (field d / y).  Confirmed by matching the
# decoded pitch against 正调 (F-tuning) string/hui math:
RIGHT_TECH = {
    "m":  "抹",    # mǒ   -- verified (note[1]: 抹7弦@7.6徽 = C5 = high-1)
    "t":  "挑",    # tiāo -- verified (note[7]: 挑5弦@X徽)
    "g":  "勾",    # gōu  -- verified (note[5]: 勾5弦@9徽 = E = mid-3)
    "q":  "剔",    # tī   -- right-hand middle finger outward (was wrongly 掐)
    "mt": "抹挑",  # combined (alternating)
    "tm": "挑抹",
    "tg": "挑勾",
    "gt": "勾挑",
    "p":  "劈",    # pī
    "tuo": "托",   # tuō
    "l":  "历",    # lì   (lide across strings)
    "bo": "拨",    # bō
    "c":  "刺",    # cì
    "duo": "打",   # dǎ
    "z":  "摘",    # zhāi
    "jia": "夹",   # jiā
    "li":  "历",   # lì (full pinyin variant; same as "l")
    "lj":  "连涓", # lián-juān
    "o":   "托",   # tuō (alt code; same as "tuo")
    "csuo": "长锁", # cháng-suǒ (long lock, repeated plucks)
    "gq":  "勾剔", # gōu + tī combined
    "b":   "擘",   # bò -- thumb inward (distinct from 劈 pī outward)
    "j":   "涓",   # juān
    "lun": "轮",   # lún (roll, repeated fingers)
    "gun": "滚",   # gǔn (rapid downward sweep)
    "qf":  "全扶", # quán-fú
    "fu":  "拂",   # fú
    "yl":  "圆搂", # yuán-lōu
    "y":   "罨",
    "d":   "打",
    "xy":  "虚罨",
    "bsuo": "背锁",
    "mg":  "抹勾",
    "suo": "锁",
    "blun": "半轮",
    "dsuo": "短锁",
    "dz":  "打摘",
    "yj":  "摘涓",
}

# Slide / articulation direction (field t on 'th' glyphs)
SLIDE_TYPE = {
    "d:":  "注下",    # down (zhù) - slide down to target
    "u:":  "绰上",    # up   (chuò) - slide up to target
    "hu:": "浒上",
    "ed:": "退复",    # tuì-fù - back & forth
    "eu:": "进复",
    "t:":  "退",      # tuì - retreat/back to target hui
    "j:":  "进",      # jìn - advance to target hui
    "td:": "淌",      # tāng - slow downward glide to target hui
    "yu:": "引上",    # yǐn-shàng - lead upward to target hui
    "zd:": "注下",    # zhù - press/slide down to target hui
}

# Compound markers on composite glyphs.  ``s:`` in a TYX glyph is the
# visible open-string (散音) marker, not an internal close delimiter.
COMPOUND_OPEN = {
    "pc:": "撮[",     # stopped string and open string sounded together
    "pp:": "泼[",
    "pl:": "剌[",
    "pfc:": "反撮[",
    "c:":  "",         # internal group start; do not render
}
COMPOUND_CLOSE = {
    "s:": "散",         # explicit open-string marker (e.g. 散勾五弦)
}

# Left-hand technique at a hui position (the "xxx:" tokens that appear in the
# c slot of a zht glyph).  Confirmed against the App's Dart technique enum.
LEFT_HUI_TECH = {
    "tq:": "滔起",     # tāo-qǐ  -- left-hand wave/raise at a hui
}

# Visible modifiers stored in the c/t slots of compact glyph types.
GLYPH_PREFIX = {
    "c:": "绰",
    "f:": "泛音",
    "sry:": "散如一",
}

# ``c:`` is overloaded.  In this exact ZHTYX shape it is an internal group
# marker: the App renders “名指十徽打二弦”, with no visible 绰.  The same raw
# shape occurs four times in the captured corpus.
ZHTYX_INTERNAL_C = {("m", "X", "c:", "d", "2")}

# Decoration / running technique tag (the ":xxx" tokens).
# These tags were confirmed by extracting the Dart AOT string pool from
# libapp.so -- the app literally interns these as its technique enum names.
DECOR_TAG = {
    # ---- base vibrato ----
    ":yin":  "吟",          # yín
    ":nao":  "猱",          # náo
    # ---- yin (vibrato) compounds ----
    ":yin:luo":    "吟猱",   # yín-náo combined
    ":yin:ding":   "吟定",   # yín then settle (定 dìng)
    ":yin:hua":    "吟滑",   # yín + slide (滑 huá)
    ":yin:ji":     "急吟",   # jí = fast/urgent
    ":yin:fei":    "飞吟",   # fēi = flying
    ":yin:sua":    "双吟",   # shuāng yín, verified against SWk1GWt2 rendering
    ":yin:xi":     "细吟",   # xì = fine/subtle
    ":yin:you":    "游吟",   # yóu = wandering
    ":yin:ca":     "吟搯",   # ca ~ chāi
    ":yin:zhuang": "撞吟",   # zhuàng
    ":yin:suax":   "吟锁",   # suǒ
    # ---- nao (larger vibrato) compounds ----
    ":nao:ca":   "长猱",     # cháng-náo (was wrongly 猱搯; ca = cháng)
    ":nao:da":   "猱打",     # dǎ
    ":nao:dang": "荡猱",     # dàng
    ":nao:fei":  "飞猱",
    ":nao:hua":  "猱滑",
    ":nao:ji":   "急猱",
    ":nao:sua":  "猱锁",
    ":nao:tui":  "猱退",     # tuì
    ":nao:zua":  "走猱",     # zǒu
    # ---- za compounds ----
    ":za":     "撞",
    ":za:fan": "撮泛",       # with harmonic
    ":za:hua": "撮滑",
    ":za:sua": "撮锁",
    # ---- harmonics ----
    ":fqi":  "泛起",         # fàn-qǐ  harmonics start
    ":fzhi": "泛止",         # fàn-zhǐ harmonics stop
    # ---- left-hand slides / articulations ----
    ":fang":   "放",         # fàng (放合)
    ":fu":     "复",         # fù (return)
    ":tui":    "退",         # tuì (back)
    ":tuifu":  "退复",       # tuì-fù
    ":jin":    "进",         # jìn (forward)
    ":jinfu":  "进复",       # jìn-fù
    ":dou":    "逗",         # dòu
    ":man":    "慢",         # màn (slow)
    ":huan":   "换",         # huàn
    ":ful":    "拂",         # fú
    ":laful":  "历拂",       # lì-fú
    ":tfu":    "拖拂",
    ":jfu":    "进复",       # jìn-fù (App's actual code for 进复; was wrongly 进拂)
    ":zz":     "再作",       # zài-zuò (was wrongly 撞; repeat once)
    # ---- right-hand / articulation tags seen on 'd' slot ----
    ":dq":     "带起",       # dài-qǐ (lift off)
    ":sx":     "少息",       # shǎo-xī (brief pause; NOT 至 -- 至 is zhi:)
    ":fenk":   "分开",       # fēn-kāi
    ":wl":     "往来",       # wǎng-lái (back and forth)
    ":zq":     "爪起",       # zhuǎ-qǐ (claw-lift)
    ":ts":     "同声",       # tóng-shēng (simultaneous)
    # ---- repetition / position markers ----
    ":bd":     "不动",       # bù-dòng (no movement)
    ":k":      "〔再作起点〕", # mark: repetition start point
    ":zkez":   "再作二声",   # from :k to here, repeat twice more
    ":zksz":   "再作三声",   # from :k to here, repeat three times more
    ":ztzz":   "从头再作",   # repeat once more from the beginning
    ":tcss":   "掐撮三声",   # qiā-cuō sān-shēng (a fixed ornament)
    ":tcsi":   "掐撮一声",   # same family: i = one rendition
    ":tces":   "掐撮二声",   # e = two renditions
    ":tples":  "掐拨剌二声", # tp+l(e/s/i) family, verified against SQquA21B rendering
    ":tplss":  "掐拨剌三声",
    ":tplsi":  "掐拨剌一声",
    ":tflss":  "掐拂歷三声", # verified against SyTIsW59 rendering
    ":tfles":  "掐拂歷二声", # same fl family, e = two renditions (SkrVrAJR)
    # ---- (data control tokens, render literally) ----
    ":d:newline": "↵",
    ":d:stop":    "停",
    ":d:t":       "·",
    ":d:null":    "",
    ":d:s":       "",
    ":d:space":   " ",
    ":d:l":       "",
    ":d:w":       "",
    ":d:wjly":    "",
    # ---- additional mappings verified against the rendered scores ----
    ":zkzz": "从ㄱ再作",
    ":yh":   "应合",
    ":tc":   "推出",
    ":fh":   "放合",
    ":ji":   "急",
    ":ry":   "如一",
    ":fc":   "反撮",
    ":dy":   "打圆",
    ":quz":  "曲终",
    ":sl":   "索铃",
    ":st":   "双弹",
}

# Jian type code -> chinese description
JIAN_TYPE = {
    "zhyx":  "左手指+徽位+右手指+弦",   # a b d e
    "zhtyx": "左手指+徽位+技巧+弦+音", # z h t y x
    "zht":   "左手指+徽位+滔起",       # a b c   (c == "tq:" 滔起)
    "th":    "徽位+滑音",              # h t
    "tyx":   "技巧+弦+音",             # c d e
    "yx":    "弦+音",                  # d e
    "tx":    "技巧+弦",                # c e
    "x":     "弦",                     # e
    "d":     "装饰记号",               # d (decor)
}

# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

DUR_PREFIX = {
    "": "四分",
    "q": "八分",
    "s": "十六分",
    "d": "三十二分",
}

NOTE_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
           "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
           "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)


def _tonic_pc(value: str) -> int:
    text = str(value).strip()
    text = text[0].upper() + text[1:] if text else text
    if text not in NOTE_PC:
        raise ValueError("unsupported tonic {!r}".format(value))
    return NOTE_PC[text]


def _pitch_name(pitch: int, octave: int, accidental: int = 0,
                ornament: bool = False) -> str:
    sign = "♯" * accidental if accidental > 0 else "♭" * (-accidental)
    dot = "̇" * octave if octave > 0 else "̣" * (-octave)
    return "{}{}{}{}".format(sign, pitch, dot, "（饰）" if ornament else "")


def transpose_storage_jianpu(note: dict, storage_tonic: str, display_tonic: str) -> None:
    """Relabel a C-relative stored note into the tonic shown by the App UI."""
    if note.get("pitch") not in range(1, 8):
        return
    source_midi = 60 + _tonic_pc(storage_tonic) + MAJOR_STEPS[note["pitch"] - 1]
    source_midi += 12 * int(note.get("octave", 0))
    display_base = 60 + _tonic_pc(display_tonic)
    candidates = []
    for degree, step in enumerate(MAJOR_STEPS, 1):
        for octave in range(-4, 5):
            accidental = source_midi - (display_base + step + 12 * octave)
            if -2 <= accidental <= 2:
                candidates.append((abs(accidental), abs(octave), degree, octave, accidental))
    if not candidates:
        return
    _cost, _oct_cost, degree, octave, accidental = min(candidates)
    note["storage_pitch"] = {
        "pitch": note["pitch"], "octave": note.get("octave", 0),
        "tonic": storage_tonic,
    }
    note["pitch"] = degree
    note["octave"] = octave
    note["accidental"] = accidental
    note["pitch_name"] = _pitch_name(degree, octave, accidental, note.get("ornament", False))


def shift_jianpu_octave(note: dict, shift: int) -> None:
    """Apply a verified score-display octave correction without losing provenance."""
    if not shift or note.get("pitch") not in range(1, 8):
        return
    note["storage_octave"] = note.get("octave", 0)
    note["octave"] = int(note.get("octave", 0)) + shift
    note["pitch_name"] = _pitch_name(
        note["pitch"], note["octave"], int(note.get("accidental", 0)),
        note.get("ornament", False),
    )


def decode_jianpu_token(token: str) -> dict:
    """Parse a raw jianpu token such as \"q1'\" into structured fields."""
    info = {"raw": token, "rest": False, "prolong": False}
    if token == "|":
        info["barline"] = True
        info["pitch"] = None
        info["pitch_name"] = "|"
        info["duration_name"] = "小节线"
        return info
    if token == "-":
        info["prolong"] = True
        info["pitch"] = None
        info["pitch_name"] = "－（延音）"
        return info
    grace = re.match(r"^y:([1-7])([',]*)(?:/(\d+))?'?$", token)
    if grace:
        pitch, octave_marks, divisor = grace.groups()
        octave = octave_marks.count("'") - octave_marks.count(",")
        info.update({
            "ornament": True,
            "duration_code": "y",
            "duration_name": "装饰音",
            "pitch": int(pitch),
            "octave": octave,
            "ornament_divisor": int(divisor) if divisor else None,
        })
        dot = "̇" * octave if octave > 0 else "̣" * (-octave)
        info["pitch_name"] = pitch + dot + "（饰）"
        return info
    m = re.match(r"^([qsd]?)([0-7])?('*)(,*)(.*)$", token)
    if not m:
        info["pitch_name"] = f"(?) {token}"
        return info
    dur, pitch, up, down, _ = m.groups()
    info["duration_code"] = dur
    info["duration_name"] = DUR_PREFIX.get(dur, f"未知({dur})")
    if pitch in (None, "", "0"):
        # Treat barlines ('|') and other non-pitch tokens as rests so the
        # sequence stays in sync with the source note list.
        info["rest"] = True
        info["pitch"] = 0
        info["pitch_name"] = ("|" if token == "|" else "0") + "（休止）"
        return info
    info["pitch"] = int(pitch)
    octave = len(up) - len(down)
    info["octave"] = octave
    # render pitch with octave dots  (high ' •, low •,)
    base = str(pitch)
    if octave > 0:
        info["pitch_name"] = base + "̇" * octave      # combining dot above
    elif octave < 0:
        info["pitch_name"] = base + "̣" * (-octave)    # combining dot below
    else:
        info["pitch_name"] = base
    return info


TONIC_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DEGREE_SEMITONES = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}

# Degree-1 anchoring must match the pitch auditor exactly (captured
# tonic_degree1_midi overrides, the empirical TONIC_DEGREE1_MIDI table where
# App's "1=B" is a B-flat do at octave 3, then the octave-4 fallback).
_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "_extractor_tonic_audit",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "audit_jianpu_jianzi_pitch.py"))
_AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT)


def _tonic_accidental(tonic: str) -> int:
    shift = 0
    for char in str(tonic)[1:]:
        if char in ("b", "♭"):
            shift -= 1
        elif char in ("#", "♯"):
            shift += 1
    return shift


def _abc_body_natural_midi(body: str) -> float:
    letter = body[0]
    value = 60 + TONIC_SEMITONES[letter.upper()]
    if letter.islower():
        value += 12
    # ABC 八度记号：撇号升八度、逗号降八度。
    value += 12 * body.count("'") - 12 * body.count(",")
    return float(value)


def _shift_body_octave(body: str, direction: int) -> str:
    letter = body.rstrip(",'")
    marks = body[len(letter):]
    if direction > 0:
        if letter.islower():
            return letter + marks + "'"
        if marks:
            return letter + marks[1:]
        return letter.lower()
    if letter.isupper():
        return letter + "," + marks
    if marks:
        return letter + marks[:-1]
    return letter.upper()


def jianpu_abc_pitch(jianpu: dict, tonic="C", octave_lowering=0,
                     degree1_midi: float | None = None):
    """Convert only the pitch portion of a numbered note to ABC.

    The per-note abc column is consumed downstream as ABSOLUTE pitch names
    without a key signature, so tonic accidentals (Bb, Eb, F#) are spelled
    explicitly and the octave is anchored to the auditor's degree-1 table
    (App's "1=B" is B-flat at octave 3, not B natural at octave 4).
    """
    pitch = jianpu.get("pitch")
    if pitch not in range(1, 8):
        return None
    letters = "CDEFGAB"
    tonic_letter = str(tonic)[0].upper()
    # Scale degrees advance diatonically from the tonic letter.
    diatonic_index = (
        letters.index(tonic_letter) + pitch - 1
        + 7 * (jianpu.get("octave", 0) - octave_lowering)
    )
    letter = letters[diatonic_index % 7]
    abc_octave = diatonic_index // 7
    if abc_octave > 0:
        body = letter.lower() + "'" * (abc_octave - 1)
    elif abc_octave < 0:
        body = letter + "," * (-abc_octave)
    else:
        body = letter
    if degree1_midi is None:
        degree1_midi = _AUDIT.parse_tonic_midi({"tonic": f"1={tonic}"})
    target_midi = (float(degree1_midi) + DEGREE_SEMITONES[pitch]
                   + (jianpu.get("accidental") or 0)
                   + 12 * (jianpu.get("octave", 0) - octave_lowering))
    diff = round(target_midi - _abc_body_natural_midi(body))
    while diff <= -6:
        body = _shift_body_octave(body, -1)
        diff = round(target_midi - _abc_body_natural_midi(body))
    while diff >= 6:
        body = _shift_body_octave(body, +1)
        diff = round(target_midi - _abc_body_natural_midi(body))
    if diff > 2 or diff < -2:
        return body
    return ("_" * -diff if diff < 0 else "^" * diff) + body


def jianpu_to_abc(jianpu: dict, tonic="C", octave_lowering=0,
                  previous_pitch=None, alt_jianpu=None,
                  degree1_midi: float | None = None) -> tuple[str, str | None]:
    """Convert one numbered-notation event, including stacked pitches, to ABC."""
    if jianpu.get("barline"):
        return "|", previous_pitch
    if jianpu.get("ornament"):
        ornament = jianpu_abc_pitch(jianpu, tonic, octave_lowering, degree1_midi)
        return ("{" + ornament + "}" if ornament else "{}"), previous_pitch
    duration = {"": "4", "q": "2", "s": "", "d": "/2"}.get(
        jianpu.get("duration_code", ""), "4"
    )
    if jianpu.get("prolong"):
        return (("-" + previous_pitch + "4") if previous_pitch else "z4",
                previous_pitch)
    if jianpu.get("rest"):
        return "z" + duration, previous_pitch

    primary = jianpu_abc_pitch(jianpu, tonic, octave_lowering, degree1_midi)
    if not primary:
        return "z" + duration, previous_pitch
    alternate = (
        jianpu_abc_pitch(alt_jianpu, tonic, octave_lowering, degree1_midi)
        if alt_jianpu else None
    )
    if alternate and alternate != primary:
        chord = "[{}{}]".format(primary, alternate)
        return chord + duration, chord
    return primary + duration, primary


def interpret_tuning(tuning: dict) -> dict:
    """Interpret Sitongli's seven string offsets without altering notation."""
    raw_values = (tuning or {}).get("value")
    if not isinstance(raw_values, list) or len(raw_values) < 7:
        return {
            "name": (tuning or {}).get("name"),
            "known": False,
            "string_semitone_offsets": None,
            "octave_lowering": None,
            "open_strings": [],
        }
    values = list(raw_values)
    string_offsets = values[:7]
    # The eighth App value is a register/control code, not a count of octaves
    # to apply to numbered notation.  Treating values such as -7 or +7 as
    # octaves produced impossible ABC output (c'''' / C,,,,,,,).
    register_code = values[7] if len(values) >= 8 else None
    base_midi = [48, 50, 53, 55, 57, 60, 62]  # C3 D3 F3 G3 A3 C4 D4
    sharp_names = ["C", "C♯", "D", "D♯", "E", "F",
                   "F♯", "G", "G♯", "A", "B♭", "B"]
    open_strings = []
    for index, (midi, offset) in enumerate(zip(base_midi, string_offsets), 1):
        sounding_midi = midi + int(offset)
        open_strings.append({
            "string": index,
            "semitone_offset": offset,
            "pitch": sharp_names[sounding_midi % 12],
            "octave": sounding_midi // 12 - 1,
        })
    return {
        "name": (tuning or {}).get("name"),
        "known": True,
        "string_semitone_offsets": string_offsets,
        "octave_lowering": 0,
        "register_code": register_code,
        "open_strings": open_strings,
    }


def resolve_tuning(data: dict) -> dict:
    """Reject legacy runtime metadata whose all-zero tuning was fabricated."""
    tuning = dict((data.get("metadata", {}) or {}).get("tuning") or {})
    provenance = data.get("provenance", {}) or {}
    legacy_runtime_capture = (
        provenance.get("method")
        == "Frida runtime object capture from authorized logged-in Android app"
        and "source" not in tuning
    )
    if legacy_runtime_capture:
        return {
            "name": None,
            "value": None,
            "source": "legacy_hardcoded_value_rejected",
            "legacy_claim": tuning,
        }
    return tuning


def _parts(std: dict) -> dict:
    """Strip the std dict down to its meaningful letter slots."""
    out = {}
    for k, v in std.items():
        if k in ("tp", "raw_type", "raw_fields"):
            continue
        if v not in (None, ""):
            out[k] = v
    return out


def format_hui(value) -> str:
    """Render compact hui positions such as 79, X8 and X23."""
    text = str(value or "")
    if not text:
        return ""
    if text == "Y":
        return "徽外"

    cn_digits = "〇一二三四五六七八九"

    def number_cn(number: int) -> str:
        if number < 10:
            return cn_digits[number]
        if number == 10:
            return "十"
        if number < 20:
            return "十" + cn_digits[number - 10]
        return str(number)

    hui = None
    fraction = None
    if text.startswith("X"):
        suffix = text[1:]
        if not suffix:
            hui = 10
        elif suffix.isdigit() and len(suffix) >= 2:
            hui = 10 + int(suffix[0])
            fraction = int(suffix[1:])
        elif suffix.isdigit() and int(suffix) <= 3:
            # X1/X2/X3 are the exact 11th/12th/13th hui.
            hui = 10 + int(suffix)
        elif suffix.isdigit():
            # X8, for example, is the 10th hui plus eight fen.
            hui = 10
            fraction = int(suffix)
    elif text.isdigit():
        if len(text) == 1:
            hui = int(text)
        else:
            hui = int(text[:-1])
            fraction = int(text[-1])

    if hui is None:
        return text
    label = number_cn(hui) + "徽"
    if fraction:
        label += number_cn(fraction) + "分"
    return label


def hui_label(value) -> str:
    return format_hui(value)


def decode_jianzi_glyph(jian: dict) -> dict:
    """Decode one jianzi glyph into a structured + readable form."""
    std = jian.get("std", {}) or {}
    tp = std.get("tp")
    p = _parts(std)
    decoded = {"tp": tp, "type_name": JIAN_TYPE.get(tp, "(复合)"),
               "raw_fields": p, "readable": "", "cn": ""}

    if tp == "zhyx":               # a 左手指  b 徽位  d 右手指  e 弦
        lf = LEFT_FINGER.get(p.get("a", ""), p.get("a", ""))
        hui = hui_label(p.get("b", ""))
        rt = RIGHT_TECH.get(p.get("d", ""), p.get("d", ""))
        xian = p.get("e", "")
        decoded["readable"] = f"{lf} {hui} {rt}第{xian}弦"
        decoded["cn"] = f"{lf}{hui}{rt}{xian}弦"

    elif tp == "zhtyx":            # z 左手指 h 徽位 t 分组 y 弦技 x 弦
        lf = LEFT_FINGER.get(p.get("z", ""), p.get("z", ""))
        hui = hui_label(p.get("h", ""))
        # In this glyph type c: is a visible 绰 prefix. Elsewhere c: can
        # still be an internal compound delimiter.
        shape = (p.get("z", ""), p.get("h", ""), p.get("t", ""),
                 p.get("y", ""), p.get("x", ""))
        grp = "" if shape in ZHTYX_INTERNAL_C else GLYPH_PREFIX.get(
            p.get("t", ""), COMPOUND_OPEN.get(p.get("t", ""), "")
        )
        rt = RIGHT_TECH.get(p.get("y", ""), p.get("y", ""))
        xian = p.get("x", "")
        grp_part = (grp + " ") if grp else ""
        decoded["readable"] = f"{grp_part}{lf} {hui} {rt}第{xian}弦".strip()
        decoded["cn"] = f"{grp}{lf}{hui}{rt}{xian}弦"

    elif tp == "zht":              # a 左手指  b 徽位  c 滔起 (tq:)
        lf = LEFT_FINGER.get(p.get("a", ""), p.get("a", ""))
        hui = hui_label(p.get("b", ""))
        tech = LEFT_HUI_TECH.get(p.get("c", ""), p.get("c", ""))
        decoded["readable"] = f"{lf} {hui} {tech}".strip()
        decoded["cn"] = f"{lf}{hui}{tech}"

    elif tp == "th":               # h 徽位  t 滑音方向
        hui = hui_label(p.get("h", ""))
        sl = SLIDE_TYPE.get(p.get("t", ""), p.get("t", ""))
        decoded["readable"] = f"{sl} 至 {hui}"
        decoded["cn"] = f"{sl}{hui}"

    elif tp == "tyx":              # c 分组(s:止/jiu:就) d 弦技 e 弦
        grp = COMPOUND_CLOSE.get(p.get("c", ""), "")   # s: -> 止
        rt = RIGHT_TECH.get(p.get("d", ""), p.get("d", ""))
        xian = p.get("e", "")
        # ``就`` and ``散`` are both front modifiers ("就剔4弦" and
        # "散勾5弦"), rather than trailing compound delimiters.
        lead = ""
        if p.get("c") == "jiu:":
            lead = "就"
            grp = ""
        elif p.get("c") == "s:":
            lead = "散"
            grp = ""
        grp_part = (" " + grp) if grp else ""
        lead_part = (lead + " ") if lead else ""
        decoded["readable"] = f"{lead_part}{rt}第{xian}弦{grp_part}".strip()
        decoded["cn"] = f"{lead}{rt}{xian}弦{grp}"

    elif tp in ("yx", "tx", "x"):
        rt = RIGHT_TECH.get(p.get("d", ""), p.get("d", ""))
        xian = p.get("e", "")
        # c="zhi:" = "至" is a front modifier on tx glyphs ("至5弦" = slide to
        # string 5), rendered before the string number.
        lead = (
            "至" if p.get("c") == "zhi:"
            else GLYPH_PREFIX.get(p.get("c", ""), "")
        )
        lead_part = (lead + " ") if lead else ""
        decoded["readable"] = f"{lead_part}{rt}第{xian}弦".strip()
        decoded["cn"] = f"{lead}{rt}{xian}弦"

    elif tp == "d":                # d 装饰
        tag = p.get("d", "")
        cn = DECOR_TAG.get(tag, tag)
        decoded["readable"] = cn
        decoded["cn"] = cn

    else:                          # None / untyped complex glyph
        # Compound techniques (a slot): pluck a stopped string together with
        # an open string (撮/泼/剌), or sound two strings at once (弹/双弹/三弹).
        # Layout: b=left finger / c=hui / d=stopped string / f=2nd left finger
        # / g=open string.
        close_tag = COMPOUND_CLOSE.get(p.get("h", ""), "")
        lf = LEFT_FINGER.get(p.get("b", ""), p.get("b", ""))
        lf2 = LEFT_FINGER.get(p.get("f", ""), p.get("f", ""))
        hui = hui_label(p.get("c", ""))
        stopped_string = p.get("d", "")
        open_string = p.get("g", "")
        compound_tech = {
            "pc:": "撮", "pp:": "泼", "pl:": "剌",
            "pt:": "弹", "pet:": "双弹", "pst:": "三弹",
            "ppl:": "泼剌", "pfc:": "反撮",
        }.get(p.get("a"))
        if compound_tech:
            # Build the two sounding notes. The stopped note needs a left
            # finger + hui; if those are absent both strings are open.
            if lf and hui and stopped_string:
                first = f"{lf} {hui} {stopped_string}弦按音"
                first_cn = f"{lf}{hui}{stopped_string}弦按音"
            elif stopped_string:
                first = f"{stopped_string}弦散音"
                first_cn = f"{stopped_string}弦散音"
            else:
                first = first_cn = ""
            # Second note: open string, or a second stopped note (pt: family
            # uses f as a second left finger at the same/own hui).
            if open_string and lf2:
                hui2 = hui_label(p.get("c", "")) if p.get("c") else ""
                second = f"{lf2} {hui2} {open_string}弦按音".strip()
                second_cn = f"{lf2}{hui2}{open_string}弦按音"
            elif open_string:
                second = f"{open_string}弦散音"
                second_cn = f"{open_string}弦散音"
            else:
                second = second_cn = ""
            if first and second:
                if p.get("a") == "pfc:":
                    # 反撮 has its own bracketed notation.  Unlike 撮/泼/剌,
                    # the stopped-string half is conventionally left without
                    # the explanatory “按音” suffix.
                    stopped_cn = f"{lf}{hui}{stopped_string}弦" if lf and hui and stopped_string else first_cn
                    decoded["readable"] = f"{compound_tech}[{stopped_cn}+{second_cn}]"
                    decoded["cn"] = f"{compound_tech}[{stopped_cn}+{second_cn}]"
                else:
                    decoded["readable"] = f"{compound_tech}（{first} + {second}）"
                    decoded["cn"] = f"{compound_tech}（{first_cn}＋{second_cn}）"
            elif first or second:
                solo = first or second
                decoded["readable"] = f"{compound_tech}（{solo}）"
                decoded["cn"] = f"{compound_tech}（{first_cn or second_cn}）"
            else:
                decoded["readable"] = compound_tech
                decoded["cn"] = compound_tech
        else:
            # Generic fallback: render slots using every table we have.
            bits = []
            for slot, val in p.items():
                if val in LEFT_FINGER:
                    bits.append(LEFT_FINGER[val])
                elif val in RIGHT_TECH:
                    bits.append(RIGHT_TECH[val])
                elif val in SLIDE_TYPE:
                    bits.append(SLIDE_TYPE[val])
                elif val in DECOR_TAG:
                    bits.append(DECOR_TAG[val])
                elif val in COMPOUND_OPEN:
                    bits.append(COMPOUND_OPEN[val])
                elif val in COMPOUND_CLOSE:
                    bits.append(COMPOUND_CLOSE[val])
                else:
                    bits.append(str(val))
            decoded["readable"] = " ".join(bits)
            decoded["cn"] = "".join(bits)
    return decoded


def decode_note(note: dict) -> dict:
    jianpu = decode_jianpu_token(note.get("note", ""))
    alt_jianpu = None
    alt_pitch = (note.get("note_decoded") or {}).get("alt_pitch")
    if isinstance(alt_pitch, dict) and not jianpu.get("rest"):
        alt_digit = alt_pitch.get("off_8")
        try:
            alt_octave = int(alt_pitch.get("off_c", 0))
            if alt_octave >= 2**63:
                alt_octave -= 2**64
        except (TypeError, ValueError):
            alt_octave = 0
        if str(alt_digit) in "1234567":
            suffix = "'" * alt_octave if alt_octave > 0 else "," * (-alt_octave)
            alt_token = "{}{}{}".format(
                jianpu.get("duration_code", ""), alt_digit, suffix
            )
            candidate = decode_jianpu_token(alt_token)
            if (candidate.get("pitch"), candidate.get("octave", 0)) != (
                    jianpu.get("pitch"), jianpu.get("octave", 0)):
                alt_jianpu = candidate
    jians = [decode_jianzi_glyph(j) for j in (note.get("jians") or [])]
    slurs = note.get("note_decoded", {}).get("slurs", []) or []
    return {
        "index": note.get("index"),
        "jianpu_raw": note.get("note"),
        "jianpu": jianpu,
        "jianpu_alt": alt_jianpu,
        "jianzi": jians,
        "lyric": note.get("lyric", "") or note.get("lyric1", "")
                 or note.get("lyric2", ""),
        "slurs": slurs,
    }


CHINESE_SECTION_NUMBERS = (
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
)


def extract_sections(notes: list[dict]) -> list[dict]:
    """Normalize the per-note App section markers into stable boundaries.

    Runtime note events keep the App's section object in ``section``.  Its
    ``off_8`` is the first note index and ``off_10`` is the optional title.
    A one-item default section carries no useful split signal, so it is not
    emitted as an inline marker.
    """
    found = []
    seen = set()
    for fallback_index, note in enumerate(notes):
        source = note.get("section")
        if not isinstance(source, dict):
            continue
        raw_start = source.get("off_8")
        if raw_start in (None, ""):
            continue
        try:
            start = int(raw_start)
        except (TypeError, ValueError):
            start = fallback_index
        if start < 0 or start >= len(notes) or start in seen:
            continue
        seen.add(start)
        found.append({
            "start_index": start,
            "title": str(source.get("off_10") or "").strip(),
            "source_extra": source.get("off_18"),
        })
    found.sort(key=lambda section: section["start_index"])
    if len(found) < 2:
        return []
    for number, section in enumerate(found, 1):
        label = (
            CHINESE_SECTION_NUMBERS[number - 1]
            if number <= len(CHINESE_SECTION_NUMBERS) else str(number)
        )
        section["number"] = number
        section["label"] = label
        section["marker"] = "<{}{}>".format(
            label, " " + section["title"] if section["title"] else ""
        )
    return found


def tokens_with_section_markers(tokens: list[str], sections: list[dict], *, abc=False) -> str:
    """Render non-sounding section boundaries before their first note."""
    starts = {section["start_index"]: section for section in sections}
    lines, current = [], []
    for index, token in enumerate(tokens):
        section = starts.get(index)
        if section:
            if current:
                lines.append(" ".join(current))
                current = []
            # A percent-prefixed line is a standard ABC comment and is silent.
            lines.append("% " + section["marker"] if abc else section["marker"])
        current.append(token)
        if len(current) == 16:
            lines.append(" ".join(current))
            current = []
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and map Jianpu and Jianzipu from raw_data.json."
    )
    parser.add_argument(
        "--raw",
        default=os.path.join(ROOT, "out", "raw_data.json"),
        help="Path to the input raw_data.json (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(ROOT, "extracted_data"),
        help="Directory for generated JSON and Markdown files (default: %(default)s)",
    )
    parser.add_argument(
        "--tonic",
        help=(
            "Numbered-notation tonic, such as C, F, Bb or F#. "
            "Overrides metadata.tonic in raw_data.json."
        ),
    )
    parser.add_argument("--tuning-name", help="Verified tuning-name override.")
    parser.add_argument(
        "--tuning-values",
        type=lambda value: [int(item) for item in value.split(",")],
        help="Eight verified tuning offsets, comma-separated.",
    )
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    raw_path = os.path.abspath(args.raw)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    raw = data["raw_score_data"]
    notes = raw["notes"]

    decoded = [decode_note(n) for n in notes]
    sections = extract_sections(notes)
    section_by_start = {section["start_index"]: section for section in sections}
    active_section = None
    for item in decoded:
        active_section = section_by_start.get(item["index"], active_section)
        if active_section:
            item["section"] = {
                "number": active_section["number"],
                "label": active_section["label"],
                "title": active_section["title"],
                "start": item["index"] == active_section["start_index"],
                "marker": active_section["marker"],
            }
    if args.tuning_values is not None:
        if len(args.tuning_values) != 8:
            raise SystemExit("--tuning-values must contain exactly eight integers")
        tuning = {
            "name": args.tuning_name,
            "value": args.tuning_values,
            "source": "verified_override",
        }
    else:
        tuning = resolve_tuning(data)
    tuning_info = interpret_tuning(tuning)
    tuning_name = tuning_info["name"] or "未知"
    abc_tonic = args.tonic or meta.get("tonic")
    if not abc_tonic:
        raise SystemExit(
            "Missing numbered-notation tonic: add metadata.tonic to raw_data.json "
            "or pass --tonic (for example --tonic C). "
            "The guqin tuning name cannot determine 1=."
        )
    if not re.fullmatch(r"[A-Ga-g](?:#|b)?", str(abc_tonic)):
        raise SystemExit(
            "Invalid tonic {!r}; expected A-G with an optional # or b.".format(
                abc_tonic
            )
        )
    abc_tonic = str(abc_tonic)[0].upper() + str(abc_tonic)[1:]
    storage_tonic = meta.get("jianpu_storage_tonic")
    if storage_tonic:
        for item in decoded:
            transpose_storage_jianpu(item["jianpu"], storage_tonic, abc_tonic)
            if item["jianpu_alt"]:
                transpose_storage_jianpu(item["jianpu_alt"], storage_tonic, abc_tonic)
    octave_shift = int(meta.get("jianpu_octave_shift", 0) or 0)
    if octave_shift:
        for item in decoded:
            shift_jianpu_octave(item["jianpu"], octave_shift)
            if item["jianpu_alt"]:
                shift_jianpu_octave(item["jianpu_alt"], octave_shift)
    previous_abc_pitch = None
    for item in decoded:
        # The numbered score token already carries its own octave marks.  App
        # tuning describes string pitch, not a transposition of the notation.
        abc, previous_abc_pitch = jianpu_to_abc(
            item["jianpu"], abc_tonic, 0,
            previous_abc_pitch, item["jianpu_alt"],
            degree1_midi=_AUDIT.parse_tonic_midi(meta),
        )
        item["abc"] = abc

    score_id = meta.get("score_id")
    score_key = meta.get("score_key")
    score_url = (
        "https://app.sitongli.net/scores/{}?shareid={}".format(score_key, score_id)
        if score_key and score_id is not None else None
    )

    # ---- structured json ------------------------------------------------
    out_json = os.path.join(out_dir, "jianpu_jianzi_mapped.json")
    result = {
        "metadata": {
            "score_id": score_id,
            "score_title": meta.get("score_title"),
            "score_key": score_key,
            "url": score_url,
            "meter": meta.get("meter"),
            "tonic": abc_tonic,
            **({"jianpu_storage_tonic": meta["jianpu_storage_tonic"],
                "jianpu_storage_note": meta.get("jianpu_storage_note", "")}
               if meta.get("jianpu_storage_tonic") else {}),
            **({"jianpu_octave_shift": int(meta["jianpu_octave_shift"]),
                "jianpu_octave_note": meta.get("jianpu_octave_note", "")}
               if meta.get("jianpu_octave_shift") else {}),
            **({"tonic_degree1_midi": meta["tonic_degree1_midi"]}
               if "tonic_degree1_midi" in meta else {}),
            "tuning": tuning,
            "tempo": (meta.get("sections") or [{}])[0].get("tempo"),
            "sections": sections,
            "notes_count": len(notes),
            "abc_tonic": abc_tonic,
            "tuning_interpretation": tuning_info,
        },
        "notes": decoded,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---- human readable markdown ---------------------------------------
    out_md = os.path.join(out_dir, "jianpu_jianzi_mapped.md")
    title = meta.get("score_title", "")
    tuning = tuning or {}
    tempo = (meta.get("sections") or [{}])[0].get("tempo")

    def _md_escape(s):
        # escape pipe (table cell separator) and backslash; keep everything else
        return str(s).replace("\\", "\\\\").replace("|", "\\|")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 《{}》 简谱 ⇆ 减字谱 映射序列\n\n".format(title))

        # metadata block
        f.write("| 字段 | 值 |\n")
        f.write("|---|---|\n")
        f.write("| score_id | {} |\n".format(score_id))
        f.write("| score_key | {} |\n".format(score_key))
        f.write("| url | {} |\n".format(score_url or ""))
        f.write("| 拍号 meter | {} |\n".format(meta.get("meter")))
        f.write("| 简谱调号 tonic | 1={} |\n".format(abc_tonic))
        if meta.get("jianpu_storage_tonic"):
            f.write("| 简谱存储转调 | 内部 1={} → 谱面 1={} |\n".format(
                meta["jianpu_storage_tonic"], abc_tonic
            ))
        if meta.get("jianpu_octave_shift"):
            f.write("| 简谱八度修正 | {:+d}（内部值 → 谱面显示） |\n".format(
                int(meta["jianpu_octave_shift"])
            ))
        if tuning_info["known"]:
            source_note = (
                "（根据候选调弦名称推定）"
                if tuning.get("source") == "candidate_tuning_name_inferred"
                else ""
            )
            tuning_display = "{}（运行时半音偏移 {}）".format(
                tuning.get("name") or "未命名", tuning.get("value")
            ) + source_note
        else:
            tuning_display = "未知（未取得实际调弦值）"
        f.write("| 调弦 tuning | {} |\n".format(tuning_display))
        f.write("| 速度 tempo | {} |\n".format(tempo))
        f.write("| 音符数 notes | {} |\n".format(len(notes)))
        f.write("\n")

        if tuning_info["known"]:
            open_string_text = "，".join(
                "{}弦={}{}（{:+}半音）".format(
                    item["string"], item["pitch"], item["octave"],
                    item["semitone_offset"]
                )
                for item in tuning_info["open_strings"]
            )
            register_text = (
                "App 寄存器值 {} 不参与简谱/ABC 的八度换算。".format(
                    tuning_info.get("register_code")
                )
                if tuning_info.get("register_code") is not None
                else "该推定仅包含七条弦的半音偏移，不臆造 App 寄存器值。"
            )
            f.write("调弦解释：{}；{}\n\n".format(
                open_string_text, register_text
            ))
        else:
            f.write("调弦解释：未知（本次采集未取得实际调弦值）。\n\n")

        # ---- ABC notation ----------------------------------------------
        f.write("## ABC notation\n\n")
        f.write("调性转换：采用谱面首调简谱 `1={}`；调弦 `{}` 仅用于解释琴弦音高，"
                "四分音符为默认一拍。".format(abc_tonic, tuning_name))
        f.write(
                "简谱延音线 `-` 写为前一音的同音连结。\n\n")
        f.write("```abc\n")
        f.write("X:1\n")
        f.write("T:{}\n".format(title))
        f.write("M:{}\n".format(meta.get("meter") or "1/4"))
        f.write("L:1/16\n")
        if tempo:
            f.write("Q:1/4={}\n".format(tempo))
        f.write("K:{}\n".format(abc_tonic))
        abc_tokens = [d["abc"] for d in decoded]
        f.write(tokens_with_section_markers(abc_tokens, sections, abc=True) + "\n")
        f.write("```\n\n")

        # ---- full jianzipu score ----
        f.write("## 减字谱（完整）\n\n")
        f.write("每个音的减字谱用方括号 `[...]` 包裹（同一音的多个减字组分连写），"
                "音与音之间以空格分隔，逐音与上方 ABC 谱对应。\n\n")
        f.write("```\n")
        jianzi_tokens = []
        for d in decoded:
            if d["jianzi"]:
                cn = "".join(g["cn"] for g in d["jianzi"])
                jianzi_tokens.append("[{}]".format(cn))
            else:
                jianzi_tokens.append("[—]")
        f.write(tokens_with_section_markers(jianzi_tokens, sections) + "\n")
        f.write("```\n\n")

        # ---- main score table ----
        f.write("## 总谱\n\n")
        f.write("| # | 简谱 | ABC | 时值 | 歌词 | 减字谱 |\n")
        f.write("|---:|:---:|:---:|:---:|:---:|:---|\n")
        for d in decoded:
            if d.get("section", {}).get("start"):
                f.write("|  | {} |  |  |  |  |\n".format(
                    _md_escape(d["section"]["marker"])
                ))
            main_pitch = _md_escape(d["jianpu"]["pitch_name"])
            if d["jianpu_alt"]:
                # The alternate pitch is the upper number in the app's stack.
                jianpu = "{}<br>{}".format(
                    _md_escape(d["jianpu_alt"]["pitch_name"]), main_pitch
                )
            else:
                jianpu = main_pitch
            abc = _md_escape(d["abc"])
            dur = _md_escape(d["jianpu"].get("duration_name", ""))
            lyric = _md_escape(d["lyric"] or "")
            # multiple glyphs -> separate with line break for readability
            if d["jianzi"]:
                glyphs = "<br>".join(_md_escape(g["cn"]) for g in d["jianzi"])
            else:
                glyphs = "—"
            f.write("| {} | {} | {} | {} | {} | {} |\n".format(
                d["index"], jianpu, abc, dur, lyric, glyphs))

    # ---- markdown-mirrored json (same info as the .md above) ----------
    out_json_readable = os.path.join(out_dir, "jianpu_jianzi_readable.json")
    abc_string = tokens_with_section_markers(abc_tokens, sections, abc=True)
    jianzipu_string = tokens_with_section_markers(jianzi_tokens, sections)
    notes_mapped = [
        {
            "index": d["index"],
            "jianpu": d["jianpu"].get("pitch_name", ""),
            "jianpu_alt": (d["jianpu_alt"] or {}).get("pitch_name"),
            "abc": d["abc"],
            "duration": d["jianpu"].get("duration_name", ""),
            "lyric": d["lyric"] or "",
            "jianzi": "".join(g["cn"] for g in d["jianzi"]) if d["jianzi"] else "",
            **({"section": d["section"]} if d.get("section") else {}),
        }
        for d in decoded
    ]
    readable = {
        "metadata": {
            "score_title": title,
            "score_id": score_id,
            "score_key": score_key,
            "meter": meta.get("meter"),
            "tonic": "1={}".format(abc_tonic),
            **({"jianpu_storage_tonic": meta["jianpu_storage_tonic"],
                "jianpu_storage_note": meta.get("jianpu_storage_note", "")}
               if meta.get("jianpu_storage_tonic") else {}),
            **({"jianpu_octave_shift": int(meta["jianpu_octave_shift"]),
                "jianpu_octave_note": meta.get("jianpu_octave_note", "")}
               if meta.get("jianpu_octave_shift") else {}),
            **({"tonic_degree1_midi": meta["tonic_degree1_midi"]}
               if "tonic_degree1_midi" in meta else {}),
            "tempo": tempo,
            "sections": sections,
            "tuning": {
                "name": tuning.get("name"),
                "value": tuning.get("value"),
                "open_strings": [
                    {
                        "string": item["string"],
                        "pitch": item["pitch"],
                        "octave": item["octave"],
                        "semitone_offset": item["semitone_offset"],
                    }
                    for item in tuning_info["open_strings"]
                ],
                "octave_lowering": tuning_info["octave_lowering"],
            },
        },
        "abc": abc_string,
        "jianzipu": jianzipu_string,
        "notes": notes_mapped,
    }
    with open(out_json_readable, "w", encoding="utf-8") as f:
        json.dump(readable, f, ensure_ascii=False, indent=2)

    # ---- summary on stdout ---------------------------------------------
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json_readable}")
    print()
    print(f"曲名: 《{meta.get('score_title')}》  共 {len(notes)} 个音")
    print(f"减字谱字形总数: {sum(len(d['jianzi']) for d in decoded)}")
    # glyph kind distribution
    kind = Counter(g["tp"] for d in decoded for g in d["jianzi"])
    print("减字类型分布:")
    for tp, c in kind.most_common():
        print(f"    {tp!s:<8} {JIAN_TYPE.get(tp,'(复合)'):<22} x{c}")

    # quick sanity: first 8 notes
    print("\n前8个音 (简谱 / 歌词 / 减字谱):")
    for d in decoded[:8]:
        glyphs = "  ".join(g["cn"] for g in d["jianzi"]) or "—"
        idx = d['index'] if d['index'] is not None else "-"
        pitch = d['jianpu']['pitch_name'] or "-"
        print(f"  [{idx:>3}] {pitch:<4} "
              f"{d['lyric']:<4}  {glyphs}")


if __name__ == "__main__":
    main()
