#!/usr/bin/env python3
"""Deterministic guqin position/pitch mapping with JSON output."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys

TUNING_PROFILES = {
    # Modern international pitch naming used by docs/skill_verify.md.
    "modern": [36, 38, 41, 43, 45, 48, 50],  # C2 D2 F2 G2 A2 C3 D3
    # Octave convention recovered from the Sitongli runtime/export pipeline.
    "sitongli": [48, 50, 53, 55, 57, 60, 62],  # C3 D3 F3 G3 A3 C4 D4
}
HUI_COORDS = [1/8, 1/6, 1/5, 1/4, 1/3, 2/5, 1/2,
              3/5, 2/3, 3/4, 4/5, 5/6, 7/8]
HARMONIC_SEMITONES = [36, 31, 28, 24, 19, 28, 12,
                      28, 19, 24, 28, 31, 36]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
NOTE_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3,
           "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8,
           "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
NAMES = ["C", "C♯", "D", "E♭", "E", "F",
         "F♯", "G", "A♭", "A", "B♭", "B"]


def emit(value, code=0):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(code)


def parse_csv_numbers(text, count=7):
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != count:
        raise ValueError(f"expected {count} comma-separated numbers")
    return values


def tuning(args):
    if args.open_midi:
        opens = parse_csv_numbers(args.open_midi)
        source = "explicit_open_midi"
    else:
        offsets = parse_csv_numbers(args.offsets) if args.offsets else [0] * 7
        base = TUNING_PROFILES[args.profile]
        opens = [m + o for m, o in zip(base, offsets)]
        source = f"{args.profile}_zhengdiao_plus_offsets"
    return opens, source


def parse_hui(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"", "OPEN", "SAN", "散", "0"}:
        return None
    if text in {"Y", "徽外"}:
        raise ValueError("hui_outside_has_no_fixed_pitch")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        # Sitongli compact form: 76=7.6, but ordinary 13 remains 13.
        if "." not in text and len(text) == 2 and int(text) > 13:
            number = int(text[0]) + int(text[1]) / 10
        return number
    if text == "X":
        return 10.0
    m = re.fullmatch(r"X(\d+)", text)
    if m:
        suffix = m.group(1)
        if suffix in {"1", "2", "3"}:
            return 10 + int(suffix)
        if len(suffix) == 1:
            return 10 + int(suffix) / 10
        return 10 + int(suffix[0]) + int(suffix[1:]) / (10 ** len(suffix[1:]))
    raise ValueError("unrecognized_hui")


def hui_coordinate(hui):
    if not 1 <= hui <= 13:
        raise ValueError("hui_out_of_range")
    lo = int(math.floor(hui))
    if hui == lo:
        return HUI_COORDS[lo - 1]
    hi = lo + 1
    frac = hui - lo
    return HUI_COORDS[lo - 1] + frac * (HUI_COORDS[hi - 1] - HUI_COORDS[lo - 1])


def pitch_obj(value):
    nearest = round(value)
    return {
        "midi": value,
        "nearest_midi": nearest,
        "name": f"{NAMES[nearest % 12]}{nearest // 12 - 1}",
        "cents_error": round((value - nearest) * 100, 3),
    }


def position_result(args):
    opens, tuning_source = tuning(args)
    if not 1 <= args.string <= 7:
        return failure("invalid", "string_out_of_range")
    try:
        hui = parse_hui(args.hui)
    except ValueError as exc:
        return failure("indeterminate" if "outside" in str(exc) else "invalid", str(exc))
    mode = args.mode
    if mode in {"open", "san"} or hui is None:
        value = opens[args.string - 1]
        return success("exact", value, opens, tuning_source, args.string, None, "open")
    if mode == "harmonic" and hui != int(hui):
        return failure("invalid", "fractional_hui_not_valid_for_harmonic")
    try:
        coord = hui_coordinate(hui)
    except ValueError as exc:
        return failure("invalid", str(exc))
    if mode == "harmonic":
        value = opens[args.string - 1] + HARMONIC_SEMITONES[int(hui) - 1]
        formula = "open_midi + harmonic_semitone_table[hui-1]"
    else:
        value = opens[args.string - 1] + 12 * math.log2(1 / coord)
        formula = "open_midi + 12*log2(1/coordinate)"
    result = success("exact" if abs(value - round(value)) < 0.06 else "approximate",
                     value, opens, tuning_source, args.string, hui, mode)
    result["evidence"]["coordinate_from_bridge"] = coord
    result["evidence"]["formula"] = formula
    return result


def success(status, value, opens, source, string, hui, mode):
    return {
        "status": status,
        "pitches": [pitch_obj(value)],
        "evidence": {"open_midi": opens, "tuning_source": source,
                     "string": string, "hui": hui, "mode": mode},
        "diagnostics": [],
    }


def failure(status, diagnostic):
    return {"status": status, "pitches": [], "evidence": {},
            "diagnostics": [{"code": diagnostic}]}


def target_midi(args):
    if getattr(args, "midi", None) is not None:
        return float(args.midi), {"kind": "midi"}
    if getattr(args, "jianpu", None) is None:
        raise ValueError("target_pitch_required")
    degree = int(args.jianpu)
    if not 1 <= degree <= 7:
        raise ValueError("jianpu_degree_out_of_range")
    if args.tonic_midi is not None:
        tonic = float(args.tonic_midi)
    else:
        key = args.tonic.upper().replace("♯", "#").replace("♭", "B")
        if key not in NOTE_PC:
            raise ValueError("unrecognized_tonic")
        tonic = 48 + NOTE_PC[key]  # explicit convention: tonic in octave 3
    value = tonic + MAJOR[degree - 1] + 12 * args.octave + args.accidental
    return value, {"kind": "jianpu", "degree": degree, "tonic_midi": tonic,
                   "octave": args.octave, "accidental": args.accidental}


def candidate_result(args):
    try:
        target, target_info = target_midi(args)
        opens, source = tuning(args)
    except ValueError as exc:
        return failure("invalid", str(exc))
    candidates = []
    for string, open_midi in enumerate(opens, 1):
        if abs(open_midi - target) * 100 <= args.tolerance_cents:
            candidates.append(candidate(open_midi, target, string, None, "open"))
        for step in range(10, 131):
            hui = step / 10
            coord = hui_coordinate(hui)
            value = open_midi + 12 * math.log2(1 / coord)
            if abs(value - target) * 100 <= args.tolerance_cents:
                candidates.append(candidate(value, target, string, hui, "stopped"))
        for hui in range(1, 14):
            value = open_midi + HARMONIC_SEMITONES[hui - 1]
            if abs(value - target) * 100 <= args.tolerance_cents:
                candidates.append(candidate(value, target, string, hui, "harmonic"))
    candidates.sort(key=lambda c: (abs(c["cents_from_target"]),
                                    c["mode"] != "open", c["string"], c["hui"] or 0))
    return {
        "status": "exact" if candidates else "indeterminate",
        "pitches": [pitch_obj(target)],
        "candidates": candidates,
        "evidence": {"target": target_info, "open_midi": opens,
                     "tuning_source": source,
                     "tolerance_cents": args.tolerance_cents},
        "diagnostics": [] if candidates else [{"code": "no_candidate_in_search_grid"}],
    }


def candidate(value, target, string, hui, mode):
    return {"string": string, "hui": hui, "mode": mode,
            "midi": value, "name": pitch_obj(value)["name"],
            "cents_from_target": round((value - target) * 100, 3)}


def validate_result(args):
    pos = position_result(args)
    if not pos["pitches"]:
        return pos
    try:
        target, target_info = target_midi(args)
    except ValueError as exc:
        return failure("invalid", str(exc))
    actual = pos["pitches"][0]["midi"]
    delta = (actual - target) * 100
    matched = abs(delta) <= args.tolerance_cents
    pos["match"] = matched
    pos["target"] = pitch_obj(target)
    pos["delta_cents"] = round(delta, 3)
    pos["evidence"]["target"] = target_info
    pos["status"] = pos["status"] if matched else "mismatch"
    if not matched:
        pos["diagnostics"].append({"code": "pitch_mismatch",
                                   "delta_cents": round(delta, 3)})
    return pos


def add_tuning(parser):
    parser.add_argument("--open-midi", help="seven comma-separated open-string MIDI values")
    parser.add_argument("--offsets", help="seven semitone offsets from 正调")
    parser.add_argument("--profile", choices=sorted(TUNING_PROFILES),
                        default="modern",
                        help="absolute-octave convention when --open-midi is absent")


def add_target(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--midi", type=float)
    group.add_argument("--jianpu", type=int)
    parser.add_argument("--tonic", default="F")
    parser.add_argument("--tonic-midi", type=float)
    parser.add_argument("--octave", type=int, default=0)
    parser.add_argument("--accidental", type=int, default=0)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("position", help="map string/hui to pitch")
    p.add_argument("--string", type=int, required=True)
    p.add_argument("--hui")
    p.add_argument("--mode", choices=["stopped", "harmonic", "open", "san"],
                   default="stopped")
    add_tuning(p)
    p.set_defaults(run=position_result)
    p = sub.add_parser("pitch", help="map target pitch to candidate positions")
    add_target(p)
    add_tuning(p)
    p.add_argument("--tolerance-cents", type=float, default=50)
    p.set_defaults(run=candidate_result)
    p = sub.add_parser("validate", help="compare a position with a target pitch")
    p.add_argument("--string", type=int, required=True)
    p.add_argument("--hui")
    p.add_argument("--mode", choices=["stopped", "harmonic", "open", "san"],
                   default="stopped")
    add_target(p)
    add_tuning(p)
    p.add_argument("--tolerance-cents", type=float, default=50)
    p.set_defaults(run=validate_result)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        emit(args.run(args))
    except Exception as exc:  # keep CLI contract JSON-only
        emit(failure("invalid", f"{type(exc).__name__}: {exc}"), 2)


if __name__ == "__main__":
    main()
