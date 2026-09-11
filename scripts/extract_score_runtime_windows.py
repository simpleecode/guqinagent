#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ADDR_KEY_RE = re.compile(r"^(.*?)![^@!]+@[0-9a-fA-F]+$")


def clean_obj(obj):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            match = ADDR_KEY_RE.match(key)
            out[match.group(1) if match else key] = clean_obj(value)
        return out
    if isinstance(obj, list):
        return [clean_obj(item) for item in obj]
    return obj


def enum_name(obj):
    if isinstance(obj, dict):
        parent = obj.get("parent!_Enum")
        if isinstance(parent, dict):
            return parent.get("off_10")
    return None


def unsigned_to_signed(value):
    try:
        n = int(value)
    except Exception:
        return value
    if n >= 2**63:
        return n - 2**64
    return n


def decode_note(lab):
    if not isinstance(lab, dict):
        return {}
    dur = lab.get("off_8", {})
    pitch = lab.get("off_14", {})
    alt_pitch = lab.get("off_18")
    slurs = []
    for item in lab.get("off_1c", []) or []:
        if isinstance(item, dict):
            slurs.append({
                "type": enum_name(item),
                "symbol": item.get("off_14"),
                "value": item.get("off_18"),
                "raw": item,
            })
    note_digit = pitch.get("off_8") if isinstance(pitch, dict) else None
    octave = unsigned_to_signed(pitch.get("off_c")) if isinstance(pitch, dict) else None
    dur_prefix = dur.get("off_1c", "") if isinstance(dur, dict) else ""
    octave_suffix = ""
    if octave == -1:
        octave_suffix = ","
    elif octave == -2:
        octave_suffix = ",,"
    elif octave == 1:
        octave_suffix = "'"
    elif octave == 2:
        octave_suffix = "''"
    note_text = f"{dur_prefix or ''}{note_digit or ''}{octave_suffix}"
    return {
        "note": note_text,
        "duration": {
            "name": enum_name(dur),
            "value": dur.get("off_14") if isinstance(dur, dict) else None,
            "code": dur_prefix,
            "unit": dur.get("off_20") if isinstance(dur, dict) else None,
        },
        "pitch": {
            "value": note_digit,
            "octave": octave,
            "accidental": enum_name(pitch.get("off_14")) if isinstance(pitch, dict) else None,
            "raw": pitch,
        },
        "alt_pitch": alt_pitch,
        "slurs": slurs,
        "raw": lab,
    }


def decode_jian_component(component):
    if not isinstance(component, dict):
        return {}
    typ = enum_name(component.get("off_1c"))
    std = {
        "tp": typ.lower() if isinstance(typ, str) else typ,
        "raw_type": typ,
        "raw_fields": component,
    }
    if typ == "ZHTYX":
        std.update({
            "z": component.get("off_8"),
            "h": component.get("off_c"),
            "t": component.get("off_10"),
            "y": component.get("off_14"),
            "x": component.get("off_18"),
        })
    elif typ == "TH":
        std.update({"h": component.get("off_c"), "t": component.get("off_10")})
    elif typ == "D":
        std.update({"d": component.get("off_10")})
    else:
        for out_key, in_key in [
            ("a", "off_8"), ("b", "off_c"), ("c", "off_10"),
            ("d", "off_14"), ("e", "off_18"), ("f", "off_20"),
            ("g", "off_24"), ("h", "off_28"),
        ]:
            if in_key in component:
                std[out_key] = component[in_key]
    return std


def decode_jian(jian):
    if not isinstance(jian, dict):
        return {"raw": jian}
    stds = []
    head = jian.get("off_8", {})
    for key in ("off_8", "off_c"):
        component = head.get(key) if isinstance(head, dict) else None
        if isinstance(component, dict):
            stds.append(decode_jian_component(component))
    return {
        "std": stds[0] if stds else {},
        "std_all": stds,
        "layout": jian.get("off_c"),
        "raw": jian,
    }


def event_hash(event):
    blob = json.dumps(event, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def recover_shallow_source_event(item):
    """Rebuild a jab event from the field-by-field Frida fallback payload."""
    if not isinstance(item, dict):
        return None
    shallow = item.get("shallow_fields")
    if not isinstance(shallow, dict):
        return None
    nested = (shallow.get("off_8") or {}).get("nested_object") or {}
    fields = nested.get("fields") or {}
    duration = (fields.get("off_8") or {}).get("value")
    pitch = (fields.get("off_14") or {}).get("value")
    if not isinstance(duration, dict) or not isinstance(pitch, dict):
        return None

    def value(name, default=""):
        field = shallow.get(name) or {}
        return field.get("value", default)

    off_c = value("off_c", [])
    alt_pitch = (fields.get("off_18") or {}).get("value")
    slurs = (fields.get("off_1c") or {}).get("value")
    return {
        "off_8": {
            "off_8": duration,
            "off_c": nested.get("off_c_native", "0"),
            "off_14": pitch,
            "off_18": alt_pitch,
            "off_1c": slurs if isinstance(slurs, list) else [],
        },
        "off_c": off_c if isinstance(off_c, list) else [],
        "off_10": value("off_10"),
        "off_14": value("off_14"),
        "off_18": value("off_18"),
        "off_1c": value("off_1c", None),
        "_reconstruction_source": "source_event_shallow_recovery",
    }


def extract_source_events(jsonl_path):
    """Read the complete pre-window jab event list from an internal hook."""
    with Path(jsonl_path).open(encoding="utf-8") as source:
        for line in source:
            try:
                payload = json.loads(line).get("message", {}).get("payload", {})
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "enter_decoded":
                continue
            if payload.get("name") != "source_jab_event":
                continue
            value = payload.get("regs", {}).get("x0", {}).get("value")
            if not isinstance(value, list):
                continue
            cleaned = clean_obj(value)
            events = []
            omitted = []
            for source_index, item in enumerate(cleaned):
                if isinstance(item, dict) and isinstance(item.get("off_8"), dict):
                    events.append(dict(
                        item,
                        _reconstruction_source="source_event_list",
                        _source_index=source_index,
                    ))
                else:
                    recovered = recover_shallow_source_event(item)
                    if recovered is not None:
                        recovered["_source_index"] = source_index
                        events.append(recovered)
                        continue
                    omitted.append({
                        "source_index": source_index,
                        "value": item,
                        "reason": (
                            "runtime_decode_error"
                            if isinstance(item, dict) and item.get("decode_error")
                            else "layout_or_control_entry"
                        ),
                    })
            if events:
                return events, {
                    "mode": "source_event_list",
                    "source_entries": len(cleaned),
                    "score_events": len(events),
                    "layout_or_control_entries": len(omitted),
                    "omitted_source_entries": omitted,
                    "runtime_decode_error_indexes": [
                        item["source_index"] for item in omitted
                        if item["reason"] == "runtime_decode_error"
                    ],
                    "note_call_only_events": 0,
                }
    return [], {}


def extract_windows(jsonl_path):
    events = []
    windows = []
    note_calls = []
    records = []
    for line in Path(jsonl_path).open(encoding="utf-8"):
        try:
            payload = json.loads(line).get("message", {}).get("payload", {})
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "enter_decoded":
            continue
        if payload.get("name") != "note_slur_eJk_hab_63f748":
            continue
        regs = payload.get("regs", {})
        note_value = regs.get("x1", {}).get("value")
        note = None
        if isinstance(note_value, dict):
            note = clean_obj(note_value)
            decoded = decode_note(note)
            if decoded.get("note"):
                note_calls.append(note)
        value = regs.get("x7", {}).get("value")
        if not isinstance(value, dict):
            continue
        value = clean_obj(value)
        window_events = value.get("off_8")
        if isinstance(window_events, list):
            windows.append(window_events)
            events.extend(window_events)
            if note is not None:
                records.append({"note": note, "window": window_events})
    seen = set()
    unique = []
    for event in events:
        digest = event_hash(event)
        if digest not in seen:
            seen.add(digest)
            unique.append(event)

    # Do not deduplicate the call stream. Consecutive identical pitches are
    # legal score events, and each note_slur invocation represents a position.
    return windows, unique, note_calls, records


def extract_runtime_tuning(jsonl_path):
    """Find the Nab tuning object observed by note_tuning_uJk_63e180."""
    candidates = []

    def visit(value):
        if isinstance(value, dict):
            name = value.get("off_8")
            offsets = value.get("off_c")
            if (
                isinstance(name, str)
                and isinstance(offsets, list)
                and len(offsets) == 8
                and all(isinstance(item, int) for item in offsets)
            ):
                candidates.append({"name": name, "value": offsets})
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    with Path(jsonl_path).open(encoding="utf-8") as source:
        for line in source:
            try:
                payload = json.loads(line).get("message", {}).get("payload", {})
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "enter_decoded":
                continue
            if payload.get("name") != "note_tuning_uJk_63e180":
                continue
            visit(clean_obj(payload.get("regs", {})))

    unique = []
    seen = set()
    for candidate in candidates:
        key = (candidate["name"], tuple(candidate["value"]))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    if len(unique) == 1:
        return dict(unique[0], source="runtime_note_tuning"), unique
    return None, unique


def note_only_event(note):
    return {
        "off_8": note,
        "off_c": [],
        "off_10": "",
        "off_14": "",
        "off_18": "",
        "off_1c": None,
        "_reconstruction_source": "note_call_only",
    }


def is_one_event_extension(previous_window, current_window):
    """Whether current_window is previous_window plus exactly one event."""
    if len(current_window) != len(previous_window) + 1:
        return False
    return all(
        event_hash(left) == event_hash(right)
        for left, right in zip(previous_window, current_window[:-1])
    )


def reconstruct_from_growth_records(records):
    """Rebuild the call stream from growing WZa render windows.

    During rendering the app exposes a prefix window of length N in x7 and the
    next note in x1.  The following call normally exposes a window of length
    N+1, whose last full event is that previous x1 note.  At a render-window
    boundary the N+1 window is absent, so only the note itself can be retained.
    """
    if not records:
        return [], {
            "mode": "growth_record_alignment",
            "note_call_events": 0,
            "window_aligned_events": 0,
            "note_call_only_events": 0,
        }

    rebuilt = []
    aligned = 0
    for previous, current in zip(records, records[1:]):
        previous_note = previous["note"]
        previous_window = previous["window"]
        current_window = current["window"]
        if is_one_event_extension(previous_window, current_window):
            candidate = current_window[-1]
            candidate_note = candidate.get("off_8") if isinstance(candidate, dict) else None
            if (isinstance(candidate_note, dict)
                    and event_hash(candidate_note) == event_hash(previous_note)):
                rebuilt.append(candidate)
                aligned += 1
                continue
        rebuilt.append(note_only_event(previous_note))

    rebuilt.append(note_only_event(records[-1]["note"]))
    note_only = len(rebuilt) - aligned
    return rebuilt, {
        "mode": "growth_record_alignment",
        "note_call_events": len(records),
        "window_aligned_events": aligned,
        "note_call_only_events": note_only,
    }


def merge_note_calls_with_windows(window_events, note_calls):
    if len(note_calls) <= len(window_events):
        return window_events, {
            "mode": "window_events",
            "note_call_events": len(note_calls),
            "window_events": len(window_events),
            "note_call_only_events": 0,
        }

    by_note_hash = {}
    for event in window_events:
        note_obj = event.get("off_8")
        digest = event_hash(note_obj) if isinstance(note_obj, dict) else None
        if not digest:
            continue
        by_note_hash.setdefault(digest, []).append(event)

    merged = []
    note_call_only = 0
    for note in note_calls:
        digest = event_hash(note)
        queue = by_note_hash.get(digest) or []
        if queue:
            merged.append(queue.pop(0))
        else:
            note_call_only += 1
            merged.append(note_only_event(note))

    return merged, {
        "mode": "note_call_stream_with_window_alignment",
        "note_call_events": len(note_calls),
        "window_events": len(window_events),
        "note_call_only_events": note_call_only,
    }


def build_outputs(events, args, reconstruction=None):
    reconstruction = reconstruction or {}
    notes = []
    for index, event in enumerate(events):
        note = decode_note(event.get("off_8"))
        jians = [decode_jian(item) for item in (event.get("off_c") or [])]
        notes.append({
            "index": index,
            "note": note.get("note", ""),
            "note_decoded": note,
            "jians": jians,
            "lyric": event.get("off_10", ""),
            "lyric1": event.get("off_14", ""),
            "lyric2": event.get("off_18", ""),
            "section": event.get("off_1c"),
            "reconstruction_source": event.get("_reconstruction_source", "window_event"),
            "raw_event": event,
        })

    stats = {
        "total_events": len(notes),
        "jianpu_events": sum(1 for item in notes if item["note"]),
        "jianzi_events": sum(1 for item in notes if item["jians"]),
        "jian_count": sum(len(item["jians"]) for item in notes),
        "successful_alignments": sum(1 for item in notes if item["note"] and item["jians"]),
        "unaligned_event_indexes": [
            item["index"] for item in notes if item["note"] and not item["jians"]
        ],
        "note_call_only_event_indexes": [
            item["index"] for item in notes
            if item["reconstruction_source"] == "note_call_only"
        ],
    }
    stats["unaligned_count"] = len(stats["unaligned_event_indexes"])
    stats["jianzi_missing_after_note_recovery"] = [
        item["index"] for item in notes
        if item["note"] and not item["jians"] and item["reconstruction_source"] == "note_call_only"
    ]
    stats["metadata_length_gap"] = {
        "metadata_notes_length": args.notes_length,
        "captured_jianpu_events": stats["jianpu_events"],
        "gap": args.notes_length - stats["jianpu_events"] if args.notes_length else None,
    }
    stats["source_metadata_length"] = {
        "value": args.notes_length,
        "meaning": "App score metadata field named length/notes_length; observed not to be a runtime event count.",
        "used_as_event_count": False,
    }
    note_call_only = len(stats["note_call_only_event_indexes"])
    metadata_gap = stats["metadata_length_gap"]["gap"]
    complete = bool(
        args.assume_complete_runtime_window
        and note_call_only == 0
        and (metadata_gap is None or metadata_gap <= 0)
    )
    stats["runtime_window_stitch"] = {
        "source": "note_slur WZa render windows",
        "risk": "high" if note_call_only or (metadata_gap and metadata_gap > 0) else "medium",
        "meaning": (
            "Events are reconstructed from repeated runtime render windows. "
            "If every scroll position is not observed, a note between two captured windows can be missed."
        ),
    }
    stats["reconstruction"] = reconstruction
    stats["capture_completeness"] = {
        "method": args.completeness_method,
        "complete": complete,
        "note": (
            "The note_slur WZa runtime data is the app's composed score event structure used for rendering, "
            f"and this export captured {len(notes)} unique runtime events. "
            "A note_call_only event means the jianpu was recovered from the note_slur argument, "
            "but the matching WZa event carrying jianzi was not captured."
        ),
    }

    tuning = {
        "name": args.tuning_name,
        "value": args.tuning_values,
        "source": (
            getattr(args, "tuning_source", "capture_argument")
            if args.tuning_values is not None
            else "unknown_not_captured"
        ),
    }
    raw = {
        "provenance": {
            "method": "Frida runtime object capture from authorized logged-in Android app",
            "source_jsonl": str(args.input),
            "api_source": f"GET https://s.sitongli.net/v2/scores/{args.score_id}/data",
            "api_response_file": str(args.api_response) if args.api_response else None,
            "target_package": "com.sitongli.app.gadget",
        },
        "metadata": {
            "score_id": args.score_id,
            "score_key": args.score_key,
            "from_id": args.from_id,
            "from_key": args.from_key,
            "score_title": args.title,
            "notes_length_metadata": args.notes_length,
            "meter": "1/4",
            "tonic": args.tonic,
            "tuning": tuning,
            "sections": [{"start": 0, "title": "", "tempo": 40}],
        },
        "raw_score_data": {
            "meter": "1/4",
            "tonic": args.tonic,
            "tuning": tuning,
            "sections": [{"start": 0, "title": "", "tempo": 40}],
            "dataops": [],
            "notes": notes,
        },
    }
    data = {
        "metadata": raw["metadata"],
        "field_notes": {
            "raw_preservation": "raw_event preserves decoded Dart runtime fields after stripping volatile heap addresses.",
            "alignment": "jianpu and jianzi are aligned by the same runtime event record.",
            "jianzi": "No OCR or Hanzi substitution is used; std/raw_fields preserve app component codes.",
            "note_call_only": (
                "note_call_only recovers a missing jianpu note from the runtime call stream only; "
                "it does not prove the corresponding jianzi was captured."
            ),
        },
        "stats": stats,
        "anomalies": [
            {
                "type": "jianpu_without_jianzi",
                "indexes": stats["unaligned_event_indexes"],
                "message": "Runtime event has jianpu but no aligned jianzi component.",
            }
        ] if stats["unaligned_event_indexes"] else [],
        "events": [
            {
                "index": item["index"],
                "jianpu": item["note"],
                "jianzi_raw": item["jians"],
                "jianzi_id_or_code": [
                    jian.get("std", {}).get("tp") for jian in item["jians"]
                ],
                "jianzi_components": [
                    jian.get("std", {}) for jian in item["jians"]
                ],
                "lyric": item["lyric"],
                "lyric1": item["lyric1"],
                "lyric2": item["lyric2"],
                "alignment": (
                    "note recovered from call stream; jianzi not captured"
                    if item["reconstruction_source"] == "note_call_only"
                    else "same runtime event"
                ),
                "reconstruction_source": item["reconstruction_source"],
                "raw_record": item,
            }
            for item in notes
        ],
    }
    return raw, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--api-response", type=Path)
    parser.add_argument("--score-id", type=int, default=161749)
    parser.add_argument("--score-key", default="SWDFmDZX")
    parser.add_argument("--from-id", type=int, default=155861)
    parser.add_argument("--from-key", default="SkvmkgjX")
    parser.add_argument("--title", default="《秋风词》")
    parser.add_argument("--notes-length", type=int, default=140)
    parser.add_argument(
        "--tonic",
        required=True,
        help="Numbered-notation tonic shown on the score, for example C in 1=C.",
    )
    parser.add_argument(
        "--tuning-name",
        help="Tuning name captured from the score, for example 正调.",
    )
    parser.add_argument(
        "--tuning-values",
        type=lambda value: [int(item) for item in value.split(",")],
        help="Eight captured tuning integers, comma-separated. Never inferred.",
    )
    parser.add_argument(
        "--assume-complete-runtime-window",
        action="store_true",
        help="Mark the captured WZa runtime render window as complete after repeat-run verification.",
    )
    parser.add_argument(
        "--completeness-method",
        default="repeat-captured note_slur WZa runtime render window",
    )
    args = parser.parse_args()
    if args.tuning_values is not None and len(args.tuning_values) != 8:
        parser.error("--tuning-values must contain exactly eight integers")

    events, reconstruction = extract_source_events(args.input)
    if events:
        windows, window_events, note_calls = [], [], []
    else:
        windows, window_events, note_calls, records = extract_windows(args.input)
        if records:
            events, reconstruction = reconstruct_from_growth_records(records)
        else:
            events, reconstruction = merge_note_calls_with_windows(
                window_events, note_calls
            )
    if not events:
        print(json.dumps({
            "windows": len(windows),
            "window_events": len(window_events),
            "note_call_events": len(note_calls),
            "captured_events": 0,
            "error": "No runtime score events captured. Open the target score body/editor page and trigger rendering by scrolling, then retry.",
            "input": str(args.input),
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(2)

    captured_tuning, tuning_candidates = extract_runtime_tuning(args.input)
    reconstruction["tuning_candidates"] = tuning_candidates
    if captured_tuning and args.tuning_values is None:
        args.tuning_name = captured_tuning["name"]
        args.tuning_values = captured_tuning["value"]
        args.tuning_source = captured_tuning["source"]
    raw, data = build_outputs(events, args, reconstruction)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw_data.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "windows": len(windows),
        "window_events": len(window_events),
        "note_call_events": len(note_calls),
        "captured_events": len(events),
        "reconstruction": reconstruction,
        "output_raw": str(args.out_dir / "raw_data.json"),
        "output_data": str(args.out_dir / "data.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
