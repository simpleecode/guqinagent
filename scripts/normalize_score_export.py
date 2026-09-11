#!/usr/bin/env python3
import argparse
import json
import pathlib
from typing import Any, Optional


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def looks_like_score_data(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("notes"), list)
        and isinstance(obj.get("sections"), list)
        and ("meter" in obj or "tuning" in obj)
    )


def note_text_size(score: dict[str, Any]) -> int:
    total = 0
    for note in score.get("notes", []):
        if isinstance(note, dict):
            total += len(str(note.get("note") or ""))
            jians = note.get("jians")
            if isinstance(jians, list):
                total += len(json.dumps(jians, ensure_ascii=False, separators=(",", ":")))
        else:
            total += len(str(note))
    return total


def matches_expected_size(score: dict[str, Any], expected_notes_length: Optional[int], strict: bool = False) -> bool:
    if expected_notes_length is None:
        return True
    size = note_text_size(score)
    # notes_length is not an event count; it is closer to serialized score text size.
    # Cloned scores need a tighter threshold because stale complete score bodies can stay in memory.
    threshold = 0.70 if strict else 0.35
    return size >= max(1, int(expected_notes_length * threshold))


def pick_score_data(path: pathlib.Path, expected_notes_length: Optional[int] = None, strict_size: bool = False) -> tuple[Any, pathlib.Path]:
    if path.is_file():
        obj = load_json(path)
        if looks_like_score_data(obj):
            if not matches_expected_size(obj, expected_notes_length, strict_size):
                raise SystemExit(f"score candidate is too small for target notes_length: {path}")
            return obj, path
        if isinstance(obj, dict) and looks_like_score_data(obj.get("raw_score_data")):
            if not matches_expected_size(obj["raw_score_data"], expected_notes_length, strict_size):
                raise SystemExit(f"score candidate is too small for target notes_length: {path}")
            return obj["raw_score_data"], path
        raise SystemExit(f"not a score data JSON: {path}")

    candidates = []
    for candidate in path.glob("candidate_*.json"):
        obj = load_json(candidate)
        if looks_like_score_data(obj) and matches_expected_size(obj, expected_notes_length, strict_size):
            candidates.append((note_text_size(obj), len(obj.get("notes", [])), candidate, obj))
    if not candidates:
        raise SystemExit(f"no score data candidate found in {path}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][3], candidates[0][2]


def pick_metadata(candidate_dir: Optional[pathlib.Path], score_key: str, score_id: Optional[int], title: str) -> dict[str, Any]:
    metadata = {
        "score_id": score_id,
        "score_key": score_key,
        "score_title": title,
    }
    if not candidate_dir or not candidate_dir.is_dir():
        return metadata

    for candidate in candidate_dir.glob("candidate_*.json"):
        try:
            obj = load_json(candidate)
        except Exception:
            continue
        records = obj.get("data") if isinstance(obj, dict) else obj
        if not isinstance(records, list):
            records = [obj]
        for record in records:
            if not isinstance(record, dict):
                continue
            if (score_key and record.get("key") == score_key) or (score_id is not None and record.get("id") == score_id):
                metadata.update(
                    {
                        "score_id": record.get("id", score_id),
                        "score_key": record.get("key", score_key),
                        "score_title": title or record.get("title"),
                        "source_title_raw": record.get("title"),
                        "notes_length_metadata": record.get("notes_length"),
                        "uid": record.get("uid"),
                        "create_time_raw": record.get("create_time"),
                        "update_time_raw": record.get("update_time"),
                        "raw_metadata_record": record,
                    }
                )
                return metadata
    return metadata


def normalize_jian(jian: Any) -> dict[str, Any]:
    components = None
    code = None
    if isinstance(jian, dict):
        components = jian.get("std") or jian.get("components") or jian.get("parts")
        if isinstance(components, dict):
            code = components.get("tp")
    return {
        "raw": jian,
        "internal_code_or_id": code,
        "components": components,
    }


def normalize(score: dict[str, Any], metadata: dict[str, Any], source: str) -> dict[str, Any]:
    notes = score.get("notes") if isinstance(score.get("notes"), list) else []
    sections = score.get("sections") if isinstance(score.get("sections"), list) else []
    events = []
    anomalies = []

    for index, raw_note in enumerate(notes):
        note_text = raw_note.get("note") if isinstance(raw_note, dict) else raw_note
        jians = raw_note.get("jians") if isinstance(raw_note, dict) and isinstance(raw_note.get("jians"), list) else []
        normalized_jians = [normalize_jian(jian) for jian in jians]
        event = {
            "index": index,
            "section_index": None,
            "measure_or_paragraph": None,
            "position_or_time": index,
            "jianpu": note_text,
            "jianzi_raw": jians if jians else None,
            "jianzi_id_or_code": [j["internal_code_or_id"] for j in normalized_jians] or None,
            "jianzi_components": [j["components"] for j in normalized_jians] or None,
            "alignment": "same note record" if jians else "jianpu_only",
            "raw_record": raw_note,
        }
        events.append(event)

    for section_index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        start = section.get("start")
        end = sections[section_index + 1].get("start") if section_index + 1 < len(sections) and isinstance(sections[section_index + 1], dict) else None
        for event in events:
            if isinstance(start, int) and event["index"] >= start and (end is None or event["index"] < end):
                event["section_index"] = section_index
                event["measure_or_paragraph"] = section.get("title") or section_index

    if metadata.get("notes_length_metadata") is not None:
        try:
            expected = int(metadata["notes_length_metadata"])
            if expected != len(notes):
                anomalies.append({"type": "notes_length_mismatch", "metadata": expected, "actual": len(notes)})
        except Exception:
            pass

    jianpu_events = sum(1 for event in events if event["jianpu"] not in (None, ""))
    jianzi_events = sum(1 for event in events if event["jianzi_raw"])
    aligned_events = sum(1 for event in events if event["jianpu"] not in (None, "") and event["jianzi_raw"])
    unaligned = [event["index"] for event in events if not event["jianzi_raw"]]

    return {
        "metadata": {
            **{k: v for k, v in metadata.items() if k != "raw_metadata_record"},
            "source": source,
            "meter": score.get("meter"),
            "lyric": score.get("lyric"),
            "tuning": score.get("tuning"),
            "sections": sections,
        },
        "field_notes": {
            "jianpu": "App note field from each raw notes[] record. Values such as q1, and | are preserved as internal simple-score text.",
            "jianzi_raw": "Original jianzi objects from notes[].jians. No OCR or Hanzi substitution is performed.",
            "jianzi_id_or_code": "Currently the std.tp field when present, e.g. x, tyx, zhyx. This is the app's component/type sequence marker.",
            "jianzi_components": "The raw std component map preserved from the app, e.g. x/y/z/h/t/tp keys.",
            "alignment": "The app stores jianpu and jianzi in the same notes[] event record here, so alignment is record-based, not inferred by OCR.",
        },
        "stats": {
            "total_events": len(events),
            "jianpu_events": jianpu_events,
            "jianzi_events": jianzi_events,
            "successful_alignments": aligned_events,
            "unaligned_event_indexes": unaligned,
            "unaligned_count": len(unaligned),
            "anomaly_count": len(anomalies),
        },
        "anomalies": anomalies,
        "events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize extracted Sitongli score data.")
    ap.add_argument("--input", required=True, help="Score candidate JSON file or candidate directory.")
    ap.add_argument("--metadata-dir", default=None, help="Optional memory candidate directory containing list metadata.")
    ap.add_argument("--outdir", default=r"cases\sitongli-guanshanyue\out")
    ap.add_argument("--score-key", default="SSG54sm8")
    ap.add_argument("--score-id", type=int, default=161577)
    ap.add_argument("--title", default="\u5173\u5c71\u6708")
    args = ap.parse_args()

    input_path = pathlib.Path(args.input)
    metadata_dir = pathlib.Path(args.metadata_dir) if args.metadata_dir else (input_path if input_path.is_dir() else input_path.parent)
    metadata = pick_metadata(metadata_dir, args.score_key, args.score_id, args.title)
    expected_notes_length = None
    if metadata.get("notes_length_metadata") is not None:
        try:
            expected_notes_length = int(metadata["notes_length_metadata"])
        except Exception:
            expected_notes_length = None
    raw_metadata = metadata.get("raw_metadata_record") if isinstance(metadata.get("raw_metadata_record"), dict) else {}
    strict_size = bool(raw_metadata.get("from_id") or raw_metadata.get("from_key") or raw_metadata.get("fromId") or raw_metadata.get("fromKey"))
    score, source_path = pick_score_data(input_path, expected_notes_length, strict_size)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_doc = {
        "provenance": {
            "method": "Frida Gadget memory string scan of authorized logged-in app process",
            "source_candidate": str(source_path),
            "target_package": "com.sitongli.app.gadget",
            "score_key": args.score_key,
            "score_id": args.score_id,
            "score_title": args.title,
        },
        "metadata": metadata,
        "raw_score_data": score,
    }
    normalized = normalize(score, metadata, str(source_path))

    (outdir / "raw_data.json").write_text(json.dumps(raw_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "data.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(normalized["stats"], ensure_ascii=False, indent=2))
    print(f"wrote {outdir / 'raw_data.json'}")
    print(f"wrote {outdir / 'data.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
