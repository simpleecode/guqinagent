#!/usr/bin/env python3
import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence


SCORE_KEYS = {
    "notes", "sections", "jians", "jian", "lyric", "lyric1", "lyric2",
    "native_score", "score_id", "score_key", "score_title", "dataops",
    "ops", "patch", "is_json_patch"
}


def walk(value, path="$"):
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from walk(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for idx, item in enumerate(value):
            yield from walk(item, f"{path}[{idx}]")


def score_candidate(obj):
    if not isinstance(obj, Mapping):
        return 0
    keys = set(map(str, obj.keys()))
    score = len(keys & SCORE_KEYS)
    for group in (("notes", "jians"), ("notes", "sections"), ("native_score",), ("dataops",), ("ops",)):
        if any(k in keys for k in group):
            score += 3
    if any(str(v).find("关山月") >= 0 or str(v).find("SSG54sm8") >= 0 for v in obj.values()):
        score += 5
    return score


def parse_capture(path):
    candidates = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = rec.get("message", {}).get("payload")
            for obj_path, obj in walk(payload):
                rank = score_candidate(obj)
                if rank > 0:
                    candidates.append({
                        "rank": rank,
                        "line": line_no,
                        "path": obj_path,
                        "hook": payload.get("name") if isinstance(payload, Mapping) else None,
                        "event": payload.get("event") if isinstance(payload, Mapping) else None,
                        "object": obj,
                    })
    candidates.sort(key=lambda item: (-item["rank"], item["line"], item["path"]))
    return candidates


def first_list(record, names):
    for _, obj in walk(record):
        if isinstance(obj, Mapping):
            for name in names:
                val = obj.get(name)
                if isinstance(val, list):
                    return val
    return []


def normalize(raw):
    notes = first_list(raw, ["notes", "note"])
    jians = first_list(raw, ["jians", "jian"])
    sections = first_list(raw, ["sections", "section"])
    total = max(len(notes), len(jians))
    events = []
    for idx in range(total):
        note = notes[idx] if idx < len(notes) else None
        jian = jians[idx] if idx < len(jians) else None
        events.append({
            "index": idx,
            "jianpu": note,
            "jianzi_raw": jian,
            "jianzi_id": jian.get("id") if isinstance(jian, Mapping) else None,
            "jianzi_components": jian.get("components") if isinstance(jian, Mapping) else None,
            "section": None,
            "measure": None,
            "time_or_position": None,
            "raw_record": {"note": note, "jian": jian},
        })
    return {
        "meta": {
            "title": "关山月",
            "score_key": "SSG54sm8",
            "source": "frida_gadget_blutter_runtime_capture",
        },
        "events": events,
        "sections": sections,
        "stats": {
            "total_events": total,
            "jianpu_events": len(notes),
            "jianzi_events": len(jians),
            "aligned_events": sum(1 for i in range(total) if i < len(notes) and i < len(jians)),
            "unaligned_jianpu": max(0, len(notes) - len(jians)),
            "unaligned_jianzi": max(0, len(jians) - len(notes)),
            "anomalies": [],
        },
        "field_notes": {
            "jianpu": "App runtime note record from notes/note arrays when present.",
            "jianzi_raw": "Original App runtime jianzi record, not OCR.",
            "jianzi_id": "ID-like field if exposed by runtime object.",
            "jianzi_components": "Component sequence if exposed by runtime object.",
            "time_or_position": "Position/time fields are preserved inside raw_record unless a stable schema is confirmed.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Extract raw and normalized score JSON from Frida Gadget capture.")
    parser.add_argument("--capture", default=r"cases\sitongli-guanshanyue\evidence\gadget_capture_score.jsonl")
    parser.add_argument("--out-dir", default=r"cases\sitongli-guanshanyue\out")
    args = parser.parse_args()

    capture = pathlib.Path(args.capture)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = parse_capture(capture)
    (out_dir / "capture_candidates.json").write_text(
        json.dumps(candidates[:200], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not candidates:
        raise SystemExit("No score-like runtime object found in capture.")

    raw = {
        "source_capture": str(capture),
        "selected_candidate": candidates[0],
        "candidate_count": len(candidates),
    }
    (out_dir / "raw_data.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    normalized = normalize(candidates[0]["object"])
    (out_dir / "data.json").write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'raw_data.json'}")
    print(f"Wrote {out_dir / 'data.json'}")
    print(json.dumps(normalized["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
