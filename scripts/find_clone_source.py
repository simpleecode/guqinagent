import argparse
import json
import pathlib


def iter_json_files(path: pathlib.Path):
    for file in sorted(path.glob("candidate_*.json")):
        try:
            yield file, json.loads(file.read_text("utf-8"))
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_dir")
    ap.add_argument("--score-key", required=True)
    ap.add_argument("--score-id", required=True)
    args = ap.parse_args()

    candidate_dir = pathlib.Path(args.candidate_dir)
    score_key = str(args.score_key)
    score_id = str(args.score_id)

    matches = []
    for file, obj in iter_json_files(candidate_dir):
        if not isinstance(obj, dict):
            continue
        if str(obj.get("key") or "") != score_key and str(obj.get("id") or "") != score_id:
            continue
        from_key = obj.get("from_key") or obj.get("fromKey")
        from_id = obj.get("from_id") or obj.get("fromId")
        if not from_key and not from_id:
            continue
        matches.append({
            "source_candidate": str(file),
            "score_key": score_key,
            "score_id": score_id,
            "from_key": str(from_key or ""),
            "from_id": str(from_id or ""),
            "from_title": obj.get("from_title") or obj.get("fromTitle") or "",
            "from_nickname": obj.get("from_nickname") or obj.get("fromNickname") or "",
        })

    # The app can keep several metadata shapes for the same cloned score in memory.
    # Prefer the richer record with from_key because scanning only by from_id is too broad.
    matches.sort(key=lambda item: (bool(item["from_key"]), bool(item["from_id"])), reverse=True)
    best = matches[0] if matches else None

    print(json.dumps(best or {}, ensure_ascii=False))


if __name__ == "__main__":
    main()
