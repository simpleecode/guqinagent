#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import re
from typing import Any


KEYWORDS = (
    "SSG54sm8",
    "关山月",
    "score_id",
    "score_key",
    "score_title",
    "notes",
    "jians",
    "sections",
    "dataops",
    "jian",
    "lyric",
)


def bytes_from_hex(s: str) -> bytes:
    return bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", s))


def text_variants(blob: bytes) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    for name, encoding in (("utf8", "utf-8"), ("utf16le", "utf-16le")):
        try:
            text = blob.decode(encoding, errors="ignore")
        except Exception:
            continue
        text = text.replace("\x00", "")
        if any(k in text for k in KEYWORDS) or "{" in text or "[" in text:
            variants.append((name, text))
    return variants


def iter_json_values(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        try:
            value, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if end >= 8:
            values.append(value)
    return values


def score_signal(value: Any) -> dict[str, Any]:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {
        "bytes": len(raw.encode("utf-8")),
        "contains": {k: (k in raw) for k in KEYWORDS},
        "top_type": type(value).__name__,
        "top_keys": list(value.keys())[:30] if isinstance(value, dict) else None,
        "list_len": len(value) if isinstance(value, list) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--out-dir", default=r"cases\sitongli-guanshanyue\evidence\memory_json_candidates")
    ap.add_argument("--min-bytes", type=int, default=32)
    args = ap.parse_args()

    src = pathlib.Path(args.jsonl)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    index: list[dict[str, Any]] = []
    raw_text_dir = out_dir / "texts"
    raw_text_dir.mkdir(exist_ok=True)

    for line_no, line in enumerate(src.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        payload = rec.get("message", {}).get("payload")
        if not isinstance(payload, dict) or payload.get("event") != "hit":
            continue
        hex_text = payload.get("hex") or ""
        if not hex_text:
            continue
        blob = bytes_from_hex(hex_text)
        for variant, text in text_variants(blob):
            if any(k in text for k in KEYWORDS):
                text_digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
                text_path = raw_text_dir / f"{line_no:05d}_{variant}_{text_digest[:12]}.txt"
                if not text_path.exists():
                    text_path.write_text(text, encoding="utf-8")
            for value in iter_json_values(text):
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if len(rendered.encode("utf-8")) < args.min_bytes:
                    continue
                digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                json_path = out_dir / f"candidate_{len(index):04d}_{digest[:12]}.json"
                json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
                entry = {
                    "candidate": json_path.name,
                    "source_line": line_no,
                    "needle": payload.get("needle"),
                    "encoding": payload.get("encoding"),
                    "address": payload.get("address"),
                    "variant": variant,
                    "sha256": digest,
                    **score_signal(value),
                }
                index.append(entry)

    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(index)} JSON candidates to {out_dir}")
    for entry in sorted(index, key=lambda e: (sum(e["contains"].values()), e["bytes"]), reverse=True)[:20]:
        print(json.dumps(entry, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
