#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse captured score-list responses into a batch manifest CSV.

Reads the `candidates/` + `url_index.json` produced by `capture_score_lists.py`,
identifies every score-list response, extracts {key, id, title, notes_length}
per record, attributes each list to a category (精选/热门/移植) by the request
URL path, deduplicates, and writes `score_manifest.csv`.

Category attribution priority:
    1. URL path keyword match (most reliable): the list endpoint URL usually
       carries a category id or slug. We map known keywords.
    2. If the URL is ambiguous or missing, fall back to capture order and
       STOP for operator confirmation rather than guess.

The script is intentionally conservative: if it cannot confidently attribute
a list to one of the three target categories, it prints the evidence and
exits non-zero so the operator can decide.
"""
import argparse
import csv
import json
import pathlib
import re
import sys
from typing import Any, Optional


# ---- category heuristics ------------------------------------------------

# Known Chinese labels for the three target tabs (defensive; the app may use
# either Chinese or pinyin/english slugs in the URL).
CATEGORY_KEYWORDS = {
    "精选": ["jingxuan", "featured", "curated", "choice", "pick", "精选", "jx"],
    "热门": ["remen", "hot", "popular", "trending", "热门", "rm"],
    "移植": ["yizhi", "transcrib", "adapt", "arrange", "移植", "yz"],
}


def category_from_url(url: str) -> Optional[str]:
    """Map a request URL to one of the three categories, or None."""
    if not url:
        return None
    low = url.lower()
    # also decode percent-encoded chinese in case the URL carries 精选/热门/移植
    try:
        from urllib.parse import unquote
        decoded = unquote(url)
    except Exception:
        decoded = url
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in low or kw in decoded:
                return cat
    return None


# ---- record extraction --------------------------------------------------

ID_KEYS = ("id", "score_id", "scoreId")
KEY_KEYS = ("key", "score_key", "scoreKey")
TITLE_KEYS = ("title", "name", "score_title", "scoreTitle")
NOTESLEN_KEYS = ("notes_length", "notesLength", "notes_len", "note_count")


def _first(rec: dict, keys: tuple) -> Any:
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return None


def extract_records(value: Any) -> list[dict]:
    """Pull every score-record dict out of a list-shaped JSON value."""
    records = value
    if isinstance(value, dict):
        for k in ("data", "list", "items", "scores", "result", "results"):
            v = value.get(k)
            if isinstance(v, list):
                records = v
                break
        # maybe the dict IS a single record
        if not isinstance(records, list):
            records = [value]
    if not isinstance(records, list):
        return []
    out = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rid = _first(rec, ID_KEYS)
        rkey = _first(rec, KEY_KEYS)
        rtitle = _first(rec, TITLE_KEYS)
        rlen = _first(rec, NOTESLEN_KEYS)
        # require at least an id and a key to be a real score record
        if rid is None or rkey is None:
            continue
        out.append({
            "score_id": rid,
            "score_key": rkey,
            "title": rtitle or "",
            "notes_length": rlen,
        })
    return out


def looks_like_score_list(value: Any) -> bool:
    records = value
    if isinstance(value, dict):
        for k in ("data", "list", "items", "scores", "result", "results"):
            v = value.get(k)
            if isinstance(v, list):
                records = v
                break
    if not isinstance(records, list) or len(records) < 2:
        return False
    hits = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if _first(rec, ID_KEYS) is not None and _first(rec, KEY_KEYS) is not None:
            hits += 1
    return hits >= max(2, len(records) // 2)


# ---- main ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse captured score lists into a batch manifest CSV."
    )
    ap.add_argument("--capture-dir", required=True,
                    help="Output dir of capture_score_lists.py (contains candidates/ + url_index.json).")
    ap.add_argument("--out-csv", required=True,
                    help="Path to write score_manifest.csv.")
    ap.add_argument("--expected", default="精选:11,热门:30,移植:9",
                    help="Expected category:count, comma-separated (for the sanity report only).")
    args = ap.parse_args()

    cap = pathlib.Path(args.capture_dir)
    cand_dir = cap / "candidates"
    url_index_path = cap / "url_index.json"

    url_index = {}
    if url_index_path.exists():
        url_index = json.loads(url_index_path.read_text(encoding="utf-8"))

    # each candidate file is one JSON value captured from a response body.
    # We also need the request URL for that body. url_index maps body file
    # name -> url, but candidates are json-value-hashed not body-hashed, so
    # we re-derive: for each candidate, scan ALL bodies for a substring match
    # to recover its url. Cheap and robust.
    bodies_dir = cap / "bodies"
    body_urls: dict[str, str] = {}      # candidate file -> url
    if bodies_dir.exists():
        # build a quick index: body file -> (text, url)
        body_index = []
        for bname, url in url_index.items():
            bpath = bodies_dir / bname
            if bpath.exists():
                body_index.append((bname, url, bpath.read_text(encoding="utf-8", errors="replace")))
        for cpath in sorted(cand_dir.glob("candidate_*.json")):
            ctext = cpath.read_text(encoding="utf-8", errors="replace")
            # use a stable distinctive slice (first 120 chars of compact form)
            try:
                cval = json.loads(ctext)
                sig = json.dumps(cval, ensure_ascii=False, separators=(",", ":"))[:200]
            except Exception:
                sig = ctext[:200]
            for bname, url, btext in body_index:
                if sig and sig[:80] in btext:
                    body_urls[cpath.name] = url
                    break

    # collect every list-shaped candidate with its (best-effort) url
    lists = []  # list of (candidate_path, url, records)
    for cpath in sorted(cand_dir.glob("candidate_*.json")):
        try:
            value = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not looks_like_score_list(value):
            continue
        url = body_urls.get(cpath.name, "")
        recs = extract_records(value)
        if recs:
            lists.append((cpath, url, recs))

    print(f"Found {len(lists)} list-shaped candidate(s).")
    for cpath, url, recs in lists:
        cat = category_from_url(url) or "(unknown)"
        print(f"  - {cpath.name}: {len(recs)} records  category={cat}  url={url[:100]}")

    # attribute category per list; flag ambiguity
    attributed = []  # (category, records)
    unresolved = []
    for cpath, url, recs in lists:
        cat = category_from_url(url)
        if cat is None:
            unresolved.append((cpath, url, recs))
        else:
            attributed.append((cat, recs))

    if unresolved:
        print("")
        print("WARNING: could not attribute category from URL for these lists:")
        for cpath, url, recs in unresolved:
            sample = recs[0] if recs else {}
            print(f"  {cpath.name}  url={url or '(none)'}  records={len(recs)}")
            print(f"    sample record keys: {list(sample.keys())[:15]}")
        print("")
        print("If the URL carries category info in a non-obvious form, inspect")
        print(f"{cap}/url_index.json and re-run after adjusting CATEGORY_KEYWORDS.")
        # do not abort -- still emit what we can, but mark unknown records.
        for cpath, url, recs in unresolved:
            attributed.append(("(unknown)", recs))

    # dedup by score_key, keeping the first category seen
    seen: dict[str, dict] = {}   # score_key -> row
    cat_counts: dict[str, int] = {}
    for cat, recs in attributed:
        cat_counts[cat] = cat_counts.get(cat, 0) + len(recs)
        for r in recs:
            key = str(r["score_key"])
            if key in seen:
                continue
            seen[key] = {
                "score_key": key,
                "score_id": r["score_id"],
                "title": r["title"],
                "category": cat,
                "notes_length": r["notes_length"] if r["notes_length"] != "" else "",
            }

    # write CSV (order: by category then original order)
    cat_order = ["精选", "热门", "移植", "(unknown)"]
    rows = sorted(seen.values(),
                  key=lambda r: (cat_order.index(r["category"]) if r["category"] in cat_order else 99,
                                 str(r["score_key"])))
    out = pathlib.Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["score_key", "score_id", "title", "category", "notes_length"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # sanity report against expected counts
    print("")
    print(f"Wrote {out}  ({len(rows)} unique scores)")
    actual = {}
    for r in rows:
        actual[r["category"]] = actual.get(r["category"], 0) + 1
    print("Counts by category:")
    expected = {}
    for pair in args.expected.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            expected[k.strip()] = int(v.strip())
    for cat in cat_order:
        if cat == "(unknown)" and actual.get(cat, 0) == 0:
            continue
        exp = expected.get(cat, "?")
        act = actual.get(cat, 0)
        flag = "" if (exp == "?" or act == exp) else "  <-- mismatch"
        print(f"  {cat:<10} expected {exp:<4} got {act}{flag}")

    if actual.get("(unknown)", 0) > 0:
        print("")
        print("NOTE: some records have unknown category. Edit url_index / keywords")
        print("or manually fix score_manifest.csv before batch export.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
