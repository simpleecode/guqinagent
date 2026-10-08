"""Knowledge-backed extraction of named guqin fingering techniques."""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = (ROOT / "agents/abc_to_jianzipu/knowledge"
                  / "complex_fingering_explanations_v3_with_effects.jsonl")
FINGERING_CATEGORIES = {"右手指法", "左手指法", "特殊/左右手配合指法"}
HAND_FINGER_MARKERS = ("大指", "食指", "中指", "名指", "跪指")
# Basic strokes are usually written as single characters and are not all
# represented by compound-name entries in the knowledge file.
BASIC_STROKES = {
    "抹": "右手指法", "挑": "右手指法", "勾": "右手指法", "剔": "右手指法",
    "打": "右手指法", "摘": "右手指法", "托": "右手指法", "擘": "右手指法",
    "挑": "右手指法", "历": "右手指法", "撮": "右手指法", "滚": "右手指法",
    "拂": "右手指法", "吟": "左手指法", "猱": "左手指法", "绰": "左手指法",
    "注": "左手指法", "撞": "左手指法", "唤": "左手指法", "逗": "左手指法",
    "上": "左手指法", "下": "左手指法", "退": "左手指法", "跪": "左手指法",
    "掐": "左手指法", "滔起": "左手指法", "拨": "右手指法",
}


@lru_cache(maxsize=4)
def _lexicon(path_string: str) -> tuple[tuple[str, str, str], ...]:
    """Return (surface spelling, canonical label, category) longest first."""
    rows = []
    for line in Path(path_string).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        category = str(row.get("category") or "")
        if category not in FINGERING_CATEGORIES:
            continue
        canonical = re.split(r"\s*/\s*", str(row.get("name") or ""), maxsplit=1)[0]
        name = str(row.get("name") or "")
        aliases = [part.strip() for part in re.split(r"\s*/\s*", name) if part.strip()]
        for alias in aliases:
            rows.append((alias, canonical, category))

    # These spelling variants are established aliases in the pitch parser and
    # knowledge descriptions, although the source knowledge row lists only one.
    for alias in ("歷", "厉"):
        rows.append((alias, "历", "右手指法"))
    rows.extend((name, name, category) for name, category in BASIC_STROKES.items())
    # Common guqin score glyph variants: these are the same named techniques,
    # so all variants aggregate under the knowledge-base spelling.
    for alias, canonical, category in (
        ("泼剌", "拨剌", "右手指法"),
        ("泼", "拨", "右手指法"),
    ):
        rows.append((alias, canonical, category))

    # Keep the first canonical mapping for duplicate spellings, then prefer
    # longer names so e.g. 掐撮三声 is not split into 掐撮 + other tokens.
    unique: dict[str, tuple[str, str, str]] = {}
    for item in rows:
        unique.setdefault(item[0], item)
    # These exact spellings denote compound gestures; the alias listed in a
    # knowledge row can be explanatory rather than a separately repeated
    # action.
    unique["绰上"] = ("绰上", "绰上", "左手指法")
    unique["注下"] = ("注下", "注下", "左手指法")
    unique["退复"] = ("退复", "退复", "左手指法")
    # Add spellings after source aliases so lexical canonicalization wins for
    # 泼/泼剌 (the knowledge file's first spelling remains authoritative).
    unique["泼"] = ("泼", "拨", "右手指法")
    unique["泼剌"] = ("泼剌", "拨剌", "右手指法")
    unique["掐拨剌三声"] = ("掐拨剌三声", "掐拨剌三声", "右手指法")
    return tuple(sorted(unique.values(), key=lambda item: (-len(item[0]), item[0])))


def extract_fingering_techniques(
    text: Any,
    *,
    knowledge_path: Path = KNOWLEDGE_PATH,
) -> list[dict[str, str]]:
    """Recognize named action tokens, treating known compounds as atomic.

    Position fields (string, hui, and left-finger names) are not emitted as
    techniques. Repeated instances of one canonical technique in a single
    notation event count once; different techniques in the event are retained.
    """
    source = str(text or "").strip().strip("[]")
    lexicon = _lexicon(str(knowledge_path.resolve()))
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    position = 0
    while position < len(source):
        marker = next((item for item in HAND_FINGER_MARKERS
                       if source.startswith(item, position)), None)
        if marker:
            position += len(marker)
            continue
        match = next((item for item in lexicon
                      if source.startswith(item[0], position)), None)
        if match:
            surface, canonical, category = match
            if canonical not in seen:
                found.append({"name": canonical, "category": category, "surface": surface})
                seen.add(canonical)
            position += len(surface)
        else:
            position += 1
    return found


def count_fingering_techniques(texts: list[Any], *, knowledge_path: Path = KNOWLEDGE_PATH) -> dict[str, int]:
    """Count technique-bearing events over a sequence of notation strings."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(item["name"] for item in extract_fingering_techniques(
            text, knowledge_path=knowledge_path
        ))
    return dict(sorted(counts.items()))
