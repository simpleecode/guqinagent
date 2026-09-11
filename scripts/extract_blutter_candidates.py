#!/usr/bin/env python3
import json
import pathlib
import re


PATTERNS = [
    "notes",
    "jians",
    "sections",
    "lyric",
    "native_score",
    "/scores/:key",
    "/scores/",
    "dataops",
    "score_key",
    "score_id",
    "score_title",
    "score_open",
    "score_save",
    "update_jianzi",
]


def main():
    pp = pathlib.Path("cases/sitongli-guanshanyue/work/blutter_out/pp.txt")
    lines = pp.read_text(encoding="utf-8", errors="ignore").splitlines()
    addr_re = re.compile(r"\((0x[0-9a-fA-F]+)\)")
    out = []
    all_addrs = {}
    for i, line in enumerate(lines):
        matched = [p for p in PATTERNS if f'String: "{p}"' in line]
        if not matched:
            continue
        ctx = lines[max(0, i - 16): min(len(lines), i + 17)]
        addrs = []
        for c in ctx:
            addrs.extend(addr_re.findall(c))
        addrs = sorted(set(addrs), key=lambda x: int(x, 16))
        for a in addrs:
            all_addrs.setdefault(a, set()).update(matched)
        out.append({
            "line": i + 1,
            "hit": line.strip(),
            "patterns": matched,
            "addrs": addrs,
            "context": ctx,
        })
    summary = [{"offset": a, "patterns": sorted(v)} for a, v in sorted(all_addrs.items(), key=lambda kv: int(kv[0], 16))]
    pathlib.Path("cases/sitongli-guanshanyue/evidence/blutter_score_candidates.json").write_text(
        json.dumps({"summary": summary, "matches": out}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
