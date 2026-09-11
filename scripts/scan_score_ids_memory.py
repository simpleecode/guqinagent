#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan the Sitongli app's memory for score list records and extract every
(score_id, score_key, title) triple we can find.

List records live in the Dart heap as UTF-16LE strings.  We scan for the
8-char base64-url-ish score_key pattern (the app's `key`/`score_key` field,
e.g. "SYuY17FF", "SCf7VJzZ") in both UTF-8 and UTF-16LE, then for each hit
read a wide context window and try to recover the neighboring numeric id
and title.

Output: <out>/score_ids_found.json  (deduped list of {id,key,title,addr})
        <out>/score_ids_found.csv  (same, csv)
"""
import argparse
import base64
import json
import pathlib
import re
import subprocess
import time

import frida


# score_key is an 8-char base62 string (alnum).  We anchor on a known one
# to calibrate, then sweep for the general pattern.
JS = r"""
const maxHits = __MAX_HITS__;
const ctx = __CONTEXT_BYTES__;

// Pattern: 8-char [A-Za-z0-9] run.  We over-match and filter on the python
// side, because frida Memory.scan wants a hex pattern, not a regex.
// Encode the regex as a scan over a sentinel we know precedes keys: in Dart
// heap, a score_key string of length 8 is stored with a length tag of 0x10
// (8 << 1, Smi tag) right before the ASCII bytes.  But that's fragile.
//
// Simpler + robust: scan for the ASCII bytes of every KNOWN key the operator
// passes via --anchor, PLUS a generic 8-char alnum sweep is too expensive.
// So we rely on anchors: the caller feeds several known score_keys; we scan
// for each and dump context.  Then we ALSO scan for the bytes of '"key":"'
// (UTF-8) and the UTF-16LE equivalent to catch serialized JSON lists.
const anchors = __ANCHORS__;
const scans = [];

// 1. each anchor key, utf8 + utf16le
for (const a of anchors) {
  scans.push({needle: a, encoding: 'utf8', pattern: a.split('').map(c=>c.charCodeAt(0).toString(16).padStart(2,'0')).join(' ')});
  let u16 = '';
  for (const c of a) { u16 += (c.charCodeAt(0).toString(16).padStart(2,'0')) + ' 00'; }
  scans.push({needle: a, encoding: 'utf16le', pattern: u16});
}
// 2. serialized-json markers (catch list JSON if present in memory)
const jsonMarks = ['"key":"', '"score_key":"', '"id":', '"title":"'];
for (const m of jsonMarks) {
  scans.push({needle: m, encoding: 'utf8', pattern: m.split('').map(c=>c.charCodeAt(0).toString(16).padStart(2,'0')).join(' ')});
  let u16 = '';
  for (const c of m) { u16 += (c.charCodeAt(0).toString(16).padStart(2,'0')) + ' 00'; }
  scans.push({needle: m, encoding: 'utf16le', pattern: u16});
}

function hexBytes(bytes){return Array.prototype.map.call(bytes,x=>('0'+x.toString(16)).slice(-2)).join(' ');}

setImmediate(function(){
  const ranges = Process.enumerateRanges({protection:'r--',coalesce:true})
    .concat(Process.enumerateRanges({protection:'rw-',coalesce:true}))
    .filter(r => r.size>0 && r.size<128*1024*1024);
  send({event:'ranges', count: ranges.length, scans: scans.length});
  for (const s of scans) {
    let hits = 0;
    for (const r of ranges) {
      if (hits >= maxHits) break;
      try {
        Memory.scanSync(r.base, r.size, s.pattern).forEach(m => {
          if (hits < maxHits) {
            try {
              const start = m.address.sub(ctx);
              const bytes = start.readByteArray(ctx*2);
              send({event:'hit', needle:s.needle, enc:s.encoding, addr:m.address.toString(),
                    hex: hexBytes(new Uint8Array(bytes))});
            } catch(e){}
            hits++;
          }
        });
      } catch(e){}
    }
    send({event:'needle_done', needle:s.needle, enc:s.encoding, hits});
  }
  send({event:'done'});
});
"""


# 8-char base62 score_key pattern
KEY_RE = re.compile(rb'[A-Za-z0-9]{8}')


def parse_context(hexstr: str) -> bytes:
    return bytes.fromhex(hexstr) if hexstr else b""


def extract_around(raw: bytes, anchor: str, encoding: str) -> dict:
    """Given a context window containing `anchor`, try to recover id/key/title."""
    info = {"key": anchor}
    # find neighbor id: a 5-7 digit number near the key
    # in Dart heap (utf16le), ascii digits are stored as XX 00
    if encoding == "utf16le":
        # decode the whole window as utf16le for readable parts
        try:
            text = raw.decode("utf-16-le", errors="replace")
        except Exception:
            text = ""
    else:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = ""
    info["context_text"] = text
    # ids: look for 5-7 digit runs
    ids = re.findall(r"\b(\d{5,7})\b", text)
    info["nearby_ids"] = ids[:5]
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan app memory for score list records.")
    ap.add_argument("--adb", default=r"C:\Users\30343\AppData\Local\Microsoft\WinGet\Packages\Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe\platform-tools\adb.exe")
    ap.add_argument("--package", default="com.sitongli.app.gadget")
    ap.add_argument("--host", default="127.0.0.1:27042")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--max-hits", type=int, default=40)
    ap.add_argument("--anchor", action="append", default=[],
                    help="Known score_key(s) to anchor the scan, e.g. SYuY17FF.")
    args = ap.parse_args()

    anchors = args.anchor or ["SYuY17FF", "SCf7VJzZ", "SSG54sm8", "SWDFmDZX"]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([args.adb, "forward", "tcp:27042", "tcp:27042"], check=False, capture_output=True)
    pid_res = subprocess.run([args.adb, "shell", "pidof", args.package], check=False, capture_output=True, text=True)
    if not pid_res.stdout.strip():
        raise RuntimeError(f"{args.package} not running")
    pid = int(pid_res.stdout.strip().split()[0])

    device = frida.get_device_manager().add_remote_device(args.host)
    session = device.attach(pid)
    script = session.create_script(
        JS.replace("__MAX_HITS__", str(args.max_hits))
          .replace("__CONTEXT_BYTES__", str(args.context))
          .replace("__ANCHORS__", json.dumps(anchors))
    )

    hits = []
    done = False
    def on_message(message, data):
        nonlocal done
        if message.get("type") != "send":
            print(json.dumps(message, ensure_ascii=False)[:200])
            return
        p = message.get("payload") or {}
        ev = p.get("event")
        if ev == "hit":
            hits.append(p)
        elif ev in ("ranges", "needle_done"):
            print(json.dumps(p, ensure_ascii=False)[:160])
        elif ev == "done":
            done = True

    script.on("message", on_message)
    script.load()
    deadline = time.time() + 180
    while not done and time.time() < deadline:
        time.sleep(0.3)
    session.detach()

    # build jsonl of raw hits
    raw_path = out.parent / (out.stem + "_raw.jsonl")
    with raw_path.open("w", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    # analyze: for each hit, extract id/key/title from context
    records = []
    seen_keys = set()
    for h in hits:
        raw = parse_context(h.get("hex", ""))
        info = extract_around(raw, h["needle"], h["enc"])
        # try to find OTHER 8-char alnum keys near this anchor (sibling list records)
        for m in KEY_RE.findall(raw):
            k = m.decode("ascii", "replace")
            if k in seen_keys:
                continue
            # filter: must look like a score_key (mix of upper+digit typically)
            if len(k) == 8 and any(c.isdigit() for c in k) and any(c.isalpha() for c in k):
                seen_keys.add(k)
                records.append({"key": k, "found_near": h["needle"], "addr": h.get("addr"),
                                "nearby_ids": info["nearby_ids"]})

    # also pull ids from all contexts
    all_ids = set()
    for h in hits:
        raw = parse_context(h.get("hex", ""))
        try:
            text = raw.decode("utf-16-le", errors="replace") if h["enc"] == "utf16le" else raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        for m in re.findall(r"\b(\d{6,7})\b", text):
            all_ids.add(m)

    result = {
        "anchors": anchors,
        "hit_count": len(hits),
        "score_keys_found": sorted(seen_keys),
        "score_ids_seen": sorted(all_ids),
        "records": records,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== RESULTS -> {out}")
    print(f"  raw hits: {len(hits)}  (raw -> {raw_path})")
    print(f"  score_keys found: {len(seen_keys)}")
    for k in sorted(seen_keys)[:30]:
        print(f"    {k}")
    print(f"  numeric ids (6-7 digit) seen: {len(all_ids)}")
    for i in sorted(all_ids)[:30]:
        print(f"    {i}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
