#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import time

import frida


JS = r"""
const scans = __SCANS__;
const maxHitsPerNeedle = __MAX_HITS__;
const contextBytes = __CONTEXT_BYTES__;

function hexBytes(bytes) {
  return Array.prototype.map.call(bytes, x => ("0" + x.toString(16)).slice(-2)).join(" ");
}

function safeRead(addr, len) {
  try {
    const buf = addr.readByteArray(len);
    return buf ? Array.from(new Uint8Array(buf)) : [];
  } catch (e) {
    return [];
  }
}

function safeUtf8(bytes) {
  try {
    return new TextDecoder("utf-8", {fatal: false}).decode(new Uint8Array(bytes));
  } catch (e) {
    return "";
  }
}

function safeUtf16(bytes) {
  try {
    return new TextDecoder("utf-16le", {fatal: false}).decode(new Uint8Array(bytes));
  } catch (e) {
    return "";
  }
}

function emitHit(needle, encoding, addr, range) {
  const start = ptr(addr).sub(contextBytes);
  const bytes = safeRead(start, contextBytes * 2);
  send({
    event: "hit",
    needle,
    encoding,
    address: ptr(addr).toString(),
    rangeBase: range.base.toString(),
    rangeSize: range.size,
    utf8: safeUtf8(bytes),
    utf16le: safeUtf16(bytes),
    hex: hexBytes(bytes)
  });
}

function scanOne(needle, encoding, pattern, ranges) {
  let hits = 0;
  for (const range of ranges) {
    if (hits >= maxHitsPerNeedle) break;
    try {
      Memory.scanSync(range.base, range.size, pattern).forEach(m => {
        if (hits < maxHitsPerNeedle) {
          emitHit(needle, encoding, m.address, range);
          hits++;
        }
      });
    } catch (e) {
    }
  }
  send({event: "needle_done", needle, encoding, hits});
}

setImmediate(function () {
  const ranges = Process.enumerateRanges({protection: "r--", coalesce: true})
    .concat(Process.enumerateRanges({protection: "rw-", coalesce: true}))
    .filter(r => r.size > 0 && r.size < 128 * 1024 * 1024);
  send({event: "ranges", count: ranges.length});
  for (const scan of scans) {
    scanOne(scan.needle, scan.encoding, scan.pattern, ranges);
  }
  send({event: "done"});
});
"""


def run_adb(adb: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=False, text=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb", default=r"C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    ap.add_argument("--package", default="com.sitongli.app.gadget")
    ap.add_argument("--host", default="127.0.0.1:27042")
    ap.add_argument("--out", default=r"cases\sitongli-guanshanyue\evidence\memory_string_hits.jsonl")
    ap.add_argument("--max-hits", type=int, default=40)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--needle", action="append", default=[])
    args = ap.parse_args()

    needles = args.needle or ["SSG54sm8", "关山月", "score_id", "score_key", "score_title", "notes", "jians", "sections", "dataops", "jian"]
    scans = []
    for needle in needles:
        scans.append({"needle": needle, "encoding": "utf8", "pattern": needle.encode("utf-8").hex(" ")})
        scans.append({"needle": needle, "encoding": "utf16le", "pattern": needle.encode("utf-16le").hex(" ")})
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"])
    pid_result = run_adb(args.adb, ["shell", "pidof", args.package])
    if not pid_result.stdout.strip():
      raise RuntimeError(f"{args.package} is not running")
    pid = int(pid_result.stdout.strip().split()[0])

    device = frida.get_device_manager().add_remote_device(args.host)
    session = device.attach(pid)
    script = session.create_script(
        JS.replace("__SCANS__", json.dumps(scans, ensure_ascii=False))
        .replace("__MAX_HITS__", str(args.max_hits))
        .replace("__CONTEXT_BYTES__", str(args.context))
    )
    done = False
    with out.open("w", encoding="utf-8") as fp:
        def on_message(message, data):
            nonlocal done
            rec = {"ts": time.time(), "message": message}
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            payload = message.get("payload") if message.get("type") == "send" else None
            if payload:
                print(json.dumps(payload, ensure_ascii=False)[:1000])
                if payload.get("event") == "done":
                    done = True
            else:
                print(json.dumps(message, ensure_ascii=False))

        script.on("message", on_message)
        script.load()
        deadline = time.time() + 60
        while not done and time.time() < deadline:
            time.sleep(0.2)
    session.detach()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
