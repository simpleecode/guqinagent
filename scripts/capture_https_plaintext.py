#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture plaintext HTTPS traffic from the Sitongli app by hooking libssl's
SSL_write / SSL_read.  This works regardless of the app being Flutter/Dart
(no Java bridge, no Dart-symbol offsets needed) because every outbound HTTPS
request and every inbound response funnels through BoringSSL's SSL_* APIs.

Primary use cases:
  * recover a live Authorization token (JWT) from outbound request headers
  * observe the exact URL / query / body of the score-list & score-detail APIs
  * capture JSON response bodies for offline parsing

Output: <out-dir>/ssl_capture.jsonl  (one record per SSL_write/SSL_read call)
        <out-dir>/bodies/*.txt       (one file per unique body, deduped)
        <out-dir>/tokens.txt         (JWT-shaped strings seen, most-recent first)
        <out-dir>/requests.log       (human-readable request line + auth header)

Operator flow:
  1. run this script
  2. in the app, perform the action whose traffic you want captured
     (e.g. open a score, open a category list)
  3. stop with Ctrl-C or wait for --duration
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import time
from typing import Any

import frida


JS = r"""
// Resolve SSL_write / SSL_read from EVERY module that exports them.
// The Sitongli app uses Cronet (Chromium network stack), whose TLS runs
// through stable_cronet_libssl.so -- NOT the system libssl.so.  So we must
// hook SSL_write/SSL_read in every SSL-providing module, not just one.
function resolveSsl() {
  const out = {SSL_write: [], SSL_read: []};
  const mods = Process.enumerateModules();
  for (const m of mods) {
    try {
      const exps = m.enumerateExports();
      for (const e of exps) {
        if (e.name === "SSL_write") {
          out.SSL_write.push({addr: e.address, module: m.name});
        } else if (e.name === "SSL_read") {
          out.SSL_read.push({addr: e.address, module: m.name});
        }
      }
    } catch (err) {}
  }
  return out;
}

function bytesToHex(buf) {
  // Robust across frida runtimes (V8/QuickJS): build a hex string byte by byte.
  // send() can't carry raw binary reliably in the gadget QuickJS runtime, so
  // we transport as hex and decode on the python side.
  try {
    const arr = new Uint8Array(buf);
    let out = "";
    for (let i = 0; i < arr.length; i++) {
      out += ("0" + arr[i].toString(16)).slice(-2);
    }
    return out;
  } catch (e) {
    return "";
  }
}

const ssl = resolveSsl();
send({
  event: "ssl_resolved",
  ssl_write: ssl.SSL_write.map(v => v.module + "@" + v.addr),
  ssl_read:  ssl.SSL_read.map(v => v.module + "@" + v.addr)
});

function hookSslEntry(sym, entry, isWrite) {
  try {
    Interceptor.attach(entry.addr, {
      onEnter: function (args) {
        // int SSL_write(SSL *ssl, const void *buf, int num)
        // int SSL_read (SSL *ssl, void *buf, int num)
        this.buf = args[1];
        this.num = args[2].toInt32();
      },
      onLeave: function (retval) {
        const n = retval.toInt32();
        if (n <= 0) return;            // error / would-block
        const take = Math.min(n, this.num, 65536);
        if (take <= 0) return;
        try {
          const data = this.buf.readByteArray(take);
          send({
            event: "ssl_data",
            dir: isWrite ? "out" : "in",
            sym: sym,
            module: entry.module,
            len: take,
            hex: bytesToHex(data)
          });
        } catch (e) {}
      }
    });
    send({event: "hooked", sym: sym, module: entry.module, at: entry.addr.toString()});
  } catch (e) {
    send({event: "hook_err", sym: sym, module: entry.module, err: String(e)});
  }
}

// hook every SSL_write / SSL_read across all modules (system libssl AND cronet)
let hookCount = 0;
for (const entry of ssl.SSL_write) { hookSslEntry("SSL_write", entry, true);  hookCount++; }
for (const entry of ssl.SSL_read)  { hookSslEntry("SSL_read",  entry, false); hookCount++; }
send({event: "ready", hooks: hookCount});
"""


JWT_RE = re.compile(rb"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")


def run_adb(adb: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=False, text=True, capture_output=True)


def hexdecode(hexstr: str) -> bytes:
    return bytes.fromhex(hexstr) if hexstr else b""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Capture Sitongli app HTTPS plaintext via SSL_write/SSL_read hooks."
    )
    ap.add_argument(
        "--adb",
        default=r"C:\Users\30343\AppData\Local\Microsoft\WinGet\Packages\Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe\platform-tools\adb.exe",
    )
    ap.add_argument("--package", default="com.sitongli.app.gadget")
    ap.add_argument("--host", default="127.0.0.1:27042")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--duration", type=int, default=60)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies_dir = out_dir / "bodies"
    bodies_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / "ssl_capture.jsonl"
    tokens_path = out_dir / "tokens.txt"
    reqlog_path = out_dir / "requests.log"

    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"])
    pid_result = run_adb(args.adb, ["shell", "pidof", args.package])
    if not pid_result.stdout.strip():
        raise RuntimeError(f"{args.package} is not running")
    pid = int(pid_result.stdout.strip().split()[0])

    device = frida.get_device_manager().add_remote_device(args.host)
    session = device.attach(pid)
    script = session.create_script(JS)

    seen_bodies: set[str] = set()
    tokens: list[str] = []          # ordered, most-recent first prepended
    token_set: set[str] = set()
    request_lines: list[str] = []
    # accumulate ssl_read across calls so we can parse HTTP responses even when
    # the body is delivered in several SSL_read invocations.
    in_buf = bytearray()

    reqlog_fp = reqlog_path.open("w", encoding="utf-8")

    def handle_data(dir_: str, hexstr: str, length: int):
        nonlocal in_buf
        try:
            raw = hexdecode(hexstr)
        except Exception:
            return
        if not raw:
            return

        # dedup whole bodies
        digest = hashlib.sha256(raw).hexdigest()
        body_name = None
        if digest not in seen_bodies:
            seen_bodies.add(digest)
            body_name = f"{dir_}_{len(seen_bodies):04d}_{digest[:10]}.txt"
            (bodies_dir / body_name).write_bytes(raw)

        # JWT hunt (both directions; tokens appear in outbound Authorization
        # headers and sometimes echoed back in responses)
        for m in JWT_RE.findall(raw):
            tok = m.decode("ascii", "replace")
            if tok not in token_set:
                token_set.add(tok)
                tokens.insert(0, tok)
                with tokens_path.open("a", encoding="utf-8") as tf:
                    tf.write(tok + "\n")

        # request-line + Authorization capture from outbound HTTP requests
        if dir_ == "out":
            # an HTTP request begins with a method line; headers follow until \r\n\r\n
            head = raw[:512]
            if re.match(rb"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) ", head):
                first = head.split(b"\r\n", 1)[0].decode("ascii", "replace")
                auth = b""
                for line in raw.split(b"\r\n"):
                    if line.lower().startswith(b"authorization:"):
                        auth = line
                        break
                rec = f"{first}   |   {auth.decode('ascii','replace')}"
                request_lines.append(rec)
                reqlog_fp.write(rec + "\n")
                reqlog_fp.flush()
                print(f"[req] {first}")
                if auth:
                    print(f"       {auth.decode('ascii','replace')[:90]}...")
        else:
            # inbound: just print status line for visibility
            head = raw[:80]
            if re.match(rb"^HTTP/\d", head):
                status = head.split(b"\r\n", 1)[0].decode("ascii", "replace")
                print(f"[resp] {status}")

    stats = {"out": 0, "in": 0}
    with jsonl_path.open("w", encoding="utf-8") as fp:
        def on_message(message, data):
            rec = {"ts": time.time(), "message": message}
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            if message.get("type") != "send":
                print(json.dumps(message, ensure_ascii=False)[:300])
                return
            p = message.get("payload") or {}
            ev = p.get("event")
            if ev == "ssl_data":
                d = p.get("dir")
                handle_data(d, p.get("hex", ""), int(p.get("len", 0)))
                stats[d] = stats.get(d, 0) + 1
            else:
                # lifecycle / hook events
                print(json.dumps({k: v for k, v in p.items()}, ensure_ascii=False)[:200])

        script.on("message", on_message)
        script.load()
        print("")
        print("=" * 64)
        print("SSL hooks live. NOW in the app, trigger the traffic you want:")
        print("  - to get a token + see score API: open any one score")
        print("  - to capture category lists: open 精选 / 热门 / 移植 tabs")
        print(f"Capturing for {args.duration}s. Ctrl-C to stop early.")
        print("=" * 64)
        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\n(interrupted)")

    reqlog_fp.close()
    session.detach()
    print("")
    print(f"out packets: {stats['out']}   in packets: {stats['in']}")
    print(f"unique bodies: {len(seen_bodies)} -> {bodies_dir}")
    print(f"requests logged: {len(request_lines)} -> {reqlog_path}")
    print(f"tokens found: {len(tokens)} -> {tokens_path}")
    if tokens:
        print(f"most recent token: {tokens[0][:60]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
