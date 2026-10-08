#!/usr/bin/env python3
import argparse
import base64
import json
import pathlib
import sys
import time

import frida


JS = r"""
const TARGET_HINTS = ["scores", "/scores", "score_", "scoreId", "score_id", "scoreKey", "score_key", "sections", "notes", "jians", "jian", "lyric", "关山月"];

function now() { return (new Date()).toISOString(); }

function looksInteresting(s) {
  if (!s) return false;
  for (const h of TARGET_HINTS) {
    if (s.indexOf(h) !== -1) return true;
  }
  return false;
}

function sendText(kind, meta, text) {
  const clipped = text.length > 262144 ? text.slice(0, 262144) : text;
  send({type: kind, ts: now(), meta: meta || {}, text: clipped, length: text.length});
}

function readUtf8(ptr, len) {
  try {
    const n = Math.min(len, 262144);
    const bytes = Memory.readByteArray(ptr, n);
    if (!bytes) return "";
    return new TextDecoder("utf-8", {fatal: false}).decode(bytes);
  } catch (e) {
    return "";
  }
}

function hookExport(moduleName, exportName, callbacks) {
  const addr = Module.findExportByName(moduleName, exportName);
  if (!addr) return false;
  Interceptor.attach(addr, callbacks);
  send({type: "hook", ts: now(), meta: {module: moduleName || "*", export: exportName, address: addr.toString()}});
  return true;
}

function hookSslExports() {
  const seen = {};
  const names = ["SSL_write", "SSL_read", "SSL_write_ex", "SSL_read_ex"];
  for (const m of Process.enumerateModules()) {
    for (const n of names) {
      let key = m.name + "!" + n;
      if (seen[key]) continue;
      let addr = null;
      try { addr = Module.findExportByName(m.name, n); } catch (e) {}
      if (!addr) continue;
      seen[key] = true;
      if (n === "SSL_write") {
        Interceptor.attach(addr, {
          onEnter(args) {
            const text = readUtf8(args[1], args[2].toInt32());
            if (looksInteresting(text)) sendText("ssl_write", {module: m.name, len: args[2].toInt32()}, text);
          }
        });
      } else if (n === "SSL_read") {
        Interceptor.attach(addr, {
          onEnter(args) { this.buf = args[1]; },
          onLeave(retval) {
            const nread = retval.toInt32();
            if (nread > 0) {
              const text = readUtf8(this.buf, nread);
              if (looksInteresting(text) || text.indexOf("{") !== -1) sendText("ssl_read", {module: m.name, len: nread}, text);
            }
          }
        });
      } else if (n === "SSL_write_ex") {
        Interceptor.attach(addr, {
          onEnter(args) {
            const text = readUtf8(args[1], args[2].toInt32());
            if (looksInteresting(text)) sendText("ssl_write_ex", {module: m.name, len: args[2].toInt32()}, text);
          }
        });
      } else if (n === "SSL_read_ex") {
        Interceptor.attach(addr, {
          onEnter(args) { this.buf = args[1]; this.outLen = args[3]; },
          onLeave(retval) {
            if (retval.toInt32() === 1) {
              let nread = 0;
              try { nread = Memory.readULong(this.outLen).toNumber(); } catch (e) {}
              if (nread > 0) {
                const text = readUtf8(this.buf, nread);
                if (looksInteresting(text) || text.indexOf("{") !== -1) sendText("ssl_read_ex", {module: m.name, len: nread}, text);
              }
            }
          }
        });
      }
      send({type: "hook", ts: now(), meta: {module: m.name, export: n, address: addr.toString()}});
    }
  }
}

function hookJava() {
  if (!Java.available) return;
  Java.perform(function () {
    try {
      const Log = Java.use("android.util.Log");
      Log.d.overload("java.lang.String", "java.lang.String").implementation = function (tag, msg) {
        if (looksInteresting(String(tag)) || looksInteresting(String(msg))) {
          sendText("android_log_d", {tag: String(tag)}, String(msg));
        }
        return this.d(tag, msg);
      };
    } catch (e) {}

    try {
      const SPImpl = Java.use("android.app.SharedPreferencesImpl");
      SPImpl.getString.implementation = function (key, defValue) {
        const v = this.getString(key, defValue);
        const k = String(key);
        const s = v ? String(v) : "";
        if (looksInteresting(k) || looksInteresting(s) || k.toLowerCase().indexOf("token") !== -1) {
          sendText("sharedpref_getString", {key: k}, s);
        }
        return v;
      };
    } catch (e) {}

    try {
      const SQLiteDatabase = Java.use("android.database.sqlite.SQLiteDatabase");
      SQLiteDatabase.rawQuery.overload("java.lang.String", "[Ljava.lang.String;").implementation = function (sql, args) {
        const s = String(sql);
        if (looksInteresting(s)) sendText("sqlite_rawQuery", {}, s);
        return this.rawQuery(sql, args);
      };
      SQLiteDatabase.execSQL.overload("java.lang.String").implementation = function (sql) {
        const s = String(sql);
        if (looksInteresting(s)) sendText("sqlite_execSQL", {}, s);
        return this.execSQL(sql);
      };
    } catch (e) {}
  });
}

hookJava();
setTimeout(hookSslExports, 1000);
setInterval(hookSslExports, 3000);
send({type: "ready", ts: now(), meta: {pid: Process.id}});
"""


def main():
    ap = argparse.ArgumentParser(description="Capture Sitongli score traffic/state with Frida.")
    ap.add_argument("--package", default="com.sitongli.app")
    ap.add_argument("--pid", type=int, default=None, help="Attach to a specific PID.")
    ap.add_argument("--output", default="../evidence/frida_capture.jsonl")
    ap.add_argument("--duration", type=int, default=90)
    ap.add_argument("--spawn", action="store_true", help="Spawn the app instead of attaching to a running process.")
    args = ap.parse_args()

    out = pathlib.Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    device = frida.get_usb_device(timeout=10)

    pid = None
    if args.pid is not None:
        session = device.attach(args.pid)
    elif args.spawn:
        pid = device.spawn([args.package])
        session = device.attach(pid)
    else:
        session = device.attach(args.package)

    script = session.create_script(JS)

    def on_message(message, data):
        rec = {"frida_message": message}
        if data:
            rec["data_b64"] = base64.b64encode(data).decode("ascii")
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        payload = message.get("payload")
        if isinstance(payload, dict):
            t = payload.get("type")
            if t in {"ready", "hook", "ssl_read", "ssl_write", "ssl_read_ex", "ssl_write_ex"}:
                meta = payload.get("meta", {})
                text = payload.get("text", "")
                print(f"[{t}] {meta} {text[:160].replace(chr(10), ' ')}")
        elif message.get("type") == "error":
            print(message, file=sys.stderr)

    script.on("message", on_message)
    script.load()
    if pid is not None:
        device.resume(pid)

    print(f"capturing {args.package} for {args.duration}s -> {out}")
    deadline = time.time() + args.duration
    try:
        while time.time() < deadline:
            time.sleep(0.5)
    finally:
        try:
            script.unload()
        finally:
            session.detach()


if __name__ == "__main__":
    main()
