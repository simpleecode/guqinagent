#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
from typing import Any

import frida


JS = r"""
const needles = __NEEDLES__;

function shouldEmit(text) {
  if (!text) return false;
  if (text.indexOf("/v2/scores/") >= 0 || text.indexOf("notes") >= 0 || text.indexOf("jians") >= 0 || text.indexOf("sections") >= 0 || text.indexOf("dataops") >= 0) return true;
  for (const n of needles) {
    if (n && text.indexOf(String(n)) >= 0) return true;
  }
  return false;
}

function emit(kind, text, meta) {
  try {
    if (shouldEmit(text)) {
      send({
        event: "http_text",
        kind,
        meta: meta || {},
        length: text.length,
        preview: text.slice(0, 500),
        text
      });
    }
  } catch (e) {
  }
}

function byteArrayToString(bytes) {
  try {
    const StringCls = Java.use("java.lang.String");
    return StringCls.$new(bytes, "UTF-8").toString();
  } catch (e) {
    return "";
  }
}

Java.perform(function () {
  send({event: "capture_ready"});

  try {
    const Request = Java.use("okhttp3.Request");
    Request.url.implementation = function () {
      const url = this.url();
      try {
        const headers = {};
        const names = this.headers().names().toArray();
        for (let i = 0; i < names.length; i++) {
          const name = String(names[i]);
          headers[name] = String(this.header(name));
        }
        const text = String(url);
        if (shouldEmit(text) || text.indexOf("/v2/scores/") >= 0) {
          send({event: "http_request", kind: "okhttp3.Request.url", url: text, headers});
        }
      } catch (e) {
      }
      return url;
    };
    send({event: "hooked", className: "okhttp3.Request.url"});
  } catch (e) {
    send({event: "hook_failed", className: "okhttp3.Request.url", error: String(e)});
  }

  try {
    const ResponseBody = Java.use("okhttp3.ResponseBody");
    ResponseBody.string.implementation = function () {
      const text = this.string();
      emit("okhttp3.ResponseBody.string", text, {});
      return text;
    };
    ResponseBody.bytes.implementation = function () {
      const bytes = this.bytes();
      emit("okhttp3.ResponseBody.bytes", byteArrayToString(bytes), {});
      return bytes;
    };
    send({event: "hooked", className: "okhttp3.ResponseBody"});
  } catch (e) {
    send({event: "hook_failed", className: "okhttp3.ResponseBody", error: String(e)});
  }

  try {
    const Buffer = Java.use("okio.Buffer");
    Buffer.readUtf8.overload().implementation = function () {
      const text = this.readUtf8();
      emit("okio.Buffer.readUtf8", text, {});
      return text;
    };
    send({event: "hooked", className: "okio.Buffer.readUtf8"});
  } catch (e) {
    send({event: "hook_failed", className: "okio.Buffer.readUtf8", error: String(e)});
  }

  try {
    const WebView = Java.use("android.webkit.WebView");
    WebView.loadUrl.overload("java.lang.String").implementation = function (url) {
      emit("android.webkit.WebView.loadUrl", String(url), {});
      return this.loadUrl(url);
    };
    send({event: "hooked", className: "android.webkit.WebView.loadUrl"});
  } catch (e) {
    send({event: "hook_failed", className: "android.webkit.WebView.loadUrl", error: String(e)});
  }
});
"""


def run_adb(adb: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=False, text=True, capture_output=True)


def iter_json_values(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for start, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if end >= 8:
            values.append(value)
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adb", default=r"C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    ap.add_argument("--package", default="com.sitongli.app.gadget")
    ap.add_argument("--host", default="127.0.0.1:27042")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--duration", type=int, default=75)
    ap.add_argument("--needle", action="append", default=[])
    ap.add_argument("--deeplink", default="")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies_dir = out_dir / "bodies"
    candidates_dir = out_dir / "candidates"
    bodies_dir.mkdir(exist_ok=True)
    candidates_dir.mkdir(exist_ok=True)
    jsonl_path = out_dir / "http_capture.jsonl"

    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"])
    pid_result = run_adb(args.adb, ["shell", "pidof", args.package])
    if not pid_result.stdout.strip():
        raise RuntimeError(f"{args.package} is not running")
    pid = int(pid_result.stdout.strip().split()[0])

    needles = [str(n) for n in args.needle if str(n)]
    device = frida.get_device_manager().add_remote_device(args.host)
    session = device.attach(pid)
    script = session.create_script(JS.replace("__NEEDLES__", json.dumps(needles, ensure_ascii=False)))

    seen_text: set[str] = set()
    seen_json: set[str] = set()

    with jsonl_path.open("w", encoding="utf-8") as fp:
        def on_message(message, data):
            rec = {"ts": time.time(), "message": message}
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            payload = message.get("payload") if message.get("type") == "send" else None
            if not isinstance(payload, dict):
                print(json.dumps(message, ensure_ascii=False)[:1000])
                return
            print(json.dumps({k: v for k, v in payload.items() if k != "text"}, ensure_ascii=False)[:1000])
            if payload.get("event") != "http_text":
                return
            text = payload.get("text") or ""
            digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
            if digest not in seen_text:
                seen_text.add(digest)
                (bodies_dir / f"body_{len(seen_text):04d}_{digest[:12]}.txt").write_text(text, encoding="utf-8")
            for value in iter_json_values(text):
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                jd = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                if jd in seen_json:
                    continue
                seen_json.add(jd)
                (candidates_dir / f"candidate_{len(seen_json):04d}_{jd[:12]}.json").write_text(
                    json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        script.on("message", on_message)
        script.load()
        if args.deeplink:
            time.sleep(2)
            run_adb(args.adb, ["shell", "am", "start", "-W", "-a", "android.intent.action.VIEW", "-d", args.deeplink, args.package])
        print(f"capture ready for {args.duration}s; open or refresh the target score page now")
        time.sleep(args.duration)

    session.detach()
    print(f"wrote capture to {out_dir}")
    print(f"wrote {len(seen_text)} body texts and {len(seen_json)} JSON candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
