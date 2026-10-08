#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys
import time

import frida


DEFAULT_TARGETS = {
    "score_notes_sections_eJk": 0x624AEC,
    "score_jians_tIk_MYa": 0x68BB50,
    "score_jian_enum_oBk_68b5c4": 0x68B5C4,
    "score_jian_enum_oBk_68b6a8": 0x68B6A8,
    "score_jian_parse_uJk_Nab": 0x68BA74,
    "score_data_decode_mMk_qgb": 0x6C2B60,
    "native_score_ZJk_Wbb": 0x6B3E70,
    "score_open_6c4f60": 0x6C4F60,
    "score_open_6c86c4": 0x6C86C4,
    "score_open_6c8714": 0x6C8714,
}


HOOK_JS = r"""
const targets = __TARGETS__;

function reg(ctx, name) {
  try {
    return ctx[name].toString();
  } catch (e) {
    return null;
  }
}

function hookAll() {
  const lib = Process.getModuleByName("libapp.so");
  send({event: "libapp", base: lib.base.toString(), size: lib.size});
  for (const t of targets) {
    const addr = lib.base.add(ptr(t.offset));
    try {
      Interceptor.attach(addr, {
        onEnter(args) {
          this.name = t.name;
          this.offset = t.offset;
          send({
            event: "enter",
            name: t.name,
            offset: "0x" + t.offset.toString(16),
            address: addr.toString(),
            x: [
              reg(this.context, "x0"), reg(this.context, "x1"),
              reg(this.context, "x2"), reg(this.context, "x3"),
              reg(this.context, "x4"), reg(this.context, "x5"),
              reg(this.context, "x6"), reg(this.context, "x7")
            ]
          });
        },
        onLeave(retval) {
          send({
            event: "leave",
            name: this.name,
            offset: "0x" + this.offset.toString(16),
            retval: retval.toString()
          });
        }
      });
      send({event: "hooked", name: t.name, offset: "0x" + t.offset.toString(16), address: addr.toString()});
    } catch (e) {
      send({event: "hook_error", name: t.name, offset: "0x" + t.offset.toString(16), error: String(e)});
    }
  }
}

setImmediate(hookAll);
"""


def run_adb(adb: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=check, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw Frida hook for Sitongli libapp offsets.")
    parser.add_argument("--adb", default=r"C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    parser.add_argument("--package", default="com.sitongli.app.gadget")
    parser.add_argument("--host", default="127.0.0.1:27042")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"], check=False)
    pid = args.pid
    if not pid:
        proc = run_adb(args.adb, ["shell", "pidof", args.package], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            pid = int(proc.stdout.strip().split()[0])

    targets = DEFAULT_TARGETS
    if args.only:
        filters = [x.lower() for x in args.only]
        targets = {k: v for k, v in DEFAULT_TARGETS.items() if any(f in k.lower() for f in filters)}
    if not targets:
        raise SystemExit("no targets selected")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = frida.get_device_manager().add_remote_device(args.host)
    session = device.attach(pid or args.package)
    script = session.create_script(HOOK_JS.replace("__TARGETS__", json.dumps(
        [{"name": k, "offset": v} for k, v in targets.items()],
        ensure_ascii=False,
    )))

    with out.open("a", encoding="utf-8") as fp:
        def on_message(message, data):
            row = {"ts": time.time(), "message": message}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            payload = message.get("payload", message)
            print(json.dumps(payload, ensure_ascii=False), flush=True)

        script.on("message", on_message)
        script.load()
        print(f"[+] attached pid={pid} out={out}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
