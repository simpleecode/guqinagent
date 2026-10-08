#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import time

import frida


HOOK_JS = r"""
const SelectedTargets = __TARGETS__;
const MaxDump = __MAX_DUMP__;

function readCompressedTaggedPointer(slot) {
  return ptr(slot.readU32());
}

function decodeTaggedBrief(raw) {
  try {
    const [tptr, cls, value] = getTaggedObjectValue(raw, 2);
    const out = {
      raw: raw.toString(),
      tagged: tptr.toString(),
      className: cls.name,
      classId: cls.id
    };
    if (cls.id === CidString || cls.id === CidTwoByteString || cls.id === CidSmi ||
        cls.id === CidMint || cls.id === CidBool || cls.id === CidDouble || cls.id === CidNull) {
      out.value = value;
    } else if (cls.id === 115 || cls.name.indexOf("Uint8") >= 0) {
      out.uint8 = dumpUint8(tptr.sub(1), cls);
    } else if (cls.id === 89 || cls.id === 91) {
      out.valueType = "list_like";
      out.value = summarizeValue(value, 3);
      out.findings = collectInteresting(value, []);
    } else if (cls.id === 85) {
      out.valueType = "map_like";
      out.value = summarizeValue(value, 3);
      out.findings = collectInteresting(value, []);
    } else {
      out.value = summarizeValue(value, 2);
      out.findings = collectInteresting(value, []);
    }
    return out;
  } catch (e) {
    return { raw: raw ? raw.toString() : null, error: String(e) };
  }
}

function summarizeValue(value, depth) {
  if (depth <= 0) return "[max-depth]";
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    const n = Math.min(value.length, 24);
    const out = [];
    for (let i = 0; i < n; i++) out.push(summarizeValue(value[i], depth - 1));
    if (value.length > n) out.push({truncated: value.length - n});
    return out;
  }
  if (typeof value === "object") {
    const out = {};
    let i = 0;
    for (const k in value) {
      if (i++ >= 48) {
        out.__truncated__ = true;
        break;
      }
      out[k] = summarizeValue(value[k], depth - 1);
    }
    return out;
  }
  return String(value);
}

function bytesFromNumberArray(arr) {
  if (!Array.isArray(arr) || arr.length < 16) return null;
  const n = Math.min(arr.length, MaxDump);
  const out = [];
  for (let i = 0; i < n; i++) {
    if (typeof arr[i] !== "number" || arr[i] < 0 || arr[i] > 255) return null;
    out.push(arr[i] & 255);
  }
  return out;
}

function scoreByteArray(arr) {
  if (!arr) return null;
  let printable = 0;
  let ascii = "";
  let hex = [];
  for (let i = 0; i < Math.min(arr.length, 512); i++) {
    const b = arr[i] & 255;
    if ((b >= 32 && b <= 126) || b === 9 || b === 10 || b === 13) printable++;
  }
  for (let i = 0; i < Math.min(arr.length, 256); i++) {
    const b = arr[i] & 255;
    ascii += (b >= 32 && b <= 126) ? String.fromCharCode(b) : ".";
    hex.push(("0" + b.toString(16)).slice(-2));
  }
  return {len: arr.length, printable512: printable, ascii, hex: hex.join(" ")};
}

function collectInteresting(value, path) {
  const findings = [];
  function walk(v, p, depth) {
    if (depth <= 0 || findings.length > 64) return;
    if (typeof v === "string") {
      if (v.indexOf("notes") >= 0 || v.indexOf("jians") >= 0 || v.indexOf("/data") >= 0 || v.indexOf("秋风词") >= 0 || v.indexOf("161749") >= 0) {
        findings.push({path: p.join("."), string: v.slice(0, 800)});
      }
      return;
    }
    const arr = bytesFromNumberArray(v);
    if (arr) {
      const score = scoreByteArray(arr);
      if (score.printable512 > 120 || score.ascii.indexOf("notes") >= 0 || score.ascii.indexOf("jians") >= 0 || score.ascii.indexOf("{") >= 0) {
        findings.push({path: p.join("."), bytes: score});
      }
      return;
    }
    if (Array.isArray(v)) {
      const n = Math.min(v.length, 64);
      for (let i = 0; i < n; i++) walk(v[i], p.concat([String(i)]), depth - 1);
      return;
    }
    if (v && typeof v === "object") {
      for (const k in v) walk(v[k], p.concat([k]), depth - 1);
    }
  }
  walk(value, path, 5);
  return findings;
}

function dumpUint8(obj, cls) {
  try {
    const len = obj.add(cls.lenOffset).readU32() >> 1;
    const n = Math.min(len, MaxDump);
    const dataPtr = obj.add(cls.dataOffset);
    const bytes = Memory.readByteArray(dataPtr, n);
    const u8 = new Uint8Array(bytes);
    let hex = [];
    let ascii = "";
    for (let i = 0; i < u8.length; i++) {
      hex.push(("0" + u8[i].toString(16)).slice(-2));
      ascii += (u8[i] >= 32 && u8[i] <= 126) ? String.fromCharCode(u8[i]) : ".";
    }
    return {
      len: len,
      dumped: n,
      data: dataPtr.toString(),
      hex: hex.join(" "),
      b64: base64ArrayBuffer(bytes),
      ascii
    };
  } catch (e) {
    return {error: String(e)};
  }
}

function base64ArrayBuffer(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const enc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let out = "";
  let i;
  for (i = 0; i + 2 < bytes.length; i += 3) {
    out += enc[bytes[i] >> 2];
    out += enc[((bytes[i] & 3) << 4) | (bytes[i + 1] >> 4)];
    out += enc[((bytes[i + 1] & 15) << 2) | (bytes[i + 2] >> 6)];
    out += enc[bytes[i + 2] & 63];
  }
  if (i < bytes.length) {
    out += enc[bytes[i] >> 2];
    if (i + 1 < bytes.length) {
      out += enc[((bytes[i] & 3) << 4) | (bytes[i + 1] >> 4)];
      out += enc[(bytes[i + 1] & 15) << 2];
      out += "=";
    } else {
      out += enc[(bytes[i] & 3) << 4];
      out += "==";
    }
  }
  return out;
}

function decodeRegs(ctx) {
  const regs = {};
  for (const name of ["x0","x1","x2","x3","x4","x5","x6","x7"]) {
    regs[name] = decodeTaggedBrief(ctx[name]);
  }
  return regs;
}

function hookOne(name, offset) {
  const target = libapp.add(ptr(offset));
  Interceptor.attach(target, {
    onEnter: function(args) {
      init(this.context);
      this.name = name;
      this.offset = offset;
      send({event: "enter", name, offset: "0x" + offset.toString(16), address: target.toString(), regs: decodeRegs(this.context)});
    },
    onLeave: function(retval) {
      send({event: "leave", name: this.name, offset: "0x" + this.offset.toString(16), retval: decodeTaggedBrief(retval)});
    }
  });
  send({event: "hooked", name, offset: "0x" + offset.toString(16), address: target.toString()});
}

function onLibappLoaded() {
  send({event: "libapp_loaded", base: libapp.toString()});
  for (const target of SelectedTargets) hookOne(target.name, target.offset);
}
"""


TARGETS = {
    "score_data_decode_mMk_qgb": 0x6C2B60,
    "score_dataops_fkk_Lz": 0x6C53E4,
    "score_patch_jkk_Pz": 0x6C59D0,
    "score_save_or_dataops_mLk_leb": 0x95E578,
    "native_score_ZJk_Wbb": 0x6B3E70,
    "score_open_qJk_Iab_6c4f60": 0x6C4F60,
    "score_open_mKk_Qvh_6c86c4": 0x6C86C4,
    "score_open_mKk_Qvh_inner_6c8714": 0x6C8714,
    "scores_api_bKk_lvh_685454": 0x685454,
    "scores_api_bKk_lvh_inner_6854b0": 0x6854B0,
    "scores_api_bKk_lvh_inner_68563c": 0x68563C,
    "scores_api_qMk_447544": 0x447544,
    "scores_api_qMk_44755c": 0x44755C,
    "scores_api_rHk_45724c": 0x45724C,
    "scores_api_rHk_457408": 0x457408,
    "scores_api_rHk_457440": 0x457440,
}


def run_adb(adb: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=check, text=True, capture_output=True)


def build_js(blutter_js: pathlib.Path, targets: dict[str, int], max_dump: int) -> str:
    helper = blutter_js.read_text(encoding="utf-8")
    helper = helper.replace(
        "    case CidClosure:\n        return getDartClosure(ptr, cls);",
        "    case 71:\n"
        "        return {nativePointer: ptr.add(7).readPointer().toString()};\n"
        "    case 90:\n"
        "        return getDartArray(ptr, cls, depthLeft);\n"
        "    case CidClosure:\n"
        "        return getDartClosure(ptr, cls);",
    )
    start = helper.index("function onLibappLoaded() {")
    end = helper.index("function tryLoadLibapp()", start)
    hook = HOOK_JS.replace("__TARGETS__", json.dumps(
        [{"name": name, "offset": offset} for name, offset in targets.items()],
        ensure_ascii=False,
    ))
    hook = hook.replace("__MAX_DUMP__", str(max_dump))
    return helper[:start] + hook + "\n" + helper[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default=r"C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    parser.add_argument("--package", default="com.sitongli.app.gadget")
    parser.add_argument("--host", default="127.0.0.1:27042")
    parser.add_argument("--usb", action="store_true", help="Use the first USB Frida device instead of a remote host.")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--blutter-js", default=r"cases\sitongli-guanshanyue\work\blutter_out\blutter_frida.js")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-dump", type=int, default=65536)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--offset", action="append", default=[])
    parser.add_argument("--spawn", action="store_true", help="Spawn the package and install hooks before app code runs.")
    parser.add_argument("--deeplink", default="", help="Optional deep link to open after hooks are loaded.")
    args = parser.parse_args()

    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"], check=False)
    pid = args.pid
    if not pid and not args.spawn:
        proc = run_adb(args.adb, ["shell", "pidof", args.package], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            pid = int(proc.stdout.strip().split()[0])

    targets = TARGETS
    if args.only:
        filters = [item.lower() for item in args.only]
        targets = {name: off for name, off in TARGETS.items() if any(f in name.lower() for f in filters)}
    for item in args.offset:
        if "=" in item:
            name, raw = item.split("=", 1)
        else:
            raw = item
            name = f"custom_{raw}"
        targets[name] = int(raw, 0)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.usb:
        device = frida.get_usb_device(timeout=10)
    else:
        device = frida.get_device_manager().add_remote_device(args.host)
    spawned_pid = None
    if args.spawn:
        run_adb(args.adb, ["shell", "am", "force-stop", args.package], check=False)
        spawned_pid = device.spawn([args.package])
        session = device.attach(spawned_pid)
        pid = spawned_pid
    else:
        session = device.attach(pid or args.package)
    script = session.create_script(build_js(pathlib.Path(args.blutter_js), targets, args.max_dump))

    with out.open("a", encoding="utf-8") as fp:
        def on_message(message, data):
            row = {"ts": time.time(), "message": message}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            payload = message.get("payload", message)
            print(json.dumps(payload, ensure_ascii=False)[:4000], flush=True)

        script.on("message", on_message)
        script.load()
        if spawned_pid is not None:
            device.resume(spawned_pid)
            if args.deeplink:
                time.sleep(2.5)
                run_adb(args.adb, ["shell", "am", "start", "-W", "-a", "android.intent.action.VIEW", "-d", args.deeplink, args.package], check=False)
        print(f"[+] attached pid={pid} out={out}", flush=True)
        while True:
            time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
