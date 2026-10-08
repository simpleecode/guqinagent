#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys
import time

import frida


DEFAULT_OFFSETS = {
    "scores_path_447544": 0x447544,
    "scores_path_44755c": 0x44755C,
    "scores_path_45724c": 0x45724C,
    "scores_path_457408": 0x457408,
    "scores_path_457440": 0x457440,
    "score_key_616d44": 0x616D44,
    "score_key_616e74": 0x616E74,
    "score_save_or_dataops_mLk_leb": 0x95E578,
    "score_dataops_fkk_Lz": 0x6C53E4,
    "score_patch_jkk_Pz": 0x6C59D0,
    "native_score_ZJk_Wbb": 0x6B3E70,
    "score_notes_sections_eJk": 0x624AEC,
    "score_jians_tIk_MYa": 0x68BB50,
    "score_jian_enum_oBk_68b5c4": 0x68B5C4,
    "score_jian_enum_oBk_68b6a8": 0x68B6A8,
    "score_jian_parse_uJk_Nab": 0x68BA74,
    "score_data_decode_mMk_qgb": 0x6C2B60,
    "scores_path_685454": 0x685454,
    "scores_path_6854b0": 0x6854B0,
    "scores_path_68563c": 0x68563C,
    "score_route_rKk_jJc": 0x691040,
    "score_route_693cc0": 0x693CC0,
    "score_route_693cf0": 0x693CF0,
    "score_route_693cfc": 0x693CFC,
    "score_route_693dfc": 0x693DFC,
    "score_route_693e38": 0x693E38,
    "score_route_693f64": 0x693F64,
    "score_route_693f70": 0x693F70,
    "score_route_694008": 0x694008,
    "score_route_6941a4": 0x6941A4,
    "score_route_694384": 0x694384,
    "score_route_694390": 0x694390,
    "score_route_69439c": 0x69439C,
    "score_route_6943a8": 0x6943A8,
    "score_route_694414": 0x694414,
    "score_open_6c4f60": 0x6C4F60,
    "score_open_6c86c4": 0x6C86C4,
    "score_open_6c8714": 0x6C8714,
    "score_key_6c9568": 0x6C9568,
    "score_key_6c95c4": 0x6C95C4,
    "scores_path_6e7284": 0x6E7284,
    "dio_header_Dsh_45f79c": 0x45F79C,
    "dio_header_jKk_Dsh": 0x45F7EC,
    "dio_header_helper_45f8ec": 0x45F8EC,
    "dio_header_helper_45f9d0": 0x45F9D0,
    "token_getter_6a513c": 0x6A513C,
    "token_plain_debug_969ccc": 0x969CCC,
    "token_plain_debug_969da4": 0x969DA4,
    "score_open_95dd48": 0x95DD48,
    "score_open_95ddf4": 0x95DDF4,
    "score_open_95de40": 0x95DE40,
    "scores_path_968a94": 0x968A94,
}


HOOK_JS = r"""
const HookTargets = __HOOK_TARGETS__;
const RawOnly = __RAW_ONLY__;
const PrimitiveOnly = __PRIMITIVE_ONLY__;

function readCompressedTaggedPointer(slot) {
  const raw = slot.readU32();
  return ptr(raw);
}

getDartArray = function(ptr, cls, depthLeft, glen = null) {
  const len = glen === null ? ptr.add(cls.lenOffset).readU32() >> 1 : glen;
  let vals = [];
  let dataPtr = ptr.add(cls.dataOffset);
  for (let i = 0; i < len; i++) {
    try {
      let dptr = readCompressedTaggedPointer(dataPtr.add(i * CompressedWordSize));
      const [tptr, ocls, fieldValue] = getTaggedObjectValue(dptr, depthLeft - 1);
      if ([CidNull, CidSmi, CidMint, CidDouble, CidBool, CidString, CidTwoByteString].includes(ocls.id)) {
        vals.push(fieldValue);
      } else {
        const key = `${ocls.name}@${tptr.toString().slice(2)}`;
        let obj = {};
        obj[key] = fieldValue;
        vals.push(obj);
      }
    } catch(e) {
      vals.push({error: String(e)});
    }
  }
  return vals;
}

getDartGrowableArray = function(ptr, cls, depthLeft) {
  const len = ptr.add(cls.lenOffset).readU32() >> 1;
  let arrTagged = decompressPointer(readCompressedTaggedPointer(ptr.add(cls.dataOffset)));
  let arrPtr = arrTagged.sub(1);
  return getDartArray(arrPtr, Classes[CidArray], depthLeft, len);
}

getDartLinkedHashData = function(ptr, cls, depthLeft, isMap) {
  const usedData = ptr.add(cls.usedOffset).readU32() >> 1;
  let arrPtr = decompressPointer(readCompressedTaggedPointer(ptr.add(cls.dataOffset))).sub(1);
  let dataCls = Classes[CidArray];
  if (isMap) {
    let result = {};
    for (let i = 0; i < usedData; i += 2) {
      try {
        let keyPtr = readCompressedTaggedPointer(arrPtr.add(dataCls.dataOffset + i * CompressedWordSize));
        let valPtr = readCompressedTaggedPointer(arrPtr.add(dataCls.dataOffset + (i + 1) * CompressedWordSize));
        const [kTptr, kCls, kVal] = getTaggedObjectValue(keyPtr, depthLeft - 1);
        const [vTptr, vCls, vVal] = getTaggedObjectValue(valPtr, depthLeft - 1);
        if (kCls.id === CidNull) continue;
        let keyStr = (kCls.id === CidString || kCls.id === CidTwoByteString) ? kVal : `${kCls.name}@${kTptr.toString().slice(2)}`;
        result[keyStr] = vVal;
      } catch(e) {
        result[`decode_error_${i}`] = String(e);
      }
    }
    return result;
  }
  let items = [];
  for (let i = 0; i < usedData; i++) {
    try {
      let valPtr = readCompressedTaggedPointer(arrPtr.add(dataCls.dataOffset + i * CompressedWordSize));
      const [vTptr, vCls, vVal] = getTaggedObjectValue(valPtr, depthLeft - 1);
      if (vCls.id === CidNull) continue;
      items.push(vVal);
    } catch(e) {
      items.push({error: String(e)});
    }
  }
  return items;
}

function stringifySafe(value) {
  try {
    return JSON.stringify(value);
  } catch (e) {
    return String(value);
  }
}

function decodeArg(ctx, index) {
  const raw = getArg(ctx, index);
  if (RawOnly) {
    return { index, raw: raw.toString() };
  }
  try {
    const decoded = PrimitiveOnly ? getPrimitiveTaggedObjectValue(raw, __DECODE_DEPTH__) : getTaggedObjectValue(raw, __DECODE_DEPTH__);
    return {
      index,
      raw: raw.toString(),
      className: decoded[1].name,
      classId: decoded[1].id,
      value: decoded[2]
    };
  } catch (e) {
    return { index, raw: raw.toString(), error: String(e) };
  }
}

function decodeRetval(retval) {
  if (RawOnly) {
    return { raw: retval.toString() };
  }
  try {
    const decoded = PrimitiveOnly ? getPrimitiveTaggedObjectValue(retval, __DECODE_DEPTH__) : getTaggedObjectValue(retval, __DECODE_DEPTH__);
    return {
      raw: retval.toString(),
      className: decoded[1].name,
      classId: decoded[1].id,
      value: decoded[2]
    };
  } catch (e) {
    return { raw: retval.toString(), error: String(e) };
  }
}

function getPrimitiveTaggedObjectValue(tptr, depthLeft) {
  if (!isHeapObject(tptr)) {
    return [tptr, Classes[CidSmi], tptr.toInt32() >> 1];
  }
  const tagged = decompressPointer(tptr);
  const ptrObj = tagged.sub(1);
  const cls = Classes[getObjectCid(ptrObj)];
  const value = getPrimitiveObjectValue(ptrObj, cls, depthLeft);
  return [tagged, cls, value];
}

function getPrimitiveObjectValue(ptrObj, cls, depthLeft) {
  try {
    switch (cls.id) {
      case CidNull:
        return null;
      case CidBool:
        return getDartBool(ptrObj, cls);
      case CidString:
        return getDartString(ptrObj, cls);
      case CidTwoByteString:
        return getDartTwoByteString(ptrObj, cls);
      case CidMint:
        return getDartMint(ptrObj, cls).toString();
      case CidDouble:
        return getDartDouble(ptrObj, cls);
      case CidClosure:
        return getDartClosure(ptrObj, cls);
      case CidArray:
        return getPrimitiveArray(ptrObj, cls, depthLeft);
      case CidGrowableArray:
        return getPrimitiveGrowableArray(ptrObj, cls, depthLeft);
      case CidMap:
        return getPrimitiveMap(ptrObj, cls, depthLeft);
      case CidSet:
        return getPrimitiveSet(ptrObj, cls, depthLeft);
      default:
        return {object: `${cls.name}@${ptrObj.toString().slice(2)}`, classId: cls.id};
    }
  } catch (e) {
    return {object: `${cls.name}@${ptrObj.toString().slice(2)}`, classId: cls.id, error: String(e)};
  }
}

function primitiveSlotValue(slot, depthLeft) {
  const dptr = readCompressedTaggedPointer(slot);
  const [tptr, cls, value] = getPrimitiveTaggedObjectValue(dptr, depthLeft);
  if ([CidNull, CidSmi, CidMint, CidDouble, CidBool, CidString, CidTwoByteString].includes(cls.id)) {
    return value;
  }
  return value;
}

function getPrimitiveArray(ptrObj, cls, depthLeft, glen = null) {
  const len = glen === null ? ptrObj.add(cls.lenOffset).readU32() >> 1 : glen;
  const out = [];
  const limit = Math.min(len, 256);
  const dataPtr = ptrObj.add(cls.dataOffset);
  for (let i = 0; i < limit; i++) {
    if (depthLeft <= 0) {
      out.push("...");
      break;
    }
    out.push(primitiveSlotValue(dataPtr.add(i * CompressedWordSize), depthLeft - 1));
  }
  if (len > limit) out.push({truncated: len - limit});
  return out;
}

function getPrimitiveGrowableArray(ptrObj, cls, depthLeft) {
  const len = ptrObj.add(cls.lenOffset).readU32() >> 1;
  const arrTagged = decompressPointer(readCompressedTaggedPointer(ptrObj.add(cls.dataOffset)));
  return getPrimitiveArray(arrTagged.sub(1), Classes[CidArray], depthLeft, len);
}

function getPrimitiveMap(ptrObj, cls, depthLeft) {
  const usedData = ptrObj.add(cls.usedOffset).readU32() >> 1;
  const arrPtr = decompressPointer(readCompressedTaggedPointer(ptrObj.add(cls.dataOffset))).sub(1);
  const dataCls = Classes[CidArray];
  const result = {};
  const limit = Math.min(usedData, 512);
  for (let i = 0; i < limit; i += 2) {
    if (depthLeft <= 0) break;
    try {
      const keySlot = arrPtr.add(dataCls.dataOffset + i * CompressedWordSize);
      const valSlot = arrPtr.add(dataCls.dataOffset + (i + 1) * CompressedWordSize);
      const keyRaw = readCompressedTaggedPointer(keySlot);
      const [keyTptr, keyCls, keyVal] = getPrimitiveTaggedObjectValue(keyRaw, depthLeft - 1);
      if (keyCls.id === CidNull) continue;
      const key = (keyCls.id === CidString || keyCls.id === CidTwoByteString) ? keyVal : `${keyCls.name}@${keyTptr.toString().slice(2)}`;
      result[key] = primitiveSlotValue(valSlot, depthLeft - 1);
    } catch (e) {
      result[`decode_error_${i}`] = String(e);
    }
  }
  if (usedData > limit) result.__truncated__ = usedData - limit;
  return result;
}

function getPrimitiveSet(ptrObj, cls, depthLeft) {
  const usedData = ptrObj.add(cls.usedOffset).readU32() >> 1;
  const arrPtr = decompressPointer(readCompressedTaggedPointer(ptrObj.add(cls.dataOffset))).sub(1);
  const dataCls = Classes[CidArray];
  const out = [];
  const limit = Math.min(usedData, 256);
  for (let i = 0; i < limit; i++) {
    if (depthLeft <= 0) break;
    out.push(primitiveSlotValue(arrPtr.add(dataCls.dataOffset + i * CompressedWordSize), depthLeft - 1));
  }
  if (usedData > limit) out.push({truncated: usedData - limit});
  return out;
}

function hookDartTarget(name, offset) {
  const address = libapp.add(ptr(offset));
  Interceptor.attach(address, {
    onEnter: function (args) {
      init(this.context);
      const decodedArgs = [];
      for (let i = 0; i < 8; i++) {
        decodedArgs.push(decodeArg(this.context, i));
      }
      this.captureName = name;
      send({
        event: "dart_enter",
        name,
        offset: "0x" + offset.toString(16),
        address: address.toString(),
        args: decodedArgs
      });
    },
    onLeave: function (retval) {
      send({
        event: "dart_leave",
        name: this.captureName,
        offset: "0x" + offset.toString(16),
        retval: decodeRetval(retval)
      });
    }
  });
}

function onLibappLoaded() {
  send({event: "libapp_loaded", base: libapp.toString(), targets: HookTargets});
  for (const t of HookTargets) {
    try {
      hookDartTarget(t.name, t.offset);
      send({event: "hooked", name: t.name, offset: "0x" + t.offset.toString(16)});
    } catch (e) {
      send({event: "hook_error", name: t.name, offset: "0x" + t.offset.toString(16), error: String(e)});
    }
  }
}
"""


def run_adb(adb: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=check, text=True, capture_output=True)


def build_js(blutter_js: pathlib.Path, targets: dict[str, int], depth: int, raw_only: bool, primitive_only: bool) -> str:
    helper = blutter_js.read_text(encoding="utf-8")
    start = helper.index("function onLibappLoaded() {")
    end = helper.index("function tryLoadLibapp()", start)
    target_list = [{"name": name, "offset": offset} for name, offset in targets.items()]
    replacement = HOOK_JS.replace("__HOOK_TARGETS__", json.dumps(target_list, ensure_ascii=False))
    replacement = replacement.replace("__DECODE_DEPTH__", str(depth))
    replacement = replacement.replace("__RAW_ONLY__", "true" if raw_only else "false")
    replacement = replacement.replace("__PRIMITIVE_ONLY__", "true" if primitive_only else "false")
    return helper[:start] + replacement + "\n" + helper[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Sitongli score objects through embedded Frida Gadget.")
    parser.add_argument("--adb", default=r"C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    parser.add_argument("--package", default="com.sitongli.app")
    parser.add_argument("--blutter-js", default=r"cases\sitongli-guanshanyue\work\blutter_out\blutter_frida.js")
    parser.add_argument("--out", default=r"cases\sitongli-guanshanyue\evidence\gadget_capture_score.jsonl")
    parser.add_argument("--host", default="127.0.0.1:27042")
    parser.add_argument("--no-launch", action="store_true", help="Do not adb launch the package before attaching.")
    parser.add_argument("--pid", type=int, default=0, help="Attach to this PID instead of resolving by package name.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Hook only names containing this substring. Can be supplied multiple times.",
    )
    parser.add_argument("--depth", type=int, default=7, help="Dart object decode depth.")
    parser.add_argument("--raw-only", action="store_true", help="Record raw Dart argument pointers without decoding objects.")
    parser.add_argument("--primitive-only", action="store_true", help="Decode only primitive Dart values and shallow List/Map containers.")
    args = parser.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"], check=False)

    if not args.no_launch:
        run_adb(args.adb, ["shell", "monkey", "-p", args.package, "-c", "android.intent.category.LAUNCHER", "1"], check=False)
        time.sleep(2)

    device = frida.get_device_manager().add_remote_device(args.host)
    attach_target = args.pid
    if attach_target == 0:
        proc = run_adb(args.adb, ["shell", "pidof", args.package], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                attach_target = int(proc.stdout.strip().split()[0])
            except ValueError:
                attach_target = 0

    session = None
    deadline = time.time() + 30
    last_error = None
    while time.time() < deadline:
        try:
            session = device.attach(attach_target or args.package)
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    if session is None:
        raise RuntimeError(f"Unable to attach to Gadget at {args.host}: {last_error}")

    targets = DEFAULT_OFFSETS
    if args.only:
        filters = [item.lower() for item in args.only]
        targets = {
            name: offset
            for name, offset in DEFAULT_OFFSETS.items()
            if any(f in name.lower() for f in filters)
        }
        if not targets:
            raise RuntimeError(f"No hook targets matched --only filters: {args.only}")

    js = build_js(pathlib.Path(args.blutter_js), targets, args.depth, args.raw_only, args.primitive_only)
    script = session.create_script(js)

    with out_path.open("a", encoding="utf-8") as fp:
        def on_message(message, data):
            record = {"ts": time.time(), "message": message}
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()
            if message.get("type") == "send":
                payload = message.get("payload", {})
                print(json.dumps(payload, ensure_ascii=False)[:2000])
            else:
                print(json.dumps(message, ensure_ascii=False), file=sys.stderr)

        script.on("message", on_message)
        script.load()
        print(f"[+] attached. Output: {out_path}")
        print("[+] Open 《关山月》编辑页, make a tiny reversible edit if needed, then tap 保存. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[+] stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
