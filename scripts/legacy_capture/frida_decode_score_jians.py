#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys
import time

import frida


HOOK_JS = r"""
const SelectedTargets = __TARGETS__;
const EntryOnly = __ENTRY_ONLY__;

function readCompressedTaggedPointer(slot) {
  return ptr(slot.readU32());
}

function getDartContext(ptrObj, depthLeft) {
  const variableCount = ptrObj.add(8).readU32() >> 1;
  const parentRaw = readCompressedTaggedPointer(ptrObj.add(12));
  const out = {variableCount, variables: []};
  if (depthLeft <= 0) return out;
  try {
    const [parentPtr, parentCls, parentValue] =
        getTaggedObjectValue(parentRaw, depthLeft - 1);
    out.parent = {
      tagged: parentPtr.toString(),
      className: parentCls.name,
      classId: parentCls.id,
      value: parentValue
    };
  } catch (e) {
    out.parent_error = String(e);
  }
  const limit = Math.min(variableCount, __ARRAY_LIMIT__);
  for (let i = 0; i < limit; i++) {
    try {
      const raw = readCompressedTaggedPointer(ptrObj.add(16 + i * CompressedWordSize));
      const [tptr, cls, value] = getTaggedObjectValue(raw, depthLeft - 1);
      out.variables.push({
        tagged: tptr.toString(),
        className: cls.name,
        classId: cls.id,
        value
      });
    } catch (e) {
      out.variables.push({decode_error: String(e)});
    }
  }
  if (variableCount > limit) out.truncated = variableCount - limit;
  return out;
}

getDartArray = function(ptrObj, cls, depthLeft, glen = null) {
  const len = glen === null ? ptrObj.add(cls.lenOffset).readU32() >> 1 : glen;
  const out = [];
  const dataPtr = ptrObj.add(cls.dataOffset);
  const limit = Math.min(len, __ARRAY_LIMIT__);
  for (let i = 0; i < limit; i++) {
    let dptr = null;
    try {
      dptr = readCompressedTaggedPointer(dataPtr.add(i * CompressedWordSize));
      const [tptr, ocls, value] = getTaggedObjectValue(dptr, depthLeft - 1);
      out.push(value);
    } catch (e) {
      const failure = {
        decode_error: String(e),
        raw_tagged_pointer: dptr === null ? null : dptr.toString(),
        array_index: i
      };
      if (dptr !== null && isHeapObject(dptr)) {
        try {
          const fullTagged = decompressPointer(dptr);
          const objectPtr = fullTagged.sub(1);
          const cid = getObjectCid(objectPtr);
          failure.decompressed_tagged_pointer = fullTagged.toString();
          failure.object_pointer = objectPtr.toString();
          failure.class_id = cid;
          failure.class_name = Classes[cid] ? Classes[cid].name : null;
          failure.object_header = hexdump(objectPtr, {
            offset: 0, length: 64, header: false, ansi: false
          });
          failure.shallow_fields = {};
          const fieldNames = ["off_8", "off_c", "off_10", "off_14", "off_18", "off_1c"];
          for (let fieldIndex = 0; fieldIndex < fieldNames.length; fieldIndex++) {
            const fieldOffset = 8 + fieldIndex * CompressedWordSize;
            let fieldRaw = null;
            try {
              fieldRaw = readCompressedTaggedPointer(objectPtr.add(fieldOffset));
              const [fieldTagged, fieldClass, fieldValue] =
                  getTaggedObjectValue(fieldRaw, fieldIndex === 1 ? 6 : 2);
              failure.shallow_fields[fieldNames[fieldIndex]] = {
                raw: fieldRaw.toString(),
                tagged: fieldTagged.toString(),
                class_name: fieldClass ? fieldClass.name : null,
                value: fieldValue
              };
            } catch (fieldError) {
              failure.shallow_fields[fieldNames[fieldIndex]] = {
                decode_error: String(fieldError),
                raw: fieldRaw === null ? null : fieldRaw.toString()
              };
              // lab has one native 64-bit field at 0xc, so its pointer fields
              // are not uniformly spaced. Decode its known layout explicitly.
              if (fieldIndex === 0 && fieldRaw !== null && isHeapObject(fieldRaw)) {
                try {
                  const nestedTagged = decompressPointer(fieldRaw);
                  const nestedPtr = nestedTagged.sub(1);
                  const nestedCid = getObjectCid(nestedPtr);
                  const nested = {
                    tagged: nestedTagged.toString(),
                    class_id: nestedCid,
                    class_name: Classes[nestedCid] ? Classes[nestedCid].name : null,
                    off_c_native: nestedPtr.add(12).readS64().toString(),
                    fields: {}
                  };
                  for (const [nestedName, nestedOffset] of [
                    ["off_8", 8], ["off_14", 20],
                    ["off_18", 24], ["off_1c", 28]
                  ]) {
                    try {
                      const nestedRaw = readCompressedTaggedPointer(nestedPtr.add(nestedOffset));
                      const [nt, nc, nv] = getTaggedObjectValue(nestedRaw, 2);
                      nested.fields[nestedName] = {
                        raw: nestedRaw.toString(), tagged: nt.toString(),
                        class_name: nc ? nc.name : null, value: nv
                      };
                    } catch (nestedError) {
                      nested.fields[nestedName] = {decode_error: String(nestedError)};
                    }
                  }
                  failure.shallow_fields[fieldNames[fieldIndex]].nested_object = nested;
                } catch (nestedInspectError) {
                  failure.shallow_fields[fieldNames[fieldIndex]].inspect_error =
                      String(nestedInspectError);
                }
              }
            }
          }
        } catch (inspectError) {
          failure.inspect_error = String(inspectError);
        }
      }
      out.push(failure);
    }
  }
  if (len > limit) out.push({truncated: len - limit});
  return out;
}

getDartGrowableArray = function(ptrObj, cls, depthLeft) {
  const len = ptrObj.add(cls.lenOffset).readU32() >> 1;
  const arrTagged = decompressPointer(readCompressedTaggedPointer(ptrObj.add(cls.dataOffset)));
  return getDartArray(arrTagged.sub(1), Classes[CidArray], depthLeft, len);
}

getDartLinkedHashData = function(ptrObj, cls, depthLeft, isMap) {
  const usedData = ptrObj.add(cls.usedOffset).readU32() >> 1;
  const arrTagged = decompressPointer(readCompressedTaggedPointer(ptrObj.add(cls.dataOffset)));
  const arrPtr = arrTagged.sub(1);
  const dataCls = Classes[CidArray];
  const limit = Math.min(usedData, __MAP_LIMIT__);
  if (isMap) {
    const result = {};
    for (let i = 0; i < limit; i += 2) {
      try {
        const keyPtr = readCompressedTaggedPointer(arrPtr.add(dataCls.dataOffset + i * CompressedWordSize));
        const valPtr = readCompressedTaggedPointer(arrPtr.add(dataCls.dataOffset + (i + 1) * CompressedWordSize));
        const [keyTptr, keyCls, keyVal] = getTaggedObjectValue(keyPtr, depthLeft - 1);
        if (keyCls.id === CidNull) continue;
        const key = (keyCls.id === CidString || keyCls.id === CidTwoByteString) ? keyVal : `${keyCls.name}@${keyTptr.toString().slice(2)}`;
        const [valTptr, valCls, valVal] = getTaggedObjectValue(valPtr, depthLeft - 1);
        result[key] = valVal;
      } catch (e) {
        result[`decode_error_${i}`] = String(e);
      }
    }
    if (usedData > limit) result.__truncated__ = usedData - limit;
    return result;
  }
  const out = [];
  for (let i = 0; i < limit; i++) {
    let valPtr = null;
    try {
      valPtr = readCompressedTaggedPointer(arrPtr.add(dataCls.dataOffset + i * CompressedWordSize));
      const [valTptr, valCls, valVal] = getTaggedObjectValue(valPtr, depthLeft - 1);
      if (valCls.id !== CidNull) out.push(valVal);
    } catch (e) {
      out.push({
        decode_error: String(e),
        raw_tagged_pointer: valPtr === null ? null : valPtr.toString(),
        set_index: i
      });
    }
  }
  if (usedData > limit) out.push({truncated: usedData - limit});
  return out;
}

function decodeTagged(raw, depth) {
  const [tptr, cls, value] = getTaggedObjectValue(raw, depth);
  return {
    raw: raw.toString(),
    tagged: tptr.toString(),
    className: cls.name,
    classId: cls.id,
    value
  };
}

function decodeRegs(ctx, depth) {
  const regs = {};
  for (const name of [
    "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"
  ]) {
    try {
      const raw = ctx[name];
      regs[name] = decodeTagged(raw, depth);
    } catch (e) {
      regs[name] = {raw: ctx[name] ? ctx[name].toString() : null, error: String(e)};
    }
  }
  return regs;
}

function hookDecode(name, offset) {
  const target = libapp.add(ptr(offset));
  let sentOnce = false;
  const callbacks = {
    onEnter: function(args) {
      if (name.startsWith("source_jab_event") && sentOnce) return;
      if (name.startsWith("source_jab_event")) sentOnce = true;
      init(this.context);
      this.name = name;
      this.offset = offset;
      send({event: "enter_decoded", name, offset: "0x" + offset.toString(16), address: target.toString(), regs: decodeRegs(this.context, __ARG_DEPTH__)});
    }
  };
  if (!EntryOnly) {
    callbacks.onLeave = function(retval) {
      let decoded = null;
      try {
        decoded = decodeTagged(retval, __DEPTH__);
      } catch (e) {
        decoded = {raw: retval.toString(), error: String(e), stack: e.stack};
      }
      send({event: "decoded_retval", name: this.name, offset: "0x" + this.offset.toString(16), retval: retval.toString(), decoded});
    };
  }
  Interceptor.attach(target, callbacks);
  send({event: "hooked", name, offset: "0x" + offset.toString(16), address: target.toString()});
}

function onLibappLoaded() {
  send({event: "libapp_loaded", base: libapp.toString()});
  for (const target of SelectedTargets) {
    hookDecode(target.name, target.offset);
  }
}
"""


TARGETS = {
    "score_jians_tIk_MYa": 0x68BB50,
    "score_notes_sections_eJk": 0x624AEC,
    "note_parse_eJk_298a1c": 0x298A1C,
    "note_tuning_uJk_63e180": 0x63E180,
    "note_slur_eJk_hab_63f748": 0x63F748,
    "note_render_uJk_663eb8": 0x663EB8,
    "jianzi_main_tIk_NYa_610580": 0x610580,
    "jianzi_inner_tIk_NYa_610bb0": 0x610BB0,
    "jianzi_inner_tIk_NYa_61070c": 0x61070C,
    "jianzi_parse_tIk_NYa_610ac8": 0x610AC8,
    "jianzi_static_tIk_688784": 0x688784,
    "jianzi_edit_tIk_LYa_8ecdb0": 0x8ECDB0,
    "score_data_decode_mMk_qgb": 0x6C2B60,
    "score_dataops_fkk_Lz": 0x6C53E4,
    "score_patch_jkk_Pz": 0x6C59D0,
    "score_save_or_dataops_mLk_leb": 0x95E578,
    "native_score_ZJk_Wbb": 0x6B3E70,
    "score_open_qJk_Iab_6c4f60": 0x6C4F60,
    "score_open_mKk_Qvh_6c86c4": 0x6C86C4,
    "score_open_mKk_Qvh_inner_6c8714": 0x6C8714,
    "score_open_mLk_95dd48": 0x95DD48,
    "score_open_mLk_95ddf4": 0x95DDF4,
    "score_open_mLk_95de40": 0x95DE40,
    "scores_api_bKk_lvh_685454": 0x685454,
    "scores_api_bKk_lvh_inner_6854b0": 0x6854B0,
    "scores_api_bKk_lvh_inner_68563c": 0x68563C,
    "scores_api_qMk_447544": 0x447544,
    "scores_api_qMk_44755c": 0x44755C,
    "scores_api_rHk_45724c": 0x45724C,
    "scores_api_rHk_457408": 0x457408,
    "scores_api_rHk_457440": 0x457440,
    "view_get_jians_XIk_QZa_65ec34": 0x65EC34,
    "view_get_jians_lJk_xab_67444c": 0x67444C,
    "view_get_jians_SIk_FZa_6745cc": 0x6745CC,
    "view_set_jians_SIk_FZa_674834": 0x674834,
    "view_jianzi_iJk_qab_67261c": 0x67261C,
    "view_jianzi_jJk_rab_674a6c": 0x674A6C,
    "view_jianzi_jJk_rab_6800f4": 0x6800F4,
    "view_jianzi_jJk_rab_6800b8": 0x6800B8,
    "view_jianzi_jJk_rab_68007c": 0x68007C,
    "jianzi_component_nLa_GYi_5fd2a8": 0x5FD2A8,
    "jianzi_component_nLa_type_5fc74c": 0x5FC74C,
    "jianzi_component_nLa_type_93dea4": 0x93DEA4,
}


def run_adb(adb: str, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=check, text=True, capture_output=True)


def build_js(
    blutter_js: pathlib.Path,
    depth: int,
    array_limit: int,
    map_limit: int,
    targets: dict[str, int],
    closure_context: bool = False,
    entry_only: bool = False,
) -> str:
    helper = blutter_js.read_text(encoding="utf-8")
    if closure_context:
        old_closure_decoder = """function getDartClosure(ptr, cls) {
    let ep = ptr.add(cls.epOffset).readPointer();
    let fnName = 'unknown';
    if (libapp !== null) {
        let offset = ep.sub(libapp);
        fnName = 'fn_' + offset.toString(16);
    }
    return `Closure(${fnName})`;
}"""
        new_closure_decoder = """function getDartClosure(ptrObj, cls, depthLeft = 1) {
    let ep = ptrObj.add(cls.epOffset).readPointer();
    let fnName = 'unknown';
    if (libapp !== null) {
        let offset = ep.sub(libapp);
        fnName = 'fn_' + offset.toString(16);
    }
    const out = {function: fnName};
    if (depthLeft > 0 && cls.contextOffset >= 0) {
        try {
            const raw = readCompressedTaggedPointer(ptrObj.add(cls.contextOffset));
            const [tptr, contextCls, value] =
                getTaggedObjectValue(raw, depthLeft - 1);
            out.context = {
                tagged: tptr.toString(),
                className: contextCls.name,
                classId: contextCls.id,
                value
            };
        } catch (e) {
            out.context_error = String(e);
        }
    }
    return out;
}"""
        if old_closure_decoder not in helper:
            raise RuntimeError("Blutter Closure decoder template not found")
        helper = helper.replace(old_closure_decoder, new_closure_decoder)
        helper = helper.replace(
            "return getDartClosure(ptr, cls);",
            "return getDartClosure(ptr, cls, depthLeft);",
        )
    closure_case = "    case CidClosure:\n"
    if closure_case not in helper:
        raise RuntimeError("Blutter Closure class switch not found")
    helper = helper.replace(
        closure_case,
        "    case 28:\n"
        "        return getDartContext(ptr, depthLeft);\n"
        "    case 71:\n"
        "        return {nativePointer: ptr.add(7).readPointer().toString()};\n"
        "    case 90:\n"
        "        return getDartArray(ptr, cls, depthLeft);\n"
        + closure_case,
        1,
    )
    start = helper.index("function onLibappLoaded() {")
    end = helper.index("function tryLoadLibapp()", start)
    hook = HOOK_JS.replace("__DEPTH__", str(depth))
    hook = hook.replace("__ARG_DEPTH__", str(min(depth, 8)))
    hook = hook.replace("__ENTRY_ONLY__", "true" if entry_only else "false")
    hook = hook.replace("__ARRAY_LIMIT__", str(array_limit))
    hook = hook.replace("__MAP_LIMIT__", str(map_limit))
    hook = hook.replace("__TARGETS__", json.dumps(
        [{"name": name, "offset": offset} for name, offset in targets.items()],
        ensure_ascii=False,
    ))
    return helper[:start] + hook + "\n" + helper[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode Sitongli score jians runtime object.")
    parser.add_argument("--adb", default=r"C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    parser.add_argument("--package", default="com.sitongli.app.gadget")
    parser.add_argument("--host", default="127.0.0.1:27042")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument(
        "--attach-name",
        default="",
        help="Attach by Frida process name (use Gadget for an on_load=wait Gadget).",
    )
    parser.add_argument("--blutter-js", default=r"cases\sitongli-guanshanyue\work\blutter_out\blutter_frida.js")
    parser.add_argument("--out", required=True)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--array-limit", type=int, default=2000)
    parser.add_argument("--map-limit", type=int, default=4000)
    parser.add_argument(
        "--closure-context",
        action="store_true",
        help="Decode the captured Dart Closure context (diagnostic; may be large).",
    )
    parser.add_argument(
        "--entry-only",
        action="store_true",
        help="Capture registers only; use for safe hooks at internal instruction offsets.",
    )
    parser.add_argument("--only", action="append", default=[], help="Hook only target names containing this substring.")
    parser.add_argument("--offset", action="append", default=[], help="Extra target as name=0xoffset or 0xoffset.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a Gadget configured with on_load=wait after all hooks are loaded.",
    )
    args = parser.parse_args()

    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"], check=False)
    pid = args.pid
    if not pid:
        proc = run_adb(args.adb, ["shell", "pidof", args.package], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            pid = int(proc.stdout.strip().split()[0])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = frida.get_device_manager().add_remote_device(args.host)
    session = device.attach(args.attach_name or pid or args.package)
    session_pid = session._impl.pid
    targets = TARGETS
    if args.only:
        filters = [item.lower() for item in args.only]
        targets = {name: off for name, off in TARGETS.items() if any(f in name.lower() for f in filters)}
    for item in args.offset:
        if "=" in item:
            name, raw_offset = item.split("=", 1)
        else:
            raw_offset = item
            name = f"custom_{raw_offset}"
        targets[name] = int(raw_offset, 0)
    if not targets:
        raise SystemExit("no targets selected")
    script = session.create_script(build_js(
        pathlib.Path(args.blutter_js),
        args.depth,
        args.array_limit,
        args.map_limit,
        targets,
        closure_context=args.closure_context,
        entry_only=args.entry_only,
    ))

    with out.open("a", encoding="utf-8") as fp:
        def on_message(message, data):
            row = {"ts": time.time(), "message": message}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            payload = message.get("payload", message)
            text = json.dumps(payload, ensure_ascii=False)
            print(text[:4000], flush=True)

        script.on("message", on_message)
        script.load()
        if args.resume:
            device.resume(session_pid)
            print(f"[+] resumed pid={session_pid} after hooks loaded", flush=True)
        print(f"[+] attached pid={session_pid} out={out}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
