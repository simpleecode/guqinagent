#!/usr/bin/env python3
"""Low-risk Dart AOT target probe for Frida Gadget.

Unlike frida_decode_score_jians.py this probe never recursively expands Dart
instances.  It records only the top-level class and primitive value for a
small, rate-limited number of calls.
"""

import argparse
import json
import pathlib
import subprocess
import time

import frida


HOOK_JS = r"""
const SelectedTargets = __TARGETS__;
const MaxHits = __MAX_HITS__;
const InspectX7Wrapper = __INSPECT_X7_WRAPPER__;

function shallow(raw) {
  try {
    if (!isHeapObject(raw)) {
      return {raw: raw.toString(), className: "_Smi", classId: CidSmi, value: raw.toInt32() >> 1};
    }
    const tagged = decompressPointer(raw);
    const object = tagged.sub(1);
    const cid = getObjectCid(object);
    const cls = Classes[cid];
    if (!cls) return {raw: raw.toString(), tagged: tagged.toString(), classId: cid, error: "unknown class"};
    let primitive = null;
    if (cid === CidString) primitive = getDartString(object, cls);
    else if (cid === CidTwoByteString) primitive = getDartTwoByteString(object, cls);
    else if (cid === CidMint) primitive = getDartMint(object, cls).toString();
    else if (cid === CidDouble) primitive = getDartDouble(object, cls);
    else if (cid === CidBool) primitive = getDartBool(object, cls);
    else if (cid === CidNull) primitive = null;
    return {
      raw: raw.toString(),
      tagged: tagged.toString(),
      className: cls.name,
      classId: cid,
      value: primitive
    };
  } catch (e) {
    return {raw: raw ? raw.toString() : null, error: String(e)};
  }
}

function hookProbe(name, offset) {
  const target = libapp.add(ptr(offset));
  let hits = 0;
  Interceptor.attach(target, {
    onEnter: function () {
      init(this.context);
      hits++;
      this.emit = hits <= MaxHits;
      this.name = name;
      this.hit = hits;
      if (!this.emit) return;
      const regs = {};
      for (const reg of ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]) {
        regs[reg] = shallow(this.context[reg]);
      }
      let x7Wrapper = null;
      if (InspectX7Wrapper && regs.x7.classId === 4837) {
        try {
          const wrapper = decompressPointer(this.context.x7).sub(1);
          const childRaw = ptr(wrapper.add(0x10).readU32());
          const child = shallow(childRaw);
          x7Wrapper = {child};
          if (child.classId === CidArray) {
            const listObject = decompressPointer(childRaw).sub(1);
            const length = listObject.add(0xc).readU32() >> 1;
            x7Wrapper.length = length;
            x7Wrapper.items = [];
            const limit = Math.min(length, 16);
            for (let i = 0; i < limit; i++) {
              x7Wrapper.items.push(shallow(ptr(listObject.add(0x10 + i * 4).readU32())));
            }
          } else if (child.classId === 4682) {
            const childObject = decompressPointer(childRaw).sub(1);
            x7Wrapper.field8 = shallow(ptr(childObject.add(0x8).readU32()));
            x7Wrapper.fieldC = shallow(ptr(childObject.add(0xc).readU32()));
            x7Wrapper.field10 = shallow(ptr(childObject.add(0x10).readU32()));
          }
        } catch (e) {
          x7Wrapper = {error: String(e)};
        }
      }
      send({event: "probe_enter", name, offset: "0x" + offset.toString(16), hit: hits, regs, x7Wrapper});
    },
    onLeave: function (retval) {
      if (this.emit) send({event: "probe_leave", name: this.name, hit: this.hit, retval: shallow(retval)});
    }
  });
  send({event: "hooked", name, offset: "0x" + offset.toString(16), address: target.toString(), maxHits: MaxHits});
}

function onLibappLoaded() {
  send({event: "libapp_loaded", base: libapp.toString()});
  for (const target of SelectedTargets) hookProbe(target.name, target.offset);
}
"""


def run_adb(adb: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([adb, *args], check=False, text=True, capture_output=True)


def build_js(helper_path: pathlib.Path, targets: list[dict], max_hits: int, inspect_x7_wrapper: bool) -> str:
    helper = helper_path.read_text(encoding="utf-8")
    start = helper.index("function onLibappLoaded() {")
    end = helper.index("function tryLoadLibapp()", start)
    hook = HOOK_JS.replace("__TARGETS__", json.dumps(targets))
    hook = hook.replace("__MAX_HITS__", str(max_hits))
    hook = hook.replace("__INSPECT_X7_WRAPPER__", "true" if inspect_x7_wrapper else "false")
    return helper[:start] + hook + "\n" + helper[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default=r"tools\platform-tools\adb.exe")
    parser.add_argument("--package", default="com.sitongli.app.gadget")
    parser.add_argument("--host", default="127.0.0.1:27042")
    parser.add_argument("--attach-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--blutter-js", default=r"work\blutter_out\blutter_frida.js")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", action="append", required=True, help="name=0xoffset")
    parser.add_argument("--max-hits", type=int, default=20)
    parser.add_argument("--inspect-x7-wrapper", action="store_true")
    args = parser.parse_args()

    targets = []
    for item in args.target:
        name, raw_offset = item.split("=", 1)
        targets.append({"name": name, "offset": int(raw_offset, 0)})

    run_adb(args.adb, ["forward", "tcp:27042", "tcp:27042"])
    pid_result = run_adb(args.adb, ["shell", "pidof", args.package])
    if not pid_result.stdout.strip():
        raise RuntimeError(f"{args.package} is not running")
    pid = int(pid_result.stdout.strip().split()[0])

    device = frida.get_device_manager().add_remote_device(args.host)
    session = device.attach(args.attach_name or pid)
    session_pid = session._impl.pid
    script = session.create_script(
        build_js(pathlib.Path(args.blutter_js), targets, args.max_hits, args.inspect_x7_wrapper)
    )
    output = pathlib.Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        def on_message(message, data):
            fp.write(json.dumps({"ts": time.time(), "message": message}, ensure_ascii=False) + "\n")
            fp.flush()
            payload = message.get("payload", message)
            print(json.dumps(payload, ensure_ascii=False)[:2000], flush=True)

        script.on("message", on_message)
        script.load()
        if args.resume:
            device.resume(session_pid)
            print(f"[+] resumed pid={session_pid} after shallow hook loaded", flush=True)
        print(f"[+] shallow probe attached pid={session_pid}", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0
        finally:
            session.detach()


if __name__ == "__main__":
    raise SystemExit(main())
