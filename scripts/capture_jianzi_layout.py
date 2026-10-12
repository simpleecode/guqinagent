#!/usr/bin/env python3
"""Bounded read-only capture at the App's Jianzi recursive compositor.

Offsets refer specifically to the archived App 2.3.0 arm64 AOT binary.
No Dart functions are invoked and no App objects are mutated. Recorded
geometry is evidence, not a claim of a completed natural-text renderer.
"""
import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ENTRY=0x5fae24
HOOK=r'''
const hookOffset=0x5fae24;
const depths=new Map();
let captured=0;
function object(raw) {
  if((raw.toInt32()&1)===0)return {smi:raw.toInt32()>>1};
  const tag=decompressPointer(raw), p=tag.sub(1);
  return {p:p,cid:(p.readU32()>>>12)&0xfffff};
}
function child(p,off) {return ptr(p.add(off).readU32());}
function scalar(raw) {
  const o=object(raw);
  if(o.smi!==undefined)return o.smi;
  if(o.cid===170)return null;
  if(o.cid===61)return o.p.add(8).readDouble();
  if(o.cid===93||o.cid===94) {
    const n=o.p.add(8).readU32()>>>1;
    if(n>128)throw Error('String outside glyph scope');
    return o.cid===93?o.p.add(16).readUtf8String(n):o.p.add(16).readUtf16String(n);
  }
  return {classId:o.cid};
}
function list(raw,depth,budget) {
  const o=object(raw);
  if(o.cid===170)return [];
  if(![89,90,91].includes(o.cid))return {classId:o.cid};
  const n=o.p.add(12).readU32()>>>1;
  if(n>32)throw Error('Too many layout children');
  const backing=o.cid===91?object(child(o.p,16)):o;
  const values=[];
  for(let i=0;i<n;i++)values.push(node(child(backing.p,16+4*i),depth+1,budget));
  return values;
}
function node(raw,depth,budget) {
  if(depth>10||--budget.remaining<0)throw Error('Layout capture budget exhausted');
  const o=object(raw);
  if(o.cid!==1322)return {classId:o.cid};
  const glyph=scalar(child(o.p,8));
  return {classId:o.cid,glyph:glyph,
          codepoints:typeof glyph==='string'?Array.from(glyph,c=>c.codePointAt(0)):[],
          scale_field:scalar(child(o.p,12)),
          x_field:o.p.add(16).readDouble(),y_field:o.p.add(24).readDouble(),
          children:list(child(o.p,32),depth,budget)};
}
function onLibappLoaded() {
  const bytes=libapp.add(hookOffset).readByteArray(16);
  // stp fp,lr,[x15,-16]!; mov fp,x15; sub x15,x15,80; fmov d1,2.0
  const expected=[0xfd,0x79,0xbf,0xa9,0xfd,0x03,0x0f,0xaa,
                  0xef,0x41,0x01,0xd1,0x01,0x10,0x60,0x1e];
  const actual=Array.from(new Uint8Array(bytes));
  if(actual.some((b,i)=>b!==expected[i]))throw Error('AOT entry mismatch: refusing hook');
  Interceptor.attach(libapp.add(hookOffset),{
    onEnter() {
      init(this.context);
      this.tid=Process.getCurrentThreadId();
      const depth=depths.get(this.tid)||0;depths.set(this.tid,depth+1);
      if(depth!==0||captured>=40)return;
      try {
        const root=node(this.context.x1,0,{remaining:128});
        if(root.classId!==1322)return;
        send({event:'layout',entry:'0x5fae24',root:root});captured++;
      }catch(e){send({event:'capture_error',error:String(e)});captured++;}
    },
    onLeave() {
      const depth=(depths.get(this.tid)||1)-1;
      if(depth<=0)depths.delete(this.tid);else depths.set(this.tid,depth);
    }
  });
  send({event:'ready',entry:'0x5fae24',limit:40});
}
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=20)
    parser.add_argument('--score-id',type=int)
    parser.add_argument('--open-score-body',action='store_true',
                        help='Tap score-detail body button (1080x2400 device only)')
    parser.add_argument('--output',type=Path,default=ROOT/'work/jianzi_renderer_probe/layout_capture.jsonl')
    args=parser.parse_args()
    if not 1<=args.seconds<=45:parser.error('--seconds must be 1..45')
    adb=ROOT/'tools/platform-tools/adb'
    state=subprocess.run([str(adb),'get-state'],capture_output=True,text=True,timeout=5)
    if state.returncode or state.stdout.strip()!='device':parser.exit(2,'No connected authorized ADB device; nothing captured.\n')
    import frida
    subprocess.run([str(adb),'forward','tcp:27042','tcp:27042'],check=True,capture_output=True,timeout=5)
    helper=(ROOT/'work/blutter_out/blutter_frida.js').read_text()
    start=helper.index('function onLibappLoaded() {');end=helper.index('function tryLoadLibapp()',start)
    js=helper[:start]+HOOK+'\n'+helper[end:]
    watchdog=threading.Timer(args.seconds+15,lambda:os._exit(2));watchdog.daemon=True;watchdog.start()
    device=frida.get_device_manager().add_remote_device('127.0.0.1:27042')
    session=device.attach('Gadget');args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8') as output:
        output.write(json.dumps({'event':'provenance','binary_sha256':hashlib.sha256(
            (ROOT/'work/apk_unzipped/lib/arm64-v8a/libapp.so').read_bytes()).hexdigest()})+'\n')
        def message(msg,data):
            payload=msg.get('payload',msg)
            output.write(json.dumps(payload,ensure_ascii=False)+'\n');output.flush()
            print(json.dumps(payload,ensure_ascii=False)[:500],flush=True)
        script=session.create_script(js);script.on('message',message)
        try:
            script.load();device.resume(session._impl.pid)
            if args.score_id:
                subprocess.run([str(adb),'shell','am','start','-a','android.intent.action.VIEW',
                    '-d',f'sitongli://app/scores/{args.score_id}','com.sitongli.app.gadget'],check=True,timeout=8)
                if args.open_score_body:
                    size=subprocess.check_output([str(adb),'shell','wm','size'],text=True,timeout=5)
                    if '1080x2400' not in size:raise RuntimeError('Refusing unverified tap coordinates for this device')
                    time.sleep(2)
                    subprocess.run([str(adb),'shell','input','tap','540','865'],check=True,timeout=5)
            time.sleep(args.seconds)
        finally:session.detach();watchdog.cancel()
    print('Detached layout probe: '+str(args.output))


if __name__=='__main__':main()
