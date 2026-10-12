#!/usr/bin/env python3
"""Bounded, read-only probes of existing Jianzi component/font loader functions."""
import argparse
import json
import subprocess
import time
import os
import threading
from pathlib import Path
import frida

ROOT=Path(__file__).resolve().parents[1]
HOOK=r'''
function describe(raw) {
  try {
    const [tag,cls,value]=getTaggedObjectValue(raw,1);
    return {className:cls.name,classId:cls.id,value:value};
  } catch(e) { return {error:String(e)}; }
}
function emitFont(raw) {
  try {
    const tag=decompressPointer(raw), obj=tag.sub(1), cid=(obj.readU32()>>>12)&0xfffff;
    const cls=Classes[cid];
    if (cid===115) {
      const len=obj.add(cls.lenOffset).readU32()>>>1;
      if(len>1000 && len<2000000) {
        const head=Array.from(new Uint8Array(obj.add(cls.dataOffset).readByteArray(8)));
        send({event:'binary',className:cls.name,length:len,header:head},obj.add(cls.dataOffset).readByteArray(len));
      }
    } else if(cid===167) {
      send({event:'font_view',className:cls.name});
      for(const off of [8,12,16,20]) {
        const child=ptr(obj.add(off).readU32());
        if((child.toUInt32()&1)!==0) emitFont(child);
      }
    }
  }catch(e) {send({event:'font_error',error:String(e)});}
}
function onLibappLoaded() {
  send({event:'module',base:libapp.toString()});
  let calls=0;
  Interceptor.attach(libapp.add(0x5fd2a8),{
    onEnter() {
      init(this.context);this.capture=++calls<=48;
      if(!this.capture)return;
      this.args={};
      for(let i=0;i<3;i++)this.args['stack'+i]=describe(getArg(this.context,i));
      send({event:'component_enter',args:this.args});
    },
    onLeave(ret) {if(this.capture)send({event:'component_return',value:describe(ret)});}
  });
  Interceptor.attach(libapp.add(0x6bce10),{
    onEnter(){init(this.context);},
    onLeave(ret){emitFont(ret);}
  });
  Interceptor.attach(libapp.add(0x6bca98),{
    onEnter(){init(this.context);for(let i=0;i<2;i++)emitFont(getArg(this.context,i));}
  });
  send({event:'ready'});
}
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=int,default=25)
    parser.add_argument('--score-id',type=int)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'work/jianzi_renderer_probe')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    # Gadget detach can hang on a stalled app. Never leave an unbounded probe.
    watchdog=threading.Timer(min(args.seconds,45)+15,lambda:os._exit(2))
    watchdog.daemon=True;watchdog.start()
    adb=ROOT/'tools/platform-tools/adb'
    subprocess.run([str(adb),'forward','tcp:27042','tcp:27042'],check=True,capture_output=True)
    device=frida.get_device_manager().add_remote_device('127.0.0.1:27042')
    session=device.attach('Gadget')
    helper=(ROOT/'work/blutter_out/blutter_frida.js').read_text()
    start=helper.index('function onLibappLoaded() {');end=helper.index('function tryLoadLibapp()',start)
    js=helper[:start]+HOOK+'\n'+helper[end:]
    script=session.create_script(js);binaries=0
    with (args.output_dir/'probe.jsonl').open('w',encoding='utf-8') as output:
        def message(msg,data):
            nonlocal binaries
            payload=msg.get('payload',msg)
            output.write(json.dumps(payload,ensure_ascii=False)+'\n');output.flush()
            if data:
                binaries+=1;(args.output_dir/f'font_candidate_{binaries}.bin').write_bytes(data)
            print(json.dumps(payload,ensure_ascii=False)[:1000],flush=True)
        script.on('message',message)
        try:
            script.load();device.resume(session._impl.pid)
            if args.score_id:
                time.sleep(2)
                subprocess.run([str(adb),'shell','am','start','-a','android.intent.action.VIEW','-d',f'sitongli://app/scores/{args.score_id}','com.sitongli.app.gadget'],check=True)
            time.sleep(min(args.seconds,45))
        finally:
            session.detach()
    print('Detached read-only probe',flush=True)
    watchdog.cancel()


if __name__=='__main__':main()
