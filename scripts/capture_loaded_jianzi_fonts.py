#!/usr/bin/env python3
"""Read only SFNT font buffers in the authorized Gadget process; bounded scan."""
import argparse
import json
import os
import threading
from pathlib import Path
import frida

JS=r'''
const seen=new Set();let scanned=0;let count=0;
for(const range of Process.enumerateRanges({protection:'r--',coalesce:false}).sort((a,b)=>a.size-b.size)) {
  if(range.file || range.size>32*1024*1024 || scanned+range.size>256*1024*1024)continue;
  scanned+=range.size;
  try {
    const hits=[...Memory.scanSync(range.base,range.size,'00 01 00 00 00 ?? 00'),...Memory.scanSync(range.base,range.size,'4f 54 54 4f 00 ?? 00')];
    for(const hit of hits) {
      try {
      const p=hit.address;
      const n=((p.add(4).readU8()<<8)|p.add(5).readU8());
      if(n<5 || n>40)continue;
      let size=12+n*16;const tags=[];
      for(let i=0;i<n;i++) {
        const q=p.add(12+i*16);const tag=String.fromCharCode(...Array.from(new Uint8Array(q.readByteArray(4))));tags.push(tag);
        const off=(q.add(8).readU8()*16777216)+(q.add(9).readU8()<<16)+(q.add(10).readU8()<<8)+q.add(11).readU8();
        const len=(q.add(12).readU8()*16777216)+(q.add(13).readU8()<<16)+(q.add(14).readU8()<<8)+q.add(15).readU8();
        size=Math.max(size,off+len);
      }
      if(!['cmap','head','maxp'].every(x=>tags.includes(x)) || size>1024*1024 || size<2000)continue;
      const key=p.toString();if(seen.has(key))continue;seen.add(key);
      send({event:'font',index:++count,size:Math.ceil(size/4)*4,tags:tags},p.readByteArray(Math.ceil(size/4)*4));
      if(count>=20)break;
      }catch(e){}
    }
  }catch(e){}
  if(count>=20)break;
}
send({event:'complete',scanned_bytes:scanned,fonts:count});
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path('work/jianzi_renderer_probe/fonts'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    timer=threading.Timer(45,lambda:os._exit(2));timer.daemon=True;timer.start()
    session=frida.get_device_manager().add_remote_device('127.0.0.1:27042').attach('Gadget')
    script=session.create_script(JS)
    def message(msg,data):
        payload=msg.get('payload',msg)
        if data:(args.output_dir/f'font_{payload["index"]}.ttf').write_bytes(data)
        print(json.dumps(payload,ensure_ascii=False),flush=True)
    script.on('message',message)
    try:script.load()
    finally:session.detach();timer.cancel()


if __name__=='__main__':main()
