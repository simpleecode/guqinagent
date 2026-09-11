# -*- coding: utf-8 -*-
"""Disassemble a region of libapp.so (arm64) for reverse-engineering.
Usage: python disasm_func.py <hex_va> [<byte_count>] [--refs]
The blutter function addresses (e.g. 0x63e02c) are virtual addresses that map
1:1 to file offsets in this .so (.text vaddr == file offset == 0x250000 base).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '.codex_deps'))
import capstone
from elftools.elf.elffile import ELFFile

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                  'work', 'apk_unzipped', 'lib', 'arm64-v8a', 'libapp.so')

POOL_BASE = 0x600080   # pp.txt header: "pool heap offset: 0x600080"

def load():
    with open(SO, 'rb') as f:
        data = f.read()
    return data

def disasm(va, count=0x400, show=True):
    data = load()
    md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    md.detail = True
    code = data[va:va+count]
    lines = []
    for ins in md.disasm(code, va):
        lines.append(ins)
    if show:
        for ins in lines:
            print(f"  {ins.address:#x}  {ins.mnemonic:<6} {ins.op_str}")
    return lines

if __name__ == '__main__':
    va = int(sys.argv[1], 16) if sys.argv[1].startswith('0x') else int(sys.argv[1], 0)
    n = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x400
    print(f"=== disasm {va:#x} ({n} bytes) ===")
    disasm(va, n)
