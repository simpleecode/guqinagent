# -*- coding: utf-8 -*-
"""Find pool-entry references by scanning raw bytes for the Dart AOT pattern.
Pattern (arm64 little-endian):
    add  xN, x27, #IMM, lsl #12   ; encoding: imm12<<10 | 0x1b<<5 | 0x27<<16 | 0x1b000000 | Rd
                                   ; actual: 0x91xxxxxx with Rn=27(0x1b), sh=1
    ldr  xM, [xN, #OFF]           ; 0xF9xxxxxx, Rn=Rd_of_add
We just search for 'add xN, x27, #HI, lsl#12' then a following 'ldr xM,[xN,#LO]'
within a few bytes, computing pool offset = HI*0x1000 + LO."""
import os, re, struct
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '.codex_deps'))
import capstone

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                  'work', 'apk_unzipped', 'lib', 'arm64-v8a', 'libapp.so')
data = open(SO, 'rb').read()
TEXT_OFF, TEXT_LEN = 0x250000, 0x74edc0

# Decode add x?, x27, #imm12, lsl#12 -> machine word:
#   ADD (immediate) shifted: 0x91 (sf=1, op=0, S=0, sh=1) | imm12<<10 | Rm=0 | imm6=0? Actually:
# The real encoding: top byte 0x91, then bits. Let's just pattern match with capstone
# but walk in 4-byte steps and only decode likely ADD (0x91 prefix) instructions.
md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)

# Map: pool_offset -> list of (add_addr, ldr_addr)
TARGET_OFFSETS = {
    0x13d50: '13-hui table',
    0x13d58: 'jianpu chromatic',
    0x13d60: 'KZa.QHi closure',
    0x13d70: 'fifth-rel [-5,..]',
    0x17340: 'open-string vec',
    0x17370: 'chromatic C-root',
    0x135e8: 'major scale',
    0x172e8: '440.0 double',
    0x17330: '69.0 double',
    0x17378: '110.0 double',
    0x172f0: 'ln2 double',
}
hits = {k: [] for k in TARGET_OFFSETS}

# We decode every 4-byte word as an instruction (Dart AOT code is dense, no data
# inside .text for these regions). Track recent add-x27.
import collections
recent = collections.deque(maxlen=4)  # (addr, reg, hi)
for off in range(TEXT_OFF, TEXT_OFF + TEXT_LEN, 4):
    w = struct.unpack_from('<I', data, off)[0]
    # fast filter: ADD imm shifted with sh=1 starts with 0x91 and Rn field = 27
    if (w & 0xFF000000) == 0x91000000:
        sh = (w >> 22) & 1
        imm12 = (w >> 10) & 0xFFF
        rn = (w >> 5) & 0x1F
        rd = w & 0x1F
        if rn == 27 and sh == 1:
            recent.append((off, rd, imm12))
            continue
    # LDR (64-bit, unsigned immediate): bits[31:22] = 1111100101 -> mask 0xFFC00000 == 0xF9400000
    if (w & 0xFFC00000) == 0xF9400000:
        imm12 = (w >> 10) & 0xFFF
        rn = (w >> 5) & 0x1F
        rt = w & 0x1F
        # scale: ldr x_, offset is imm12*8
        lo = imm12 * 8
        # find a matching recent add with same reg
        for a, rd, hi in recent:
            if rd == rn:
                pool_off = hi * 0x1000 + lo
                if pool_off in TARGET_OFFSETS:
                    hits[pool_off].append((a, off))
                break

for off, label in TARGET_OFFSETS.items():
    print(f"=== {label}  pp+{off:#x}  xrefs={len(hits[off])} ===")
    for a, b in hits[off][:25]:
        print(f"   add@{a:#x}  ldr@{b:#x}")
