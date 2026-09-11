# -*- coding: utf-8 -*-
"""Find every instruction in libapp.so that references our known constant
tables. In Dart AOT, object-pool loads look like:
    add xN, x27, #<hi>, lsl #12      ; x27 = pool base (held in r27)
    ldr xM, [xN, #<lo>]              ; pool entry
The pool offset (hi*0x1000 + lo) is the pp.txt [pp+0x...] address.
We scan .text for the (hi,lo) pairs of our tables and report each call site
plus a few surrounding instructions."""
import sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '.codex_deps'))
import capstone
from elftools.elf.elffile import ELFFile

SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                  'work', 'apk_unzipped', 'lib', 'arm64-v8a', 'libapp.so')

# targets: (label, pp_offset)
TARGETS = [
    ('13-hui table [36,31,..]', 0x13d50),
    ('jianpu chromatic names',  0x13d58),
    ('KZa.QHi closure',         0x13d60),
    ('fifth-rel offsets [-5,..]',0x13d70),
    ('open-string vec [0,2,..]',0x17340),
    ('chromatic C-root [C,#C..]',0x17370),
    ('major scale [0,2,4..]',   0x135e8),
    ('Nab 正调 list obj',        0x0),   # placeholder
]
# Also raw immediates that might be loaded directly:
#   12 (mod), 440.0, 69.0, 110.0, ln2

def hi_lo(off):
    # Dart emits pool ref as add xN, x27, #imm12, lsl #12 ; ldr xM, [xN, #imm12]
    # The add immediate is the high (page) part, ldr offset the low part.
    return off >> 12, off & 0xFFF   # rough; Dart may split differently

with open(SO, 'rb') as f:
    data = f.read()
md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
TEXT_OFF = 0x250000
TEXT_END = 0x250000 + 0x74edc0
text = data[TEXT_OFF:TEXT_END]

# Strategy: decode each (hi,lo) we expect, then scan for matching add+ldr pairs.
# Dart's split: offset = hi*0x1000 + lo, with add imm = hi (lsl 12), ldr imm = lo.
# But ldr offset must be < 0x8000 and aligned; pp offsets here are small (<0x40000)
# so hi = off>>12 (0..0x40), lo = off & 0xFFF.
results = {label: [] for label, _ in TARGETS}
# Precompute target (hi, lo)
tgt_hl = [(label, off, off >> 12, off & 0xFFF) for label, off in TARGETS if off]

# Walk instruction stream; track last 'add xN, x27, #imm12, lsl #12' to pair with ldr.
prev_add = None  # (reg, hi)
for ins in md.disasm(text, TEXT_OFF):
    if ins.mnemonic == 'add' and 'x27' in ins.op_str and 'lsl #12' in ins.op_str:
        # add xN, x27, #imm, lsl #12
        parts = ins.op_str.split(',')
        try:
            reg = parts[0].strip()
            imm = int(parts[2].strip().replace('#',''), 0)
        except Exception:
            imm = None
        if imm is not None:
            prev_add = (reg, imm, ins.address)
    elif ins.mnemonic in ('ldr','ldur') and prev_add is not None:
        # ldr xM, [xN, #lo]
        op = ins.op_str
        if prev_add[0] in op:
            # extract offset
            import re
            m = re.search(r'#(0x[0-9a-fA-F]+|-?\d+)', op)
            if m:
                lo = int(m.group(1), 0)
                hi = prev_add[1]
                off = hi*0x1000 + lo
                for label, toff, th, tl in tgt_hl:
                    if off == toff:
                        results[label].append((prev_add[2], ins.address))
for label, off, _, _ in tgt_hl:
    print(f"\n=== {label}  (pp+{off:#x})  refs: {len(results[label])} ===")
    for a, b in results[label][:20]:
        print(f"   add@{a:#x}  ldr@{b:#x}")
