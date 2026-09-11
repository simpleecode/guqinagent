#!/usr/bin/env python3
"""Find ARM64 Dart pool-pointer references in libapp.so.

Blutter prints constants as ``[pp+0xNNNN]``.  Dart AOT normally loads them
with an ``add xN, x27, #page, lsl #12`` followed by an unsigned ``ldr``.
This scanner recognizes that narrow sequence and maps hits to the nearest
function name recovered in Blutter's addNames.py.
"""

import argparse
import bisect
import re
import struct
import zipfile
from pathlib import Path


NAME_RE = re.compile(r'idaapi\.set_name\(0x([0-9a-fA-F]+), "([^"]+)"\)')
POOL_COMMENT_RE = re.compile(
    r"ida_struct\.get_member\(struc, (\d+)\), '''(.*?)'''", re.DOTALL
)


def load_functions(path: Path) -> tuple[list[int], dict[int, str]]:
    names: dict[int, str] = {}
    for match in NAME_RE.finditer(path.read_text(encoding="utf-8", errors="ignore")):
        address = int(match.group(1), 16)
        name = match.group(2)
        if not name.endswith("_check"):
            names[address] = name
    return sorted(names), names


def load_pool_comments(path: Path) -> dict[int, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        int(match.group(1)): " ".join(match.group(2).split())
        for match in POOL_COMMENT_RE.finditer(text)
    }


def load_libapp(apk: Path, member: str) -> bytes:
    with zipfile.ZipFile(apk) as archive:
        return archive.read(member)


def nearest_function(address: int, addresses: list[int], names: dict[int, str]) -> str:
    index = bisect.bisect_right(addresses, address) - 1
    if index < 0:
        return "<unknown>"
    start = addresses[index]
    return f"{names[start]}+0x{address - start:x}"


def find_xrefs(data: bytes, target: int, lookahead: int = 8):
    words = struct.unpack_from(f"<{len(data) // 4}I", data)
    for index, word in enumerate(words):
        # ADD (immediate), 64-bit: add Xd, Xn, #imm12 [, lsl #12]
        if word & 0xFF000000 != 0x91000000:
            continue
        source = (word >> 5) & 31
        if source != 27:  # x27 is Dart's pool pointer (PP)
            continue
        dest = word & 31
        page = ((word >> 10) & 0xFFF) << (12 if ((word >> 22) & 1) else 0)
        for distance in range(1, lookahead + 1):
            if index + distance >= len(words):
                break
            load = words[index + distance]
            # LDR Xt, [Xn, #imm12 * 8], unsigned immediate, 64-bit.
            if load & 0xFFC00000 != 0xF9400000:
                continue
            base = (load >> 5) & 31
            if base != dest:
                continue
            offset = ((load >> 10) & 0xFFF) * 8
            if page + offset == target:
                yield index * 4, (index + distance) * 4, dest


def iter_pool_loads(data: bytes, start: int, end: int, lookahead: int = 8):
    words = struct.unpack_from(f"<{len(data) // 4}I", data)
    for index in range(start // 4, min(end // 4, len(words))):
        word = words[index]
        if word & 0xFF000000 != 0x91000000:
            continue
        source = (word >> 5) & 31
        if source != 27:
            continue
        dest = word & 31
        page = ((word >> 10) & 0xFFF) << (12 if ((word >> 22) & 1) else 0)
        for distance in range(1, lookahead + 1):
            if index + distance >= len(words):
                break
            load = words[index + distance]
            if load & 0xFFC00000 != 0xF9400000:
                continue
            if ((load >> 5) & 31) != dest:
                continue
            offset = ((load >> 10) & 0xFFF) * 8
            yield index * 4, (index + distance) * 4, page + offset
            break


def iter_direct_calls(data: bytes, start: int, end: int):
    words = struct.unpack_from(f"<{len(data) // 4}I", data)
    for index in range(start // 4, min(end // 4, len(words))):
        word = words[index]
        if word & 0xFC000000 != 0x94000000:  # BL signed imm26
            continue
        immediate = word & 0x03FFFFFF
        if immediate & 0x02000000:
            immediate -= 0x04000000
        address = index * 4
        yield address, address + (immediate << 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--add-names", type=Path, required=True)
    parser.add_argument("--pool-offset", action="append", default=[])
    parser.add_argument(
        "--function-address",
        action="append",
        default=[],
        help="List every annotated PP load in the function starting at this address.",
    )
    parser.add_argument("--member", default="lib/arm64-v8a/libapp.so")
    args = parser.parse_args()

    data = load_libapp(args.apk, args.member)
    addresses, names = load_functions(args.add_names)
    comments = load_pool_comments(args.add_names)
    for raw_target in args.pool_offset:
        target = int(raw_target, 0)
        hits = list(find_xrefs(data, target))
        print(f"pp+0x{target:x}: {len(hits)} hit(s)")
        for add_address, load_address, register in hits:
            print(
                f"  add=0x{add_address:x} load=0x{load_address:x} "
                f"reg=x{register} function={nearest_function(add_address, addresses, names)}"
            )
    for raw_start in args.function_address:
        start = int(raw_start, 0)
        position = bisect.bisect_right(addresses, start)
        end = addresses[position] if position < len(addresses) else len(data)
        print(f"function 0x{start:x}..0x{end:x} {names.get(start, '<unknown>')}")
        seen = set()
        for add_address, load_address, pool_offset in iter_pool_loads(data, start, end):
            key = (add_address, pool_offset)
            if key in seen:
                continue
            seen.add(key)
            comment = comments.get(pool_offset, "")
            print(
                f"  add=0x{add_address:x} load=0x{load_address:x} "
                f"pp+0x{pool_offset:x} {comment}"
            )
        print("  direct calls:")
        for call_address, target in iter_direct_calls(data, start, end):
            target_name = names.get(target, nearest_function(target, addresses, names))
            print(f"    bl@0x{call_address:x} -> 0x{target:x} {target_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
