#!/usr/bin/env python3
"""Patch jng_gold so CRXTexture::Load stores its 4 dimension members /4.

Textures upload from the SDL surface (Upload reads surface->w/h), so the members
(+0xc imgW, +0x10 imgH, +0x14 POTw, +0x18 POTh) are layout-only. A 4x surface with
members/4 => original size/pos; UV=origSrc/origPOT samples the full 4x texture.

Inject AFTER the magenta color-key loop (which uses +0x14/+0x18 as its pixel bounds)
so keying still covers the whole surface. Convergence point of both Load success
paths is 0x80c014d. No guard: full 4x coverage guarantees every dim >= 4.
"""
import sys, struct

BASE = 0x8048000
def off(v): return v - BASE
def rel32(frm_next, to): return struct.pack("<i", to - frm_next)

INJECT_VA = 0x80c014d
ORIG7 = bytes.fromhex("8d460489442404")   # lea eax,[esi+0x4]; mov [esp+0x4],eax
RET_VA = INJECT_VA + 7                     # 0x80c0154
CAVE_VA = 0x80d0d00; CAVE_LEN = 32

def build():
    c = bytearray()
    for d in (0x0c, 0x10, 0x14, 0x18):
        c += bytes((0xc1, 0x7e, d, 0x02))  # sar dword [esi+d],2
    c += ORIG7                              # replay displaced instrs
    jmp_va = CAVE_VA + len(c)
    c += b"\xe9" + rel32(jmp_va + 5, RET_VA)
    return bytes(c)

def main(src, dst):
    data = bytearray(open(src, "rb").read())
    ij = off(INJECT_VA)
    assert data[ij:ij+7] == ORIG7, f"inject mismatch {data[ij:ij+7].hex()}"
    cave = build()
    assert len(cave) <= CAVE_LEN, len(cave)
    assert data[off(CAVE_VA):off(CAVE_VA)+len(cave)] == b"\x00"*len(cave), "cave not empty"
    data[off(CAVE_VA):off(CAVE_VA)+len(cave)] = cave
    data[ij:ij+7] = b"\xe9" + rel32(INJECT_VA + 5, CAVE_VA) + b"\x90\x90"
    open(dst, "wb").write(data)
    import os; os.chmod(dst, 0o755)
    print(f"patched -> {dst}")
    print(f"  cave@0x{CAVE_VA:x} ({len(cave)}b): {cave.hex()}")
    print(f"  inject@0x{INJECT_VA:x}: {bytes(data[ij:ij+7]).hex()}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
