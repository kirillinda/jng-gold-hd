#!/usr/bin/env python3
"""Patch jng_gold so CRXTexture::Load stores its 4 dimension members /4.

Textures upload from the SDL surface (Upload reads surface->w/h), so the members
(+0xc imgW, +0x10 imgH, +0x14 POTw, +0x18 POTh) are layout-only. A 4x surface with
members/4 => original size/pos; UV=origSrc/origPOT samples the full 4x texture.

Inject AFTER the magenta color-key loop (which uses +0x14/+0x18 as its pixel bounds)
so keying still covers the whole surface. No guard: full 4x coverage guarantees
every dim >= 4.

Both official builds are supported and auto-detected by file magic:
  * Linux   ELF (jng_gold)     — `this` in ESI; inject 0x80c014d, cave 0x80d0d00.
  * Windows PE  (jng_gold.exe) — `this` in EBX; inject 0x408609,  cave 0x43ac43.
Each patch overwrites a short jmp to a code cave, does four `sar [reg+off],2`,
replays the displaced bytes, and jumps back. Nothing is relocated. Fully
reversible — keep the original binary backup.
"""
import sys, struct


def rel32(frm_next, to):
    return struct.pack("<i", to - frm_next)


# ---- Linux / ELF -----------------------------------------------------------
# `this` is in ESI; the displaced instrs are the &this->field_4 setup that feeds
# the surface-lock call at the convergence of both Load success paths.
ELF_BASE = 0x8048000
ELF_INJECT_VA = 0x80c014d
ELF_ORIG = bytes.fromhex("8d460489442404")   # lea eax,[esi+0x4]; mov [esp+0x4],eax
ELF_CAVE_VA = 0x80d0d00
ELF_CAVE_LEN = 32


def patch_elf(data: bytearray) -> bytearray:
    def off(v):
        return v - ELF_BASE
    ret_va = ELF_INJECT_VA + len(ELF_ORIG)
    c = bytearray()
    for d in (0x0c, 0x10, 0x14, 0x18):
        c += bytes((0xc1, 0x7e, d, 0x02))     # sar dword [esi+d], 2
    c += ELF_ORIG
    jmp_va = ELF_CAVE_VA + len(c)
    c += b"\xe9" + rel32(jmp_va + 5, ret_va)

    ij = off(ELF_INJECT_VA)
    assert data[ij:ij+len(ELF_ORIG)] == ELF_ORIG, f"inject mismatch {data[ij:ij+len(ELF_ORIG)].hex()}"
    assert len(c) <= ELF_CAVE_LEN, len(c)
    assert data[off(ELF_CAVE_VA):off(ELF_CAVE_VA)+len(c)] == b"\x00"*len(c), "cave not empty"
    data[off(ELF_CAVE_VA):off(ELF_CAVE_VA)+len(c)] = c
    data[ij:ij+len(ELF_ORIG)] = b"\xe9" + rel32(ELF_INJECT_VA + 5, ELF_CAVE_VA) + b"\x90\x90"
    print(f"  ELF cave@0x{ELF_CAVE_VA:x} ({len(c)}b): {c.hex()}")
    print(f"  ELF inject@0x{ELF_INJECT_VA:x}: {bytes(data[ij:ij+len(ELF_ORIG)]).hex()}")
    return data


# ---- Windows / PE ----------------------------------------------------------
# `this` is in EBX; the displaced instrs (lea esi,[ebx+4]; push esi; push 1) are
# the &this->field_4 / mutex-lock setup at the same convergence point. The cave
# is int3 (0xcc) inter-function padding inside .text.
PE_INJECT_VA = 0x408609
PE_ORIG = bytes.fromhex("8d7304566a01")       # lea esi,[ebx+4]; push esi; push 1
PE_CAVE_VA = 0x43ac43
PE_CAVE_LEN = 45                              # bytes of 0xcc available at the cave


def _pe_va_to_off(data, va):
    """Map a virtual address to a file offset using the PE section table."""
    e_lfanew = struct.unpack_from("<I", data, 0x3c)[0]
    assert data[e_lfanew:e_lfanew+4] == b"PE\x00\x00", "not a PE"
    num_sections = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    opt_size = struct.unpack_from("<H", data, e_lfanew + 0x14)[0]
    image_base = struct.unpack_from("<I", data, e_lfanew + 0x18 + 0x1c)[0]
    sect = e_lfanew + 0x18 + opt_size
    rva = va - image_base
    for i in range(num_sections):
        s = sect + i * 0x28
        vaddr = struct.unpack_from("<I", data, s + 0x0c)[0]
        vsize = struct.unpack_from("<I", data, s + 0x08)[0]
        raw_size = struct.unpack_from("<I", data, s + 0x10)[0]
        raw_ptr = struct.unpack_from("<I", data, s + 0x14)[0]
        if vaddr <= rva < vaddr + max(vsize, raw_size):
            return raw_ptr + (rva - vaddr)
    raise ValueError(f"VA 0x{va:x} not in any section")


def _pe_set_large_address_aware(data: bytearray):
    """Set IMAGE_FILE_LARGE_ADDRESS_AWARE in the PE FILE_HEADER.

    The game is a 32-bit process. Without this flag it is capped at a 2 GB user
    address space on 64-bit Windows (vs. 3 GB on 32-bit Linux, which is why the
    mod fits there). HD art is 16x the pixels, so a level's textures — the 64 MB
    fog tiles especially — exhaust 2 GB and SDL_CreateRGBSurface starts returning
    NULL ("Unable to load texture"). Flipping this one bit lifts the ceiling to
    4 GB and is exactly the standard large-texture-pack fix. Harmless on Linux
    (the ELF path never calls this)."""
    e_lfanew = struct.unpack_from("<I", data, 0x3c)[0]
    ch_off = e_lfanew + 4 + 0x12          # Characteristics within FILE_HEADER
    ch = struct.unpack_from("<H", data, ch_off)[0]
    struct.pack_into("<H", data, ch_off, ch | 0x0020)
    print(f"  PE LARGE_ADDRESS_AWARE: 0x{ch:04x} -> 0x{ch | 0x20:04x}")


def patch_pe(data: bytearray) -> bytearray:
    ret_va = PE_INJECT_VA + len(PE_ORIG)
    c = bytearray()
    for d in (0x0c, 0x10, 0x14, 0x18):
        c += bytes((0xc1, 0x7b, d, 0x02))     # sar dword [ebx+d], 2
    c += PE_ORIG
    jmp_va = PE_CAVE_VA + len(c)
    c += b"\xe9" + rel32(jmp_va + 5, ret_va)

    ij = _pe_va_to_off(data, PE_INJECT_VA)
    cv = _pe_va_to_off(data, PE_CAVE_VA)
    assert data[ij:ij+len(PE_ORIG)] == PE_ORIG, f"inject mismatch {data[ij:ij+len(PE_ORIG)].hex()}"
    assert len(c) <= PE_CAVE_LEN, len(c)
    assert data[cv:cv+len(c)] == b"\xcc"*len(c), "cave not int3 padding (already patched?)"
    data[cv:cv+len(c)] = c
    # 5-byte jmp to the cave + one 0x90 to fill the 6th displaced byte.
    data[ij:ij+len(PE_ORIG)] = b"\xe9" + rel32(PE_INJECT_VA + 5, PE_CAVE_VA) + b"\x90"
    _pe_set_large_address_aware(data)
    print(f"  PE cave@0x{PE_CAVE_VA:x} ({len(c)}b): {c.hex()}")
    print(f"  PE inject@0x{PE_INJECT_VA:x}: {bytes(data[ij:ij+len(PE_ORIG)]).hex()}")
    return data


def main(src, dst):
    data = bytearray(open(src, "rb").read())
    if data[:2] == b"MZ":
        kind = "PE (Windows)"
        data = patch_pe(data)
    elif data[:4] == b"\x7fELF":
        kind = "ELF (Linux)"
        data = patch_elf(data)
    else:
        raise SystemExit(f"unrecognized binary format: {data[:4]!r}")
    open(dst, "wb").write(data)
    try:
        import os
        os.chmod(dst, 0o755)
    except OSError:
        pass
    print(f"patched {kind} -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
