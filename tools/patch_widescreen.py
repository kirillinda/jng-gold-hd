#!/usr/bin/env python3
"""Binary-patch the 4:3 assumptions that survive in gameplay logic.

The engine renders in `Game.cfg`'s Width/Height (glOrtho(0,W,H,0)), and most of the
code asks `g_Screen` for the resolution -- shot culling, enemy despawn, the tilemap
visibility rect and the enemy-spawn trigger all read `g_Screen->Width`. A handful of
gameplay sites instead hardcode the 800x600 the game shipped with. On a 1067x600
(16:9) logical screen those sites misfire:

1. CEnemy::Update fires the `on_screen_right` behavior event at `x > 800 - w` rather
   than `x > Width - w`. Enemies spawn at the right edge (x ~ Width, correctly derived
   from g_Screen), so on a widescreen they are ALREADY past the stale 800 bound and
   fire `on_screen_right` on their very first frame. Behaviors bound to that event mean
   "you reached the right edge, turn back / leave" -- so torpedoes switch to `torp_exit`
   and fade out immediately, walkers turn around at 3/4 across, and the tanker escort
   fires `trigger.TANKER_FAILED` the moment it appears. This is the visible
   "sprites vanish long before reaching the left side" bug.

2. GetNearestEnemy (homing / auto-aim target selection) only considers enemies inside a
   hardcoded 800x600 box, so nothing in the extra width is ever targeted.

3. CEnemyPart::ParseCollisions saves the CHECK_POINT respawn scroll position as
   `worldX - 800` instead of `worldX - Width`, so you resume with the checkpoint
   mid-screen instead of at the right edge.

Every fix reads `g_Screen->Width`/`->Height` at RUNTIME (via the PIC register ebx, which
GCC keeps pinned to the GOT base 0x80e08f0 in every function here) -- no resolution is
baked in, so the binary stays correct for any Width/Height you put in Game.cfg.

Sites 1-3 are rewritten in place: the replacement is byte-for-byte the size of the
original straight-line region, and no branch enters those regions from outside (verified).
Site 3 needs 19 bytes it cannot borrow, so it calls a small routine placed in unreferenced
.rodata alignment padding (.rodata is mapped R E in this binary).

Reversible: keep the original binary. Linux ELF only -- the Windows PE is stripped and
would need its own address survey; see docs.
"""
import sys, struct

GOT = 0x80E08F0          # ebx in every PIC function in this binary
VA0, RAW0 = 0x8048000, 0  # first PT_LOAD maps file 0 -> VA 0x8048000

def foff(va):
    return va - VA0 + RAW0

def rel32(from_next, to):
    return struct.pack("<i", to - from_next)

def gotdisp(va):
    """[ebx+disp32] encoding of an absolute VA."""
    return struct.pack("<i", va - GOT)

# --- building blocks -------------------------------------------------------
# g_Screen is reached as **[ebx-0x8c] (a GOT slot holding &g_Screen).
# CRXScreen: +0xc = Width, +0x10 = Height (both int, read from Game.cfg).
def load_screen_dim(reg_rm, field):
    """reg = g_Screen->field.  reg_rm: edx=2, eax=0.  11 bytes."""
    return (bytes((0x8B, 0x80 | (reg_rm << 3) | 3)) + gotdisp(GOT - 0x8C)  # mov reg,[ebx-0x8c]
            + bytes((0x8B, (reg_rm << 3) | reg_rm))                        # mov reg,[reg]
            + bytes((0x8B, 0x40 | (reg_rm << 3) | reg_rm, field)))         # mov reg,[reg+field]

PUSH_EDX, POP_EDX = b"\x52", b"\x5a"
FILD_ESP = b"\xdb\x04\x24"        # fild dword ptr [esp]
FXCH1    = b"\xd9\xc9"            # fxch st(1)
FUCOMI1  = b"\xdb\xe9"            # fucomi  st,st(1)
FUCOMIP1 = b"\xdf\xe9"            # fucomip st,st(1)
FSTP1    = b"\xdd\xd9"            # fstp st(1)
FSTP0    = b"\xdd\xd8"            # fstp st(0)

WIDTH, HEIGHT = 0x0C, 0x10

# --- patch sites -----------------------------------------------------------
def site_on_screen_right():
    """CEnemy::Update: bound = (type ? Width - type->w : Width); leave st0=x, st1=bound.

    Region 0x80a2118..0x80a2140 (40b). Entry: st0 = spr->x, ecx = spr->type, edx dead.
    Falls through to `fucomip st,st(1)` / `ja on_screen_right` at 0x80a2140.
    """
    code = (load_screen_dim(2, WIDTH)          # mov edx, g_Screen->Width
            + b"\x85\xc9"                      # test ecx,ecx
            + b"\x74\x03"                      # je +3 -> skip the sub
            + b"\x2b\x51\x04"                  # sub edx,[ecx+0x4]   (type->width)
            + b"\x89\x54\x24\x28"              # mov [esp+0x28],edx
            + b"\xdb\x44\x24\x28"              # fild [esp+0x28]     st0=bound, st1=x
            + FXCH1)                           # fxch st(1)          st0=x, st1=bound
    return dict(
        va=0x80A2118, size=40, code=code,
        orig=bytes.fromhex("d98354e8feff85c97416ddd8ba200300002b51048954242"
                           "8db442428d9c9eb08d9c98db600000000".replace(" ", "")),
    )

def site_target_box(va, size, size_field, fld_field, jb_to, jbe_to, load_x):
    """GetNearestEnemy: replace the hardcoded 800/600 half of an on-screen test.

    Layout per axis:
        fld [edx+off]        ; st0 = pos
        lo = type ? -type->dim : 0
        if (pos < lo)  -> jb_to     (skip enemy)
        if (dim <= pos) -> jbe_to   (skip enemy)
    """
    body = (load_x                                  # fld [edx+off]  st0 = pos
            + b"\x31\xd2"                           # xor edx,edx           lo = 0
            + b"\x85\xc9"                           # test ecx,ecx
            + b"\x74\x05"                           # je +5 -> lo stays 0
            + bytes((0x8B, 0x51, size_field))       # mov edx,[ecx+size_field]
            + b"\xf7\xda"                           # neg edx               lo = -dim
            + PUSH_EDX + FILD_ESP + POP_EDX         # st0 = lo, st1 = pos
            + FXCH1                                 # st0 = pos, st1 = lo
            + FUCOMI1 + FSTP1)                      # cmp pos,lo (no pop); st0 = pos
    jb_at = va + len(body)
    body += b"\x0f\x82" + rel32(jb_at + 6, jb_to)   # jb  -> skip (pos < lo)
    body += (load_screen_dim(2, fld_field)          # mov edx, g_Screen->Width/Height
             + PUSH_EDX + FILD_ESP + POP_EDX        # st0 = dim, st1 = pos
             + FUCOMIP1 + FSTP0)                    # cmp dim,pos; pop; pop
    jbe_at = va + len(body)
    body += b"\x0f\x86" + rel32(jbe_at + 6, jbe_to)  # jbe -> skip (dim <= pos)
    return dict(va=va, size=size, code=body, orig=None)

def site_checkpoint(cave_va):
    """CEnemyPart::ParseCollisions: st0 -= (float)Width instead of 800.0f.

    0x80a016d `fsub [ebx-0x117ac]` (6b) -> `call cave` (5b) + nop. edx is dead here
    (it held the camera ptr, last used by `fld [edx]`, and is reloaded later).
    """
    cave = (load_screen_dim(2, WIDTH)
            + PUSH_EDX + FILD_ESP + POP_EDX   # st0 = Width, st1 = sum
            + b"\xde\xe9"                     # fsubp st(1),st  -> st0 = sum - Width
            + b"\xc3")                        # ret
    call_va = 0x80A016D
    patch = b"\xe8" + rel32(call_va + 5, cave_va) + b"\x90"
    return dict(va=call_va, size=6, code=patch,
                orig=bytes.fromhex("d8a354e8feff")), dict(va=cave_va, code=cave)

def main(src, dst):
    data = bytearray(open(src, "rb").read())
    if data[:4] != b"\x7fELF":
        sys.exit("expected the Linux ELF `jng_gold` (the Windows PE is not supported yet)")

    CAVE_VA = 0x80D0EA4          # unreferenced .rodata alignment padding (28b free)

    sites = [
        site_on_screen_right(),
        # X: fld [edx+0x8] = spr->x ; type->width @ +0x4 ; g_Screen->Width
        site_target_box(0x80A069D, 61, 0x04, WIDTH, 0x80A0630, 0x80A0640,
                        b"\xd9\x42\x08" + b"\x89\x54\x24\x18"),   # fld [edx+8]; mov [esp+0x18],edx
        # Y: fld [edx+0xc] = spr->y ; type->height @ +0x8 ; g_Screen->Height
        site_target_box(0x80A06DE, 60, 0x08, HEIGHT, 0x80A0634, 0x80A0640,
                        b"\xd9\x42\x0c"),                          # fld [edx+0xc]
    ]
    cp_site, cp_cave = site_checkpoint(CAVE_VA)
    sites.append(cp_site)

    # cave first, so a failed assert leaves the file untouched
    cv = foff(cp_cave["va"])
    assert all(b == 0 for b in data[cv:cv + len(cp_cave["code"])]), "cave padding not free"

    for s in sites:
        o, n = foff(s["va"]), s["size"]
        if s["orig"] is not None:
            assert bytes(data[o:o + n]) == s["orig"], (
                f"site 0x{s['va']:x}: expected {s['orig'].hex()} got {bytes(data[o:o+n]).hex()}")
        assert len(s["code"]) <= n, f"site 0x{s['va']:x}: {len(s['code'])}b > {n}b region"

    data[cv:cv + len(cp_cave["code"])] = cp_cave["code"]
    for s in sites:
        o, n = foff(s["va"]), s["size"]
        pad = n - len(s["code"])
        data[o:o + n] = s["code"] + b"\x90" * pad

    open(dst, "wb").write(data)
    import os; os.chmod(dst, 0o755)

    print(f"widescreen-patched {src} -> {dst}")
    for s in sites:
        print(f"  0x{s['va']:x}  {len(s['code']):2}b code + {s['size']-len(s['code']):2}b nop"
              f"  = {s['size']}b region")
    print(f"  cave@0x{cp_cave['va']:x} ({len(cp_cave['code'])}b): {cp_cave['code'].hex()}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: patch_widescreen.py <in-binary> <out-binary>")
    main(sys.argv[1], sys.argv[2])
