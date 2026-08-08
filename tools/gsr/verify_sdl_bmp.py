#!/usr/bin/env python3
"""Prove that the 32-bit BMPs we emit really do load with alpha in SDL2.

The mod writes sprite art as BMP with a BITMAPV4HEADER instead of the old
TGA-bytes-in-a-.bmp-path trick, on the strength of two claims:

  1. SDL2's BMP loader honours the alpha mask in a V4 header.
  2. CRXTexture::Load keeps that alpha — its colour-key pass masks alpha out of
     the comparison (and $0xffffff) and only zeroes exact key matches.

(2) is read off the disassembly. (1) is testable, so test it rather than trust
it: drive the real libSDL2 through ctypes, exactly as the game does via
IMG_Load_RW (which dispatches BMPs straight to SDL_LoadBMP_RW).

The game is a 32-bit binary using the system's 32-bit SDL2, but BMP header
parsing is identical in the 64-bit build, so the host library is a valid oracle.

  python tools/gsr/verify_sdl_bmp.py
"""
import os, sys, ctypes, ctypes.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from build_hd_gsr import enc_bmp32


class Surface(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32), ("format", ctypes.c_void_p),
                ("w", ctypes.c_int), ("h", ctypes.c_int), ("pitch", ctypes.c_int),
                ("pixels", ctypes.c_void_p)]


class PixelFormat(ctypes.Structure):
    _fields_ = [("format", ctypes.c_uint32), ("palette", ctypes.c_void_p),
                ("BitsPerPixel", ctypes.c_uint8), ("BytesPerPixel", ctypes.c_uint8),
                ("pad", ctypes.c_uint8 * 2),
                ("Rmask", ctypes.c_uint32), ("Gmask", ctypes.c_uint32),
                ("Bmask", ctypes.c_uint32), ("Amask", ctypes.c_uint32)]


def load_sdl():
    for n in ("libSDL2-2.0.so.0", ctypes.util.find_library("SDL2")):
        if not n:
            continue
        try:
            return ctypes.CDLL(n)
        except OSError:
            pass
    return None


def main():
    sdl = load_sdl()
    if sdl is None:
        print("SKIP: no libSDL2 on this host (the check needs it to be meaningful)")
        return 0
    sdl.SDL_RWFromMem.restype = ctypes.c_void_p
    sdl.SDL_RWFromMem.argtypes = [ctypes.c_void_p, ctypes.c_int]
    sdl.SDL_LoadBMP_RW.restype = ctypes.POINTER(Surface)
    sdl.SDL_LoadBMP_RW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    sdl.SDL_GetError.restype = ctypes.c_char_p

    # A deliberately awkward case: a fully transparent pixel, a fully opaque one,
    # and partial coverage — the exact thing a 40-byte BI_RGB header cannot state.
    rng = np.random.RandomState(0)
    src = rng.randint(0, 256, (7, 5, 4)).astype(np.uint8)
    src[0, 0] = (255, 0, 255, 0)          # keyed / transparent
    src[1, 1] = (10, 200, 30, 255)        # opaque
    src[2, 2] = (99, 40, 7, 128)          # half-covered edge pixel

    blob = enc_bmp32(src)
    buf = ctypes.create_string_buffer(blob, len(blob))
    rw = sdl.SDL_RWFromMem(ctypes.cast(buf, ctypes.c_void_p), len(blob))
    surf = sdl.SDL_LoadBMP_RW(rw, 1)
    if not surf:
        print("FAIL: SDL2 refused the BMP:", sdl.SDL_GetError().decode())
        return 1

    s = surf.contents
    fmt = ctypes.cast(s.format, ctypes.POINTER(PixelFormat)).contents
    print(f"SDL2 loaded {s.w}x{s.h}  {fmt.BitsPerPixel}bpp  pitch={s.pitch}")
    print(f"  masks R={fmt.Rmask:#010x} G={fmt.Gmask:#010x} "
          f"B={fmt.Bmask:#010x} A={fmt.Amask:#010x}")
    if fmt.Amask == 0:
        print("FAIL: SDL2 reports no alpha channel — the alpha mask was not honoured")
        return 1

    raw = ctypes.string_at(s.pixels, s.pitch * s.h)
    got = np.frombuffer(raw, np.uint8).reshape(s.h, s.pitch // 4, 4)[:, :s.w]
    # Map SDL's channel order back to RGBA using the reported masks.
    idx = {0x000000ff: 0, 0x0000ff00: 1, 0x00ff0000: 2, 0xff000000: 3}
    order = [idx[m] for m in (fmt.Rmask, fmt.Gmask, fmt.Bmask, fmt.Amask)]
    got = got[:, :, order]

    if not np.array_equal(got, src):
        bad = np.argwhere(np.any(got != src, axis=-1))
        y, x = bad[0]
        print(f"FAIL: {len(bad)} pixels differ, e.g. ({y},{x}) "
              f"wrote {tuple(src[y, x])} got {tuple(got[y, x])}")
        return 1

    print(f"  alpha round-trip exact for all {src.shape[0] * src.shape[1]} pixels "
          f"(incl. 0, 128 and 255)")
    print("PASS: SDL2 reads our BMP as 32-bit RGBA with alpha intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
