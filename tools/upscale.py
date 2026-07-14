#!/usr/bin/env python3
"""Asset upscaling for the JnG HD mod.

Transparency in JnG sprites comes in a few flavours; we detect and preserve each:
  - magenta color-key (255,0,255): most enemies/ships. We convert to alpha, BLEED
    sprite colours into the keyed region (so the upscaler never blends toward magenta
    -> no pink halos), upscale RGBA, then re-key back to a magenta BMP.
  - additive FX (blend_mode=1, black background): fire/explosions. Black contributes
    nothing when added, so we upscale RGB directly (black stays black).
  - grayscale masks (mode 'L', e.g. *_w): additive intensity; upscale directly.
  - existing RGBA (TGA) / opaque (JPG backgrounds): upscale in their own mode.
"""
import os, subprocess, tempfile, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

RE = config.UPSCALER
MODELS = config.MODELS_DIR
MAGENTA = (255, 0, 255)   # the game's transparent color-key for sprites

def _run(inp, outp, model, scale):
    subprocess.run([RE, "-i", inp, "-o", outp, "-n", model, "-s", str(scale), "-m", MODELS],
                   check=True, capture_output=True)

def _gpu_upscale(img: Image.Image, model="realesrgan-x4plus-anime", scale=4) -> Image.Image:
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.png"); b = os.path.join(td, "b.png")
        img.save(a)
        _run(a, b, model, scale)
        return Image.open(b).copy()

def _bleed(rgb: np.ndarray, transparent: np.ndarray, iters=24) -> np.ndarray:
    """Fill `transparent` pixels with the average of their filled neighbours,
    iteratively, so the keyed region takes on nearby sprite colours."""
    out = rgb.astype(np.float32)
    filled = ~transparent
    offs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for _ in range(iters):
        if filled.all():
            break
        nb = np.zeros_like(out); cnt = np.zeros(out.shape[:2], np.float32)
        for dy, dx in offs:
            s = np.roll(np.roll(out, dy, 0), dx, 1)
            m = np.roll(np.roll(filled, dy, 0), dx, 1).astype(np.float32)
            nb += s * m[..., None]; cnt += m
        new = (~filled) & (cnt > 0)
        out[new] = nb[new] / cnt[new][..., None]
        filled |= new
    return np.clip(out, 0, 255).astype(np.uint8)

def classify(im: Image.Image) -> str:
    if im.mode == "L":
        return "gray"
    arr = np.array(im.convert("RGB"))
    magenta = np.all(arr == MAGENTA, axis=-1)
    if magenta.mean() > 0.005:
        return "magenta"
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        return "rgba"
    return "opaque"

def _up4x(img: Image.Image, model, scale=4) -> Image.Image:
    """Upscale to EXACTLY scale x. The model is unreliable on tiny images (<=8px on
    a side), so fall back to LANCZOS there. Callers must pass magenta-free RGB (bleed
    first) so resizing never creates un-keyable near-magenta edges."""
    w, h = img.size
    tgt = (scale * w, scale * h)
    if min(w, h) <= 8:
        return img.resize(tgt, Image.LANCZOS)
    out = _gpu_upscale(img, model, scale)
    return out if out.size == tgt else out.resize(tgt, Image.LANCZOS)

def upscale_sprite(im: Image.Image, scale=4, model="realesrgan-x4plus-anime"):
    """Return (PIL image, suggested_format) for a game sprite, preserving transparency.
    Output is always exactly scale x the input dimensions."""
    kind = classify(im)
    if kind == "magenta":
        rgb = np.array(im.convert("RGB"))
        transparent = np.all(rgb == MAGENTA, axis=-1)
        bled = Image.fromarray(_bleed(rgb, transparent))      # magenta removed from RGB
        alpha = Image.fromarray(np.where(transparent, 0, 255).astype(np.uint8))
        rgba = bled.convert("RGBA"); rgba.putalpha(alpha)
        up = _up4x(rgba, model, scale).convert("RGBA")        # safe at any size (no magenta to blur)
        a = np.array(up)[..., 3]; rgb_up = np.array(up)[..., :3]
        keyed = rgb_up.copy()
        keyed[a < 128] = MAGENTA                              # re-key: exact magenta, no halo
        return Image.fromarray(keyed, "RGB"), "BMP"
    if kind == "gray":
        return _up4x(im.convert("RGB"), model, scale).convert("L"), "BMP"
    if kind == "rgba":
        return _up4x(im.convert("RGBA"), model, scale).convert("RGBA"), "PNG"
    up = _up4x(im.convert("RGB"), model, scale)
    return up, "JPEG" if im.format == "JPEG" else "BMP"

def upscale_fullscreen_ui(im: Image.Image, logical=(1067, 600), scale=4,
                          model="realesrgan-x4plus-anime") -> Image.Image:
    """Upscale a 4:3 full-screen image and fit it to the widescreen logical size:
    sharp original centered (pillarbox), sides filled with a blurred cover so it
    reads as filling the screen rather than black bars."""
    from PIL import ImageFilter
    LW, LH = logical
    up = _gpu_upscale(im.convert("RGB"), model, scale)
    # blurred cover for the background (fill width, crop)
    cover = up.copy()
    cw = LW; ch = int(up.height * LW / up.width)
    if ch < LH:
        ch = LH; cw = int(up.width * LH / up.height)
    cover = cover.resize((cw, ch)).crop(((cw-LW)//2, (ch-LH)//2, (cw-LW)//2+LW, (ch-LH)//2+LH))
    cover = cover.filter(ImageFilter.GaussianBlur(24))
    # sharp foreground fit to height, centered
    fh = LH; fw = int(up.width * LH / up.height)
    fg = up.resize((fw, fh))
    canvas = cover
    canvas.paste(fg, ((LW-fw)//2, 0))
    return canvas
