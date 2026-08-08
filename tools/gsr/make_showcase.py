#!/usr/bin/env python3
"""Build the before/after comparison figures used in the README.

Everything here is real pipeline output read from the caches — nothing is
re-rendered or retouched for the docs. Notes on making the comparison honest:

  * Sources are multi-frame sprite SHEETS, so we extract a single frame on the
    same w//cols grid the engine samples. Comparing whole sheets would just show
    a grid of thumbnails.
  * Detail crops are shown at 1:1 with the 4x output. Fitting a 4x image into a
    small box resamples away the very difference being demonstrated.
  * Sprites are magenta-keyed, so they are composited onto a neutral
    checkerboard rather than shown as solid pink.

    python tools/gsr/make_showcase.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config
from build_hd_gsr import build_sheet_map, cell_bounds

REPO = config.REPO
AA_CACHE = os.path.join(REPO, "upscaled_gsr", "4x-UltraSharpV2_Lite_aa")
NOAA_CACHE = os.path.join(REPO, "upscaled_gsr", "4x-UltraSharpV2_Lite")
OUT = os.path.join(REPO, "docs", "images")
MAGENTA = (255, 0, 255)
BG_A, BG_B = (58, 60, 66), (46, 48, 53)
PAD, GAP, LABEL_H = 18, 14, 28
INK, DIM, PAPER = (232, 234, 238), (150, 154, 162), (26, 27, 31)
SCALE = 4

os.makedirs(OUT, exist_ok=True)
SHEETS = build_sheet_map(config.ASSETS)


def font(sz, bold=True):
    names = ["DejaVuSans-Bold.ttf"] if bold else ["DejaVuSans.ttf"]
    for d in ("/usr/share/fonts/dejavu-sans-fonts/", "/usr/share/fonts/truetype/dejavu/",
              "/usr/share/fonts/dejavu/"):
        for n in names:
            if os.path.exists(d + n):
                return ImageFont.truetype(d + n, sz)
    return ImageFont.load_default()


def checker(w, h, sq=8):
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.where((((yy // sq) + (xx // sq)) % 2)[..., None], np.array(BG_A), np.array(BG_B))
    return Image.fromarray(a.astype(np.uint8), "RGB")


def to_rgba(im):
    if im.mode == "RGBA":
        return im
    rgb = np.asarray(im.convert("RGB"), np.uint8)
    a = np.where(np.all(rgb == MAGENTA, axis=-1), 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


def flatten(im):
    r = to_rgba(im)
    bg = checker(*r.size)
    bg.paste(r, (0, 0), r)
    return bg


def frame0(im, rel):
    """Crop frame (0,0) on the engine's grid, so we show a sprite not a sheet."""
    key = next((k for k in SHEETS if k.lower().endswith(rel.lower())), None)
    if not key:
        return im, 1, 1
    cols, rows = SHEETS[key]
    xs = cell_bounds(im.width, cols)
    ys = cell_bounds(im.height, rows)
    return im.crop((xs[0], ys[0], xs[1], ys[1])), cols, rows


def source(rel):
    return Image.open(os.path.join(config.ASSETS, rel))


def upscaled(cache, rel):
    return Image.open(os.path.join(cache, rel))


def pick_crop(fr, cw, ch):
    """Choose the most detailed opaque region: max local gradient energy."""
    r = to_rgba(fr)
    g = np.asarray(r.convert("L"), np.float32)
    a = np.asarray(r, np.uint8)[..., 3] > 127
    e = np.abs(np.gradient(g)[0]) + np.abs(np.gradient(g)[1])
    e *= a
    H, W = e.shape
    cw, ch = min(cw, W), min(ch, H)
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, bx, by = -1.0, 0, 0
    for y in range(0, H - ch + 1, 4):
        for x in range(0, W - cw + 1, 4):
            s = ii[y+ch, x+cw] - ii[y, x+cw] - ii[y+ch, x] + ii[y, x]
            if s > best:
                best, bx, by = s, x, y
    return bx, by, cw, ch


def panel(img, label, sub=None):
    w, h = img.size
    lh = LABEL_H + (16 if sub else 0)
    p = Image.new("RGB", (w, h + lh), PAPER)
    p.paste(img, (0, 0))
    d = ImageDraw.Draw(p)
    f = font(15)
    d.text(((w - d.textbbox((0, 0), label, font=f)[2]) // 2, h + 6), label, font=f, fill=INK)
    if sub:
        f2 = font(12, bold=False)
        d.text(((w - d.textbbox((0, 0), sub, font=f2)[2]) // 2, h + 6 + 17), sub,
               font=f2, fill=DIM)
    return p


def row(panels, title=None, note=None):
    w = sum(p.width for p in panels) + GAP * (len(panels) - 1) + PAD * 2
    top = PAD + (34 if title else 0)
    out = Image.new("RGB", (w, max(p.height for p in panels) + top + PAD), PAPER)
    if title:
        d = ImageDraw.Draw(out)
        d.text((PAD, PAD - 2), title, font=font(17), fill=INK)
        if note:
            tw = d.textbbox((0, 0), title, font=font(17))[2]
            d.text((PAD + tw + 12, PAD + 2), note, font=font(12, bold=False), fill=DIM)
    x = PAD
    for p in panels:
        out.paste(p, (x, top)); x += p.width + GAP
    return out


def stack(rows):
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows)), PAPER)
    y = 0
    for r in rows:
        out.paste(r, ((w - r.width) // 2, y)); y += r.height
    return out


def variants(rel):
    """(source frame, nearest4x, bicubic4x, gsr4x) — all as flattened RGB frames."""
    o = source(rel)
    fr, cols, rows = frame0(o, rel)
    w, h = fr.size
    near = flatten(fr.resize((w * SCALE, h * SCALE), Image.NEAREST))
    bic = flatten(fr.resize((w * SCALE, h * SCALE), Image.BICUBIC))
    up = upscaled(AA_CACHE, rel)
    ufr, _, _ = frame0(up, rel)
    return fr, near, bic, flatten(ufr), (cols, rows)


# --------------------------------------------------------------------------- #
# Figure 1 — quality, 1:1 pixel crops
# --------------------------------------------------------------------------- #
GALLERY = [
    ("DATA/enemy/ground.goliath/goliath.bmp",  "Goliath"),
    ("DATA/enemy_gold/roger/roger.bmp",        "Roger"),
    ("DATA/enemy/water/bigsub.bmp",            "Submarine"),
]
CW, CH = 108, 84
rows_out = []
for rel, name in GALLERY:
    fr, near, bic, gsr, grid = variants(rel)
    x, y, cw, ch = pick_crop(fr, CW, CH)
    b = (x * SCALE, y * SCALE, (x + cw) * SCALE, (y + ch) * SCALE)
    note = f"sheet {grid[0]}x{grid[1]}, frame 1 - {cw}x{ch}px source region at 1:1"
    rows_out.append(row([
        panel(near.crop(b), "original", f"{SCALE}x nearest"),
        panel(bic.crop(b),  "bicubic",  f"{SCALE}x"),
        panel(gsr.crop(b),  "GSR (this mod)", f"{SCALE}x"),
    ], name, note))
stack(rows_out).save(os.path.join(OUT, "upscale-detail.png"), optimize=True)
print("wrote upscale-detail.png")

# --------------------------------------------------------------------------- #
# Figure 2 — whole sprite, before/after
# --------------------------------------------------------------------------- #
rows_out = []
for rel, name in [("DATA/enemy_gold/roger/roger.bmp", "Roger"),
                  ("DATA/enemy/air.zeppelin/dragon_zep.bmp", "Dragon zeppelin")]:
    fr, near, bic, gsr, grid = variants(rel)
    rows_out.append(row([panel(near, "before", "original, 4x nearest"),
                         panel(gsr,  "after",  "GSR 4x")],
                        name, f"whole frame at 1:1 ({fr.width}x{fr.height} source)"))
stack(rows_out).save(os.path.join(OUT, "upscale-comparison.png"), optimize=True)
print("wrote upscale-comparison.png")

# --------------------------------------------------------------------------- #
# Figure 3 — edge de-jagging (GSR_AA=0 vs GSR_AA=1), zoomed with NEAREST so the
# reader sees real output pixels: a staircase vs a traced curve.
# --------------------------------------------------------------------------- #
EDGES = [
    ("DATA/enemy/air.zeppelin/zeppelin.bmp", "Zeppelin - hull edge"),
    ("DATA/enemy_gold/roger/roger.bmp",      "Roger - canopy edge"),
]
Z, EW, EH = 3, 52, 38
rows_out = []
for rel, name in EDGES:
    o = source(rel)
    fr, _, _ = frame0(o, rel)
    # Pick a crop centred on the silhouette boundary (max alpha gradient).
    a = (np.asarray(to_rgba(fr), np.uint8)[..., 3] > 127).astype(np.float32)
    e = np.abs(np.gradient(a)[0]) + np.abs(np.gradient(a)[1])
    H, W = e.shape
    ew, eh = min(EW, W), min(EH, H)
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, bx, by = -1, 0, 0
    for yy in range(0, H - eh + 1, 2):
        for xx in range(0, W - ew + 1, 2):
            s = ii[yy+eh, xx+ew] - ii[yy, xx+ew] - ii[yy+eh, xx] + ii[yy, xx]
            if s > best:
                best, bx, by = s, xx, yy
    b = (bx * SCALE, by * SCALE, (bx + ew) * SCALE, (by + eh) * SCALE)

    def z(img):
        c = img.crop(b)
        return c.resize((c.width * Z, c.height * Z), Image.NEAREST)

    near = flatten(fr.resize((fr.width * SCALE, fr.height * SCALE), Image.NEAREST))
    noaa, _, _ = frame0(upscaled(NOAA_CACHE, rel), rel)
    aa, _, _ = frame0(upscaled(AA_CACHE, rel), rel)
    rows_out.append(row([
        panel(z(near),          "original",       "4x nearest"),
        panel(z(flatten(noaa)), "GSR, 1-bit key", "GSR_AA=0"),
        panel(z(flatten(aa)),   "GSR, de-jagged", "GSR_AA=1 (default)"),
    ], name, f"{Z}x zoom, nearest - {ew}x{eh}px source region"))
stack(rows_out).save(os.path.join(OUT, "edge-dejag.png"), optimize=True)
print("wrote edge-dejag.png")

for f in ("upscale-detail.png", "upscale-comparison.png", "edge-dejag.png"):
    p = os.path.join(OUT, f)
    print(f"  {f}: {Image.open(p).size}  {os.path.getsize(p)/1024:.0f} KB")
