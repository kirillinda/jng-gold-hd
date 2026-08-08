#!/usr/bin/env python3
"""A/B lab for upscaler settings — the harness behind the detail-preservation work.

The problem this exists to solve: UltraSharpV2 is trained with a degradation
pipeline (JPEG/noise/blur), so it treats small high-frequency structure as noise
to be removed before resynthesising a clean surface. Hand-placed 2-4px game art
(rivets, bolts, dither, isometric facet shading, baked-in lettering) lands
squarely in that band, so the model erases it and hallucinates a smooth blob in
its place. See the detail-preservation block in build_hd_gsr.py for the fix.

Every variant here is a CONFIGURATION of the shipping Upscaler, not a
reimplementation, so what the lab measures is exactly what a build produces.

  python tools/gsr/lab.py --out lab                    # all cases, all variants
  python tools/gsr/lab.py --only roger --variants base,pre2_inject

Metrics, computed by box-downscaling the 4x result back to the source grid:
  mae  low-frequency fidelity. Must stay low; a jump means the model invented.
  ret  gradient-energy retention vs the source. 1.0 = fine structure survived,
       <1 = it was smoothed away (the reported complaint), >1 = over-sharpened.
  cor  correlation with the source's gradient map. Detail in the RIGHT PLACE, so
       a variant cannot score well by sprinkling on unrelated texture.
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config
from build_hd_gsr import (Upscaler, build_sheet_map, layout_for, cell_bounds,
                          bleed, lanczos_rgb, MAGENTA, SCALE)

MODELS_DIR = os.path.join(HERE, "models")
BG_A, BG_B = (58, 60, 66), (46, 48, 53)
INK, DIM, PAPER = (232, 234, 238), (150, 154, 162), (26, 27, 31)

# Test cases, chosen from the scan_detail.py ranking to span both regimes.
# `roger` is the asset the regression was reported on (melted strut, soft bolts).
# `sideex`/`hp100`/`walk` are the worst measured structure losses: small, hard-
# edged, graphic art where the model's natural-image prior does real violence
# (a square ring came out circular; round speckles came out as diagonal gashes).
CASES = [
    ("roger",   "DATA/enemy_gold/roger/roger.bmp",       None),
    ("goliath", "DATA/enemy/ground.goliath/goliath.bmp", None),
    ("sideex",  "DATA/gui/sideex.bmp",                   None),
    ("hp100",   "DATA/enemy/special.bonus/hp100.bmp",    None),
    ("walk",    "DATA/enemy/man.space/walk.bmp",         None),
    # Lettering baked INTO artwork (not the font sheets, which take the LANCZOS
    # path). Small, low-contrast glyphs are hit hardest by the noise-band problem,
    # so this is where "the text looks bad" actually comes from.
    ("rogertext", "DATA/enemy_gold/roger/roger.bmp",     (56, 60, 116, 48)),
    ("atarix",    "DATA/menu/Atarix.bmp",                (96, 108, 96, 44)),
]

# `base` reproduces the old behaviour (straight model, no corrections);
# `pre2_inject` is the new default. Values are Upscaler attributes.
_D = dict(prescale=1, prefilter="nearest", ensemble=1, inject=False)
VARIANTS = {
    "lanczos":      None,                                     # no model at all
    "base":         {**_D},                                   # the old behaviour
    "inject":       {**_D, "inject": True},                   # texture only, no geometry fix
    "pre2":         {**_D, "prescale": 2},                    # geometry only, no texture fix
    "pre2_inject":  {**_D, "prescale": 2, "inject": True},    # the new default
    "pre2l_inject": {**_D, "prescale": 2, "prefilter": "lanczos", "inject": True},
    "pre3_inject":  {**_D, "prescale": 3, "inject": True},
    "ens8_inject":  {**_D, "prescale": 2, "ensemble": 8, "inject": True},
}


def render(up, name, src):
    if name == "lanczos":
        return lanczos_rgb(src)
    for k, v in VARIANTS[name].items():
        setattr(up, k, v)
    return up.up_many([src])[0]


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def _grad(x):
    g = np.asarray(Image.fromarray(x.astype(np.uint8), "RGB").convert("L"), np.float32)
    gy, gx = np.gradient(g)
    return np.abs(gy) + np.abs(gx)


def metrics(src, out4, mask=None):
    h, w = src.shape[:2]
    d = np.asarray(Image.fromarray(out4, "RGB").resize((w, h), Image.BOX), np.float32)
    s = src.astype(np.float32)
    m = np.ones((h, w), bool) if mask is None else mask
    a, b = _grad(s), _grad(d)
    return (float(np.abs(d - s)[m].mean()),
            float(b[m].mean() / max(a[m].mean(), 1e-6)),
            float(np.corrcoef(a[m].ravel(), b[m].ravel())[0, 1]))


# --------------------------------------------------------------------------- #
#  rendering
# --------------------------------------------------------------------------- #
def font(sz, bold=True):
    for d in ("/usr/share/fonts/dejavu-sans-fonts/", "/usr/share/fonts/truetype/dejavu/",
              "/usr/share/fonts/dejavu/"):
        n = d + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        if os.path.exists(n):
            return ImageFont.truetype(n, sz)
    return ImageFont.load_default()


def checker(w, h, sq=8):
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.where((((yy // sq) + (xx // sq)) % 2)[..., None], np.array(BG_A), np.array(BG_B))
    return Image.fromarray(a.astype(np.uint8), "RGB")


def flatten(rgb, alpha):
    bg = checker(rgb.shape[1], rgb.shape[0])
    im = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    bg.paste(im, (0, 0), im)
    return bg


def pick_crop(fr, keyed, cw, ch):
    """Most detailed opaque region: max local gradient energy."""
    e = _grad(fr) * (~keyed)
    H, W = e.shape
    cw, ch = min(cw, W), min(ch, H)
    ii = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best, bx, by = -1.0, 0, 0
    for y in range(0, H - ch + 1, 4):
        for x in range(0, W - cw + 1, 4):
            s = ii[y + ch, x + cw] - ii[y, x + cw] - ii[y + ch, x] + ii[y, x]
            if s > best:
                best, bx, by = s, x, y
    return bx, by, cw, ch


def panel(img, label, sub):
    f, f2 = font(15), font(12, False)
    m = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tw = max(m.textbbox((0, 0), label, font=f)[2], m.textbbox((0, 0), sub, font=f2)[2])
    w = max(img.width, tw + 12)                # never clip the caption
    p = Image.new("RGB", (w, img.height + 44), PAPER)
    p.paste(img, ((w - img.width) // 2, 0))
    d = ImageDraw.Draw(p)
    d.text(((w - m.textbbox((0, 0), label, font=f)[2]) // 2, img.height + 6),
           label, font=f, fill=INK)
    d.text(((w - m.textbbox((0, 0), sub, font=f2)[2]) // 2, img.height + 25),
           sub, font=f2, fill=DIM)
    return p


def grid(panels, title, per_row=4, pad=18, gap=14):
    rows = [panels[i:i + per_row] for i in range(0, len(panels), per_row)]
    rw = max(sum(p.width for p in r) + gap * (len(r) - 1) for r in rows)
    rh = max(p.height for p in panels)
    out = Image.new("RGB", (rw + pad * 2, 34 + pad + len(rows) * (rh + gap)), PAPER)
    ImageDraw.Draw(out).text((pad, pad - 2), title, font=font(17), fill=INK)
    y = 34 + pad - gap
    for r in rows:
        x = pad
        for p in r:
            out.paste(p, (x, y)); x += p.width + gap
        y += rh + gap
    return out


# --------------------------------------------------------------------------- #
def figure(up, out_path):
    """One docs-quality before/after figure: the old behaviour vs the shipping
    pipeline, on the assets that showed the worst structure loss. Both columns
    are real pipeline output — the 'before' is produced by configuring the same
    Upscaler the old way, not by reading a stale cache."""
    sheets = build_sheet_map(config.ASSETS)
    # (title, asset, crop, zoom). Zoom is NEAREST on the 4x output, so these stay
    # real output pixels — it only makes a 30px sprite legible on a docs page.
    picks = [("Roger — strut and bolts", "DATA/enemy_gold/roger/roger.bmp", (44, 12, 120, 92), 1),
             ("Roger — baked-in lettering", "DATA/enemy_gold/roger/roger.bmp", (56, 60, 116, 48), 1),
             ("Side-exit icon — a square ring", "DATA/gui/sideex.bmp", (0, 0, 96, 32), 2),
             ("Health pickup — round speckles", "DATA/enemy/special.bonus/hp100.bmp", (0, 0, 30, 30), 3)]
    rows = []
    for title, rel, (cx, cy, cw, ch), zoom in picks:
        im = Image.open(os.path.join(config.ASSETS, rel))
        w, h = im.size
        cols, rws = layout_for(rel, w, h, sheets)
        xb, yb = cell_bounds(w, cols), cell_bounds(h, rws)
        fr = im.crop((xb[0], yb[0], xb[1], yb[1]))
        rgb = np.asarray(fr.convert("RGB"), np.uint8)
        keyed = np.all(rgb == MAGENTA, axis=-1)
        src = bleed(rgb, keyed)
        a4 = np.asarray(Image.fromarray(np.where(keyed, 0, 255).astype(np.uint8), "L")
                        .resize((fr.width * SCALE, fr.height * SCALE), Image.LANCZOS), np.uint8)
        box = (cx * SCALE, cy * SCALE, (cx + cw) * SCALE, (cy + ch) * SCALE)
        sub, msk = src[cy:cy + ch, cx:cx + cw], ~keyed[cy:cy + ch, cx:cx + cw]
        panels = []
        for v, label in (("base", "before"), ("pre2_inject", "after")):
            o = render(up, v, src)
            _, ret, _ = metrics(sub, o[box[1]:box[3], box[0]:box[2]], msk)
            crop = flatten(o, a4).crop(box)
            if zoom > 1:
                crop = crop.resize((crop.width * zoom, crop.height * zoom), Image.NEAREST)
            panels.append(panel(crop, label, f"fine-structure retention {ret:.2f}"))
        note = f"{cw}x{ch} source region at 1:1"
        if zoom > 1:
            note += f", shown {zoom}x (nearest)"
        rows.append(row_of(panels, title, note))
    stack_rows(rows).save(out_path, optimize=True)
    print(f"wrote {out_path}", flush=True)


def row_of(panels, title, note, pad=18, gap=14):
    ft, fn = font(17), font(12, False)
    m = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    tw = m.textbbox((0, 0), title, font=ft)[2]
    head = tw + 12 + m.textbbox((0, 0), note, font=fn)[2]
    body = sum(p.width for p in panels) + gap * (len(panels) - 1)
    w = max(body, head) + pad * 2              # never clip the heading either
    out = Image.new("RGB", (w, max(p.height for p in panels) + 34 + pad * 2), PAPER)
    d = ImageDraw.Draw(out)
    d.text((pad, pad - 2), title, font=ft, fill=INK)
    d.text((pad + tw + 12, pad + 2), note, font=fn, fill=DIM)
    x = (w - body) // 2
    for p in panels:
        out.paste(p, (x, pad + 34)); x += p.width + gap
    return out


def stack_rows(rows):
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows)), PAPER)
    y = 0
    for r in rows:
        out.paste(r, ((w - r.width) // 2, y)); y += r.height
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(config.REPO, "lab"))
    ap.add_argument("--only", default="")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--model", default="lite", choices=("lite", "full"))
    ap.add_argument("--crop", type=int, nargs=2, default=(120, 92))
    ap.add_argument("--figure", default="", help="render the README figure here")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    sheets = build_sheet_map(config.ASSETS)
    mf = {"lite": "4x-UltraSharpV2_Lite.safetensors",
          "full": "4x-UltraSharpV2.safetensors"}[args.model]
    up = Upscaler(os.path.join(MODELS_DIR, mf))

    if args.figure:
        os.makedirs(os.path.dirname(args.figure) or ".", exist_ok=True)
        figure(up, args.figure)
        return

    for name, rel, fixed in CASES:
        if args.only and args.only not in name:
            continue
        im = Image.open(os.path.join(config.ASSETS, rel))
        w, h = im.size
        cols, rows = layout_for(rel, w, h, sheets)
        xb, yb = cell_bounds(w, cols), cell_bounds(h, rows)
        fr = im.crop((xb[0], yb[0], xb[1], yb[1]))

        rgb = np.asarray(fr.convert("RGB"), np.uint8)
        keyed = np.all(rgb == MAGENTA, axis=-1)
        src = bleed(rgb, keyed)                       # same prep as the pipeline
        alpha = np.where(keyed, 0, 255).astype(np.uint8)
        a4 = np.asarray(Image.fromarray(alpha, "L").resize(
            (fr.width * SCALE, fr.height * SCALE), Image.LANCZOS), np.uint8)

        cx, cy, cw, ch = fixed or pick_crop(rgb, keyed, *args.crop)
        box = (cx * SCALE, cy * SCALE, (cx + cw) * SCALE, (cy + ch) * SCALE)
        sub_src, sub_msk = src[cy:cy + ch, cx:cx + cw], ~keyed[cy:cy + ch, cx:cx + cw]

        print(f"\n=== {name}  {rel}  frame {fr.width}x{fr.height}  "
              f"crop {cw}x{ch}@{cx},{cy} ===", flush=True)
        print(f"{'variant':<18}{'mae':>7}{'ret':>7}{'cor':>7}")
        panels = []
        for v in want:
            out4 = render(up, v, src)
            mae, ret, cor = metrics(sub_src, out4[box[1]:box[3], box[0]:box[2]], sub_msk)
            print(f"{v:<18}{mae:7.2f}{ret:7.3f}{cor:7.3f}", flush=True)
            panels.append(panel(flatten(out4, a4).crop(box), v,
                                f"mae {mae:.1f}  ret {ret:.2f}  cor {cor:.2f}"))

        p = os.path.join(args.out, f"lab-{name}.png")
        grid(panels, f"{name} — {rel}  ({cw}x{ch} source region at 1:1)").save(p)
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
