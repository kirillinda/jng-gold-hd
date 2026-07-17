#!/usr/bin/env python3
"""GPU super-resolution HD overlay builder for Jets'n'Guns Gold.

Replaces the old realesrgan-ncnn (NMKD-Siax) path, whose output was soft/"soapy".
Runs a modern GAN-trained super-resolution model through PyTorch/spandrel on the
7900 XTX (ROCm), in FP16 so conv/matmul use the RDNA3 WMMA matrix cores.

Default model: 4x-UltraSharpV2_Lite (RealPLKSR, conv-only) — crisp, detail-
synthesising, and — being attention-free — far faster on gfx1100, where
Flash-Attention isn't a win. 4x-UltraSharpV2 (DAT2) is available via GSR_MODEL
for maximum quality where the extra time is acceptable.

What makes this correct for a 20-year-old sprite engine (not just "run a model"):

  * EXACT 4x. The binary patch divides every texture dimension by 4, so every
    image the game loads must be exactly 4x its original size. Output dims are
    always (4w, 4h).

  * ANIMATION SHEETS ARE SPLIT PER FRAME. A sheet with `frames_wh = N, cols, rows`
    is a grid; upscaling it whole smears detail across frame borders (the reported
    "smeared" look). We cut each frame out, upscale it alone, and lay the frames
    back on the SAME grid the engine reads: frame width = w // cols (integer, as
    the C++ engine computes it), so 4x frame i lands at 4*i*(w//cols). Cells are
    contiguous and the last cell absorbs any non-divisible remainder, so the 4x
    canvas tiles exactly with no seams and no smear.

  * TRANSPARENCY PRESERVED PER FLAVOUR (as the old tool did, kept faithful):
      - magenta color-key (255,0,255): bleed sprite colour into the keyed region
        (so the model never blends toward magenta -> no pink halo), upscale RGB,
        upscale the alpha mask with LANCZOS, then re-key exact magenta -> BMP.
      - RGBA (.tga): RGB (edge-bled) via model + alpha via LANCZOS -> TGA.
      - grayscale additive masks (mode L): model on replicated RGB -> L.
      - opaque (.jpg/.bmp): model -> JPEG/BMP. 4:3 full-screen menus -> widescreen.

  * TEXT IS NOT AI'd. Font/glyph sheets and HUD digits are scaled with LANCZOS
    (faithful, no glyph warping) but still 4x'd so the /4 patch keeps them the
    right size. Tiny (<=8px) images likewise (the model is unreliable that small).

  * THE HTML MANUAL (DATA/manual/*) is left out of the overlay entirely — it is
    rendered by the HTML viewer, not CRXTexture::Load, so it must stay 1x.

  * AVATAR GRID. DATA/menu/hero_faces.jpg is a 5x7 face grid (235x511 -> 47x73
    cells); split like an animation sheet so faces don't bleed into each other.

Output cached per model under upscaled_gsr/<model>/ and packed into build/hd.dat.
"""
import os, sys, io, re, glob, json, time, shutil, argparse
import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # tools/ -> config, jngdat
import config
# jngdat (LZO) is imported lazily in main() only when packing, so pure upscale /
# --no-pack runs don't require the archive backend (liblzo2/lzallright).

MAGENTA = (255, 0, 255)
SCALE = 4
MODEL_NAME = os.environ.get("GSR_MODEL", "4x-UltraSharpV2_Lite")
MODEL_PATH = os.path.join(HERE, "models", MODEL_NAME + ".safetensors")
CACHE = os.path.join(config.REPO, "upscaled_gsr", MODEL_NAME)
LOGICAL = config.LOGICAL

IMG_EXT = (".bmp", ".tga", ".jpg", ".jpeg", ".gif")
# 4:3 full-screen menu art that must be fitted into the 16:9 logical screen.
FS_DIRS = ("DATA/menu/screen/", "DATA/menu/screen2006/")
FS_FILES = ("DATA/menu/failed.jpg",)
# Glyph / digit sheets: 4x'd but with LANCZOS, never the model (no letter warping).
TEXT_FILES = {
    "DATA/fonts/font_big.bmp", "DATA/fonts/font_mgb.bmp", "DATA/fonts/font_shop.bmp",
    "DATA/fonts/font_sml.bmp", "DATA/fonts/font_sml.tga",
    "DATA/enemy/boss.comp/font.bmp", "DATA/hud/switchnumbers.bmp",
}
# Excluded from the overlay (HTML manual, rendered outside the texture path).
SKIP_PREFIXES = ("DATA/manual/",)
# Known non-frames_wh grids we can split with confidence (cols, rows).
AVATAR_GRIDS = {"DATA/menu/hero_faces.jpg": (5, 7)}
TINY = 8            # <= this on either side -> LANCZOS

# --------------------------------------------------------------------------- #
#  sprite-sheet layout map: normalized bitmap path -> (cols, rows)
# --------------------------------------------------------------------------- #
def build_sheet_map(assets_dir):
    sprite_re = re.compile(r'\[\s*sprite\s*=', re.I)
    bitmap_re = re.compile(r'bitmap\s*=\s*(.+)', re.I)
    fwh_re = re.compile(r'frames_wh\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', re.I)
    m = {}
    for tf in glob.glob(os.path.join(assets_dir, "DATA/**/*.txt"), recursive=True):
        cur = None
        for line in open(tf, errors="replace"):
            if sprite_re.search(line):
                cur = None
            g = bitmap_re.search(line)
            if g:
                cur = g.group(1).strip().replace("\\", "/").lower(); continue
            g = fwh_re.search(line)
            if g and cur:
                _, cols, rows = (int(x) for x in g.groups())
                # Highest (cols*rows) wins on conflict: a real split beats a stray 1,1,1.
                prev = m.get(cur)
                if prev is None or cols * rows > prev[0] * prev[1]:
                    m[cur] = (cols, rows)
                cur = None
    return m


def layout_for(rel, w, h, sheet_map):
    if rel in AVATAR_GRIDS:
        return AVATAR_GRIDS[rel]
    key = rel.replace("\\", "/").lower()
    cols, rows = sheet_map.get(key, (1, 1))
    # Never split into sub-cells the model can't handle, and guard bad divisors.
    if cols < 1 or rows < 1 or w // max(cols, 1) < 1 or h // max(rows, 1) < 1:
        return (1, 1)
    return (cols, rows)


def cell_bounds(size, n):
    """n contiguous integer boundaries over [0,size]; step = size//n (engine's
    integer frame size), last cell absorbs the non-divisible remainder."""
    step = size // n
    b = [i * step for i in range(n)] + [size]
    return b

# --------------------------------------------------------------------------- #
#  classification / transparency
# --------------------------------------------------------------------------- #
def classify(im):
    if im.mode == "L":
        return "gray"
    arr = np.array(im.convert("RGB"))
    if np.all(arr == MAGENTA, axis=-1).mean() > 0.005:
        return "magenta"
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        return "rgba"
    return "opaque"


def bleed(rgb, transparent, iters=24):
    """Fill transparent pixels from filled neighbours so upscaling never blends
    toward the key colour (no halos)."""
    out = rgb.astype(np.float32)
    filled = ~transparent
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for _ in range(iters):
        if filled.all():
            break
        nb = np.zeros_like(out); cnt = np.zeros(out.shape[:2], np.float32)
        for dy, dx in offs:
            s = np.roll(np.roll(out, dy, 0), dx, 1)
            mk = np.roll(np.roll(filled, dy, 0), dx, 1).astype(np.float32)
            nb += s * mk[..., None]; cnt += mk
        new = (~filled) & (cnt > 0)
        out[new] = nb[new] / cnt[new][..., None]
        filled |= new
    return np.clip(out, 0, 255).astype(np.uint8)


def lanczos_rgb(arr, scale=SCALE):
    im = Image.fromarray(arr, "RGB")
    return np.array(im.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.LANCZOS))


def lanczos_alpha(a, scale=SCALE):
    im = Image.fromarray(a, "L")
    return np.array(im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.LANCZOS))

# --------------------------------------------------------------------------- #
#  the model
# --------------------------------------------------------------------------- #
class Upscaler:
    def __init__(self, path):
        import torch
        from spandrel import ModelLoader, ImageModelDescriptor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        md = ModelLoader().load_from_file(path)
        assert isinstance(md, ImageModelDescriptor), type(md)
        assert md.scale == SCALE, f"model scale {md.scale} != {SCALE}"
        md.to(self.dev).eval()
        self.half = (self.dev == "cuda")
        if self.half:
            md.model.half()
        self.md = md
        # Sprites are all different sizes, so each is a fresh conv shape. cudnn.benchmark
        # (MIOpen exhaustive autotune per shape) then costs ~1-2s EVERY image and never
        # amortises — the dominant cost. Off by default: use immediate/heuristic kernels
        # (runtime is negligible on these tiny tensors anyway). Set GSR_BENCHMARK=1 for the
        # rare same-shape-heavy workload.
        torch.backends.cudnn.benchmark = os.environ.get("GSR_BENCHMARK", "0") == "1"
        self.tile = int(os.environ.get("GSR_TILE", "512"))   # max input side before tiling
        self.pad = 16
        # Over a long batch the ROCm/MIOpen conv path occasionally emits a garbage
        # tile (VRAM fragmentation under sustained mixed sizes — reproduces mid-run,
        # never in isolation). A correct 4x result, box-downscaled to the source, is
        # ~identical to it (err ~1-3/255); garbage differs wildly (err ~37-59). So we
        # verify every output, and on failure flush VRAM + retry, then fall back to
        # LANCZOS — no garbage can ever reach the archive.
        self.garbage_thresh = float(os.environ.get("GSR_GARBAGE_THRESH", "15"))
        self.retries = self.fallbacks = 0
        name = self.torch.cuda.get_device_name(0) if self.dev == "cuda" else "CPU"
        print(f"[gsr] model={os.path.basename(path)} dev={self.dev} half={self.half} "
              f"gpu={name} tile={self.tile}", flush=True)

    def empty(self):
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()

    def _plausible(self, src, out):
        """A real 4x SR, downscaled back, matches the source in the low frequencies."""
        h, w = src.shape[:2]
        down = np.asarray(Image.fromarray(out, "RGB").resize((w, h), Image.BOX), np.float32)
        return np.abs(down - src.astype(np.float32)).mean() < self.garbage_thresh

    def _fwd(self, t):
        torch = self.torch
        with torch.inference_mode():
            x = t.to(self.dev)
            x = x.half() if self.half else x.float()
            y = self.md(x)
            return y.float().clamp_(0, 1).cpu()

    def _batch_same_size(self, arrs):
        """arrs: list of HxWx3 uint8 (identical H,W) -> list of 4H x 4W x3 uint8."""
        torch = self.torch
        t = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous().float() / 255.0
        out = []
        # pixel budget per sub-batch to stay within VRAM (input px * batch).
        h, w = arrs[0].shape[:2]
        per = max(1, int(2_000_000 / max(h * w, 1)))
        for i in range(0, t.shape[0], per):
            y = self._fwd(t[i:i + per])
            y = (y.permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
            out.extend(list(y))
        return out

    def _tiled_one(self, arr):
        """Large image: overlap-tile so DAT/PLKSR memory stays bounded, no seams."""
        torch = self.torch
        h, w = arr.shape[:2]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        out = torch.zeros(1, 3, h * SCALE, w * SCALE)
        wsum = torch.zeros(1, 1, h * SCALE, w * SCALE)
        ts, pad = self.tile, self.pad
        ys = list(range(0, h, ts)); xs = list(range(0, w, ts))
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + ts, h), min(x0 + ts, w)
                yy0, xx0 = max(0, y0 - pad), max(0, x0 - pad)
                yy1, xx1 = min(h, y1 + pad), min(w, x1 + pad)
                sub = t[:, :, yy0:yy1, xx0:xx1]
                up = self._fwd(sub)
                # region of `up` that corresponds to the un-padded [y0:y1, x0:x1]
                oy0, ox0 = (y0 - yy0) * SCALE, (x0 - xx0) * SCALE
                oy1, ox1 = oy0 + (y1 - y0) * SCALE, ox0 + (x1 - x0) * SCALE
                out[:, :, y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] += up[:, :, oy0:oy1, ox0:ox1]
                wsum[:, :, y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] += 1
        out = (out / wsum.clamp_min(1)).clamp_(0, 1)
        return (out[0].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)

    def _up_many_raw(self, arrs):
        """Upscale a group of same-size small RGB tiles (batched) OR fall back to
        per-image tiling for large ones. Returns list aligned with `arrs`."""
        if not arrs:
            return []
        h, w = arrs[0].shape[:2]
        if max(h, w) <= self.tile and all(a.shape == arrs[0].shape for a in arrs):
            return self._batch_same_size(arrs)
        return [self._tiled_one(a) if max(a.shape[:2]) > self.tile
                else self._batch_same_size([a])[0] for a in arrs]

    def up_many(self, arrs):
        """As _up_many_raw, but verifies each output and repairs any garbage tile
        (flush VRAM + retry once; LANCZOS as a last resort). Guarantees no garbage."""
        outs = self._up_many_raw(arrs)
        bad = [i for i, (a, o) in enumerate(zip(arrs, outs)) if not self._plausible(a, o)]
        if bad:
            self.empty()
            for i in bad:
                o = self._up_many_raw([arrs[i]])[0]      # retry alone, fresh VRAM
                if self._plausible(arrs[i], o):
                    self.retries += 1
                else:
                    o = lanczos_rgb(arrs[i]); self.fallbacks += 1
                outs[i] = o
        return outs

# --------------------------------------------------------------------------- #
#  per-cell transparency prep / recombine
# --------------------------------------------------------------------------- #
def prepare_cell(cell, kind):
    """cell: PIL sub-image. -> (model_input_rgb_uint8 HxWx3, alpha_uint8 or None)."""
    if kind == "magenta":
        rgb = np.array(cell.convert("RGB"))
        mask = np.all(rgb == MAGENTA, axis=-1)
        return bleed(rgb, mask), np.where(mask, 0, 255).astype(np.uint8)
    if kind == "rgba":
        arr = np.array(cell.convert("RGBA"))
        rgb, a = arr[..., :3], arr[..., 3]
        return bleed(rgb, a < 8), a
    if kind == "gray":
        g = np.array(cell.convert("L"))
        return np.stack([g, g, g], -1), None
    return np.array(cell.convert("RGB")), None


def resize_exact(arr, h, w):
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    mode = "L" if arr.ndim == 2 else ("RGBA" if arr.shape[2] == 4 else "RGB")
    return np.array(Image.fromarray(arr, mode).resize((w, h), Image.LANCZOS))

# --------------------------------------------------------------------------- #
#  widescreen fit for 4:3 full-screen menus
# --------------------------------------------------------------------------- #
def compose_widescreen(up, logical):
    LW, LH = logical
    cover = up.copy()
    cw, ch = LW, int(up.height * LW / up.width)
    if ch < LH:
        ch, cw = LH, int(up.width * LH / up.height)
    cover = cover.resize((cw, ch)).crop(((cw - LW) // 2, (ch - LH) // 2,
                                         (cw - LW) // 2 + LW, (ch - LH) // 2 + LH))
    cover = cover.filter(ImageFilter.GaussianBlur(96))
    fw = int(up.width * LH / up.height)
    cover.paste(up.resize((fw, LH)), ((LW - fw) // 2, 0))
    return cover

# --------------------------------------------------------------------------- #
#  encode to the game's expected container
# --------------------------------------------------------------------------- #
def enc(img, fmt):
    b = io.BytesIO()
    if fmt == "JPEG":
        img.save(b, "JPEG", quality=94, subsampling=1)
    elif fmt == "TGA":
        img.save(b, "TGA")            # uncompressed 32-bit, like the game's own .tga
    else:
        img.save(b, "BMP")
    return b.getvalue()


def is_fullscreen(rel, size):
    if size != (800, 600):
        return False
    return rel in FS_FILES or any(rel.lower().startswith(d.lower()) for d in FS_DIRS)

# --------------------------------------------------------------------------- #
#  process one image -> (bytes, fmt)  |  None to leave it vanilla
# --------------------------------------------------------------------------- #
def process_image(rel, path, up, sheet_map):
    if any(rel.startswith(p) for p in SKIP_PREFIXES):
        return None
    im = Image.open(path)
    w, h = im.size
    kind = classify(im)
    fmt0 = im.format
    method = "lanczos" if (rel in TEXT_FILES or min(w, h) <= TINY) else "model"

    # full-screen 4:3 menu art -> upscale whole, fit to 16:9.
    if method == "model" and is_fullscreen(rel, (w, h)):
        rgb = np.array(im.convert("RGB"))
        up4 = up.up_many([rgb])[0]
        up4 = resize_exact(up4, 4 * h, 4 * w)
        canvas = compose_widescreen(Image.fromarray(up4, "RGB"),
                                    (SCALE * LOGICAL[0], SCALE * LOGICAL[1]))
        return enc(canvas, "JPEG"), "JPEG"

    cols, rows = (1, 1) if method == "lanczos" else layout_for(rel, w, h, sheet_map)
    xb, yb = cell_bounds(w, cols), cell_bounds(h, rows)

    # --- split into cells, prepare model inputs ---
    prepped = []            # (row, col, (cy0,cy1,cx0,cx1), model_rgb, alpha)
    for r in range(rows):
        for c in range(cols):
            cy0, cy1, cx0, cx1 = yb[r], yb[r + 1], xb[c], xb[c + 1]
            cell = im.crop((cx0, cy0, cx1, cy1))
            mrgb, alpha = prepare_cell(cell, kind)
            prepped.append((r, c, (cy0, cy1, cx0, cx1), mrgb, alpha))

    # --- upscale RGB (batched by identical size when using the model) ---
    inputs = [p[3] for p in prepped]
    if method == "model":
        outs = up.up_many(inputs)
    else:
        outs = [lanczos_rgb(a) for a in inputs]

    # --- allocate canvas by kind, paste cells ---
    W4, H4 = SCALE * w, SCALE * h
    if kind == "magenta":
        rgb_cv = np.empty((H4, W4, 3), np.uint8)
        a_cv = np.zeros((H4, W4), np.uint8)
    elif kind == "rgba":
        rgb_cv = np.empty((H4, W4, 3), np.uint8)
        a_cv = np.zeros((H4, W4), np.uint8)
    elif kind == "gray":
        l_cv = np.empty((H4, W4), np.uint8)
    else:
        rgb_cv = np.empty((H4, W4, 3), np.uint8)

    for (r, c, (cy0, cy1, cx0, cx1), _mrgb, alpha), rgb4 in zip(prepped, outs):
        ch, cw = (cy1 - cy0) * SCALE, (cx1 - cx0) * SCALE
        rgb4 = resize_exact(rgb4, ch, cw)
        Y0, X0 = cy0 * SCALE, cx0 * SCALE
        if kind == "gray":
            l_cv[Y0:Y0 + ch, X0:X0 + cw] = np.array(Image.fromarray(rgb4, "RGB").convert("L"))
        else:
            rgb_cv[Y0:Y0 + ch, X0:X0 + cw] = rgb4
            if kind in ("magenta", "rgba"):
                # alpha always LANCZOS: a hard key/edge must not be model-hallucinated.
                a_cv[Y0:Y0 + ch, X0:X0 + cw] = resize_exact(lanczos_alpha(alpha), ch, cw)

    # --- finalize to the game's container/format ---
    if kind == "magenta":
        rgb_cv[a_cv < 128] = MAGENTA
        return enc(Image.fromarray(rgb_cv, "RGB"), "BMP"), "BMP"
    if kind == "rgba":
        rgba = np.dstack([rgb_cv, a_cv])
        return enc(Image.fromarray(rgba, "RGBA"), "TGA"), "TGA"
    if kind == "gray":
        return enc(Image.fromarray(l_cv, "L"), "BMP"), "BMP"
    fmt = "JPEG" if fmt0 == "JPEG" else "BMP"
    return enc(Image.fromarray(rgb_cv, "RGB"), fmt), fmt

# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process at most N images")
    ap.add_argument("--only", default="", help="substring filter on rel path")
    ap.add_argument("--no-pack", action="store_true", help="fill cache but skip hd.dat")
    ap.add_argument("--samples", default="", help="also dump before/after PNGs here")
    ap.add_argument("--max-new", type=int, default=0,
                    help="stop after processing N uncached images (0=all). Lets a wrapper "
                         "restart the process for a fresh GPU context — mitigates the "
                         "cumulative ROCm/gfx1100 degradation seen over long runs.")
    args = ap.parse_args()

    E = config.ASSETS
    todo = []
    for dp, _, fs in os.walk(E):
        for fn in fs:
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                rel = os.path.relpath(os.path.join(dp, fn), E).replace("\\", "/")
                if args.only and args.only not in rel:
                    continue
                todo.append(rel)
    todo.sort()
    if args.limit:
        todo = todo[:args.limit]
    print(f"[gsr] assets={E}  images={len(todo)}  model={MODEL_NAME}", flush=True)

    sheet_map = build_sheet_map(E)
    print(f"[gsr] sprite-sheet layouts parsed: {len(sheet_map)}", flush=True)

    files, need = {}, []
    for rel in todo:
        if any(rel.startswith(p) for p in SKIP_PREFIXES):
            continue                                  # left vanilla (manual)
        cp = os.path.join(CACHE, rel)
        mp = cp + ".fmt"
        if os.path.exists(cp) and os.path.exists(mp):
            files[rel] = open(cp, "rb").read()
        else:
            need.append(rel)
    print(f"[gsr] cached={len(files)}  to-process={len(need)}", flush=True)

    up = Upscaler(MODEL_PATH) if need else None
    if args.samples:
        os.makedirs(args.samples, exist_ok=True)

    t0 = time.time(); done = 0; skipped = 0
    for rel in need:
        path = os.path.join(E, rel)
        try:
            res = process_image(rel, path, up, sheet_map)
        except Exception as ex:
            print(f"[gsr] FAIL {rel}: {ex}", flush=True); raise
        if res is None:
            skipped += 1; continue
        data, _fmt = res
        cp = os.path.join(CACHE, rel)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        with open(cp, "wb") as f:
            f.write(data)
        open(cp + ".fmt", "w").write(_fmt)
        files[rel] = data
        if args.samples:
            Image.open(path).convert("RGB").save(os.path.join(args.samples, rel.replace("/", "__") + ".in.png"))
            Image.open(io.BytesIO(data)).convert("RGB").save(os.path.join(args.samples, rel.replace("/", "__") + ".out.png"))
        done += 1
        if up:
            up.empty()                       # release VRAM between images (anti-fragmentation)
        if args.max_new and done >= args.max_new:
            print(f"[gsr] --max-new {args.max_new} reached; exiting for a fresh context", flush=True)
            break
        if done % 100 == 0:
            dt = time.time() - t0
            extra = f"  retries={up.retries} fallbacks={up.fallbacks}" if up else ""
            print(f"[gsr] {done}/{len(need)}  {dt:.0f}s  {done/dt:.1f} img/s{extra}", flush=True)

    dt = time.time() - t0
    rf = f"  (garbage repaired: {up.retries} retries, {up.fallbacks} LANCZOS fallbacks)" if up else ""
    print(f"[gsr] processed {done} ({skipped} left-vanilla) in {dt:.0f}s{rf}", flush=True)

    if args.no_pack:
        print("[gsr] --no-pack: hd.dat not written", flush=True); return
    from jngdat import pack
    out = config.OUT_DAT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = pack(files, out)
    print(f"[gsr] DONE {len(files)} files -> {out} ({n/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
