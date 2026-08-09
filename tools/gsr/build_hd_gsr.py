#!/usr/bin/env python3
"""GPU super-resolution HD overlay builder for Jets'n'Guns Gold.

Replaces the old realesrgan-ncnn (NMKD-Siax) path, whose output was soft/"soapy".
Runs a modern GAN-trained super-resolution model through PyTorch/spandrel on the
7900 XTX (ROCm), in FP16 so conv/matmul use the RDNA3 WMMA matrix cores.

Default model: 4x-UltraSharpV2_Lite (RealPLKSR, conv-only) — crisp, detail-
synthesising, and — being attention-free — far faster on gfx1100, where
Flash-Attention isn't a win. 4x-UltraSharpV2 (DAT2) is available via GSR_MODEL
for maximum quality where the extra time is acceptable — it overflows FP16 and is
detected and run in FP32 automatically.

What makes this correct for a 20-year-old sprite engine (not just "run a model"):

  * EXACT 4x. The binary patch divides every texture dimension by 4, so every
    image the game loads must be exactly 4x its original size. Output dims are
    always (4w, 4h).

  * SMALL STRUCTURE SURVIVES. The model is trained to remove noise, and 2-4px
    hand-placed detail (rivets, dither, isometric facets, baked-in lettering)
    looks like noise to it. Left alone it erases that detail and invents a smooth
    surface. See the "detail preservation" section below for the two corrections,
    and tools/gsr/lab.py / scan_detail.py for the measurements behind them.

  * ANIMATION SHEETS ARE SPLIT PER FRAME. A sheet is a grid; upscaling it whole
    smears detail across frame borders (the reported "smeared" look). We cut each
    frame out, upscale it alone, and lay the frames back on the SAME grid the
    engine reads: frame width = w // cols (integer, as the C++ engine computes
    it), so 4x frame i lands at 4*i*(w//cols). Cells are contiguous and the last
    cell absorbs any non-divisible remainder, so the 4x canvas tiles exactly with
    no seams and no smear. Layout comes from either defs syntax — `frames_wh =
    N, cols, rows` or a bare `frames = N, -1, -1` horizontal strip.

  * TRANSPARENCY PRESERVED PER FLAVOUR (as the old tool did, kept faithful):
      - magenta color-key (255,0,255): bleed sprite colour into the keyed region
        (so the model never blends toward magenta -> no pink halo), upscale RGB,
        upscale the alpha mask with LANCZOS, then re-key exact magenta -> BMP.
      - RGBA (.tga): RGB (edge-bled) via model + alpha via LANCZOS -> TGA.
      The container always matches the path's extension, so the shipped art opens
      in an ordinary image editor: alpha-bearing .bmp paths get a real 32-bit BMP
      with a BITMAPV4HEADER alpha mask (see enc_bmp32), not TGA bytes.
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
import os, sys, io, re, glob, json, time, shutil, struct, argparse, subprocess, tempfile
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
LOGICAL = config.LOGICAL

# Anti-alias sprite edges. The game keys magenta -> fully-transparent (1-bit), so
# the model's crisp interior meets a hard, staircased silhouette. With AA on we
# instead deliver magenta sprites as a real smooth-alpha 32-bit TGA (bled colour +
# a de-jagged coverage alpha, see dejag_alpha): the engine alpha-blends it (same
# path the .tga fog sprites use, via its IMG_LoadTGA_RW fallback), so edges are
# smooth. Separate cache dir so AA and legacy 1-bit outputs don't collide.
# Default on, matching run.sh, so a direct invocation builds the same archive.
AA = os.environ.get("GSR_AA", "1") == "1"
CACHE = os.path.join(config.REPO, "upscaled_gsr", MODEL_NAME + ("_aa" if AA else ""))

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
MIN_CELL = 4        # a declared layout implying a smaller cell is a misparse

# --------------------------------------------------------------------------- #
#  sprite-sheet layout map: normalized bitmap path -> (cols, rows)
# --------------------------------------------------------------------------- #
def build_sheet_map(assets_dir):
    """normalized bitmap path -> (cols, rows), from BOTH declaration syntaxes.

    The defs declare sheet layout two different ways, and missing the second one
    meant 141 sheets — every walking/dying character, among others — were upscaled
    as one image, letting the model smear detail across frame borders:

      frames_wh = N, cols, rows   an explicit grid.
      frames    = N, -1, -1       N frames, layout implied. Measured against the
                                  actual bitmaps (adjacent-frame similarity under
                                  each hypothesis) this is a HORIZONTAL strip: 50
                                  assets score decisively that way and none score
                                  decisively vertical. The rest are ~16px-tall
                                  character strips where a vertical read would
                                  imply 2px-tall frames, which is nonsense.
    """
    sprite_re = re.compile(r'\[\s*sprite\s*=', re.I)
    bitmap_re = re.compile(r'bitmap\s*=\s*(.+)', re.I)
    fwh_re = re.compile(r'frames_wh\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', re.I)
    # Not frames_wh: 'frames' there is followed by '_', never by '='.
    fr_re = re.compile(r'(?<![_a-z])frames\s*=\s*(\d+)\s*(?:,\s*-1\s*,\s*-1)?\s*$', re.I)
    m = {}

    def put(cur, cols, rows):
        # Highest (cols*rows) wins on conflict: a real split beats a stray 1,1,1.
        prev = m.get(cur)
        if prev is None or cols * rows > prev[0] * prev[1]:
            m[cur] = (cols, rows)

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
                put(cur, cols, rows); cur = None; continue
            g = fr_re.search(line.strip())
            if g and cur:
                n = int(g.group(1))
                if n >= 1:              # 'frames = 0' means "not animated"
                    put(cur, n, 1)
                cur = None
    return m


def layout_for(rel, w, h, sheet_map):
    if rel in AVATAR_GRIDS:
        return AVATAR_GRIDS[rel]
    key = rel.replace("\\", "/").lower()
    cols, rows = sheet_map.get(key, (1, 1))
    # Never split into sub-cells the model can't handle, and guard bad divisors.
    # MIN_CELL: a declared layout that implies a 1-3px cell is a misparse, not a
    # sheet — splitting on it would slice through artwork and seam it. Upscale
    # whole instead.
    if cols < 1 or rows < 1 or w // max(cols, 1) < MIN_CELL or h // max(rows, 1) < MIN_CELL:
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
#  detail preservation
# --------------------------------------------------------------------------- #
# UltraSharpV2 is trained with a degradation pipeline (JPEG / noise / blur), so it
# has learned that small high-frequency structure is NOISE to be removed before
# resynthesising a clean surface. Hand-placed 2-4px game art — rivets, bolts,
# dither, isometric facet shading, baked-in lettering — sits squarely in that
# band. The model therefore erases real detail and invents a smooth surface over
# it: round speckles came out as diagonal gashes, a square icon ring came out
# circular, an astronaut's helmet melted. Two independent corrections, both chosen
# by measurement in tools/gsr/lab.py and tracked by tools/gsr/scan_detail.py:
#
#  PRESCALE  NEAREST-upscale the input P x before the model and BOX-downscale the
#            result by P after. This invents nothing — every source pixel becomes
#            an exact PxP block — but it presents each feature at P times the
#            size, so a 3px rivet reaches the model as a 6px structure, above the
#            band it learned to destroy. Costs P^2 in model pixels.
#
#  INJECT    Add the source's own high-frequency content back on top of the
#            model's output: keep the model's clean gradients and smooth edges,
#            restore the structure it removed. Crucially the strength is NOT a
#            fixed knob — it is solved per image so the result's gradient energy
#            matches the source's. Assets the model already handled well receive
#            almost none; the ones it flattened receive a lot. That makes this the
#            cheap, measurable form of "context-aware" upscaling: it adapts to
#            what each image actually needs, with no classifier and no seams.
#
# Measured retention (1.0 = fine structure fully preserved), worst cases:
#   asset                 before  after
#   DATA/gui/sideex.bmp    0.72    0.97
#   special.bonus/hp100    0.50    0.94
#   man.space/walk.bmp     0.79    0.99
DETAIL_SIGMA = float(os.environ.get("GSR_DETAIL_SIGMA", "2.0"))   # 4x-space; ~½ src px
INJECT_MAX = float(os.environ.get("GSR_INJECT_MAX", "1.5"))


def _luma(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114   # as PIL "L"


def grad_energy(a):
    gy, gx = np.gradient(_luma(a))
    return float((np.abs(gy) + np.abs(gx)).mean())


def box_down(a, s=SCALE):
    H, W = a.shape[:2]
    return a[:H - H % s, :W - W % s].reshape(H // s, s, W // s, s, -1).mean((1, 3))


def detail_layer(src, sigma=DETAIL_SIGMA):
    """The source's own fine structure, carried to 4x. BICUBIC as the carrier
    (less ringing to re-inject than LANCZOS)."""
    h, w = src.shape[:2]
    r = Image.fromarray(src, "RGB").resize((w * SCALE, h * SCALE), Image.BICUBIC)
    return (np.asarray(r, np.float32)
            - np.asarray(r.filter(ImageFilter.GaussianBlur(sigma)), np.float32))


def solve_inject_alpha(src, out4, D, target=1.0, amax=INJECT_MAX):
    """The injection strength that makes this cell's gradient energy match the
    source's. BOX-downscaling is linear, so alpha can be solved entirely on the
    small source grid — down(out + a*D) == down(out) + a*down(D) — with the
    full-size result composed only once. Retention rises monotonically with
    alpha, so bisect."""
    gs = grad_energy(src.astype(np.float32))
    if gs < 1e-3:
        return 0.0                                    # flat art: nothing to restore
    db, dD = box_down(out4.astype(np.float32)), box_down(D)

    def ret(a):
        return grad_energy(np.clip(db + a * dD, 0, 255)) / gs

    if ret(amax) < target:
        return amax                                   # unreachable target; cap
    lo, hi = 0.0, amax
    for _ in range(12):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if ret(m) < target else (lo, m)
    return 0.5 * (lo + hi)


def inject_many(srcs, outs, sigma=DETAIL_SIGMA, target=1.0, amax=INJECT_MAX):
    """Restore the fine structure the model removed, across all cells of ONE image.

    The strength is solved per cell and then applied as a single area-weighted
    value to every cell. Solving it per cell and applying it per cell would be
    marginally more accurate in isolation, but a sheet's cells are consecutive
    animation frames — giving neighbouring frames different amounts of detail
    makes the texture pop as the sprite animates, which is precisely the class of
    artifact this pipeline exists to avoid. Consistency beats per-frame optimality.
    """
    Ds = [detail_layer(s, sigma) for s in srcs]
    alphas = [solve_inject_alpha(s, o, D, target, amax) for s, o, D in zip(srcs, outs, Ds)]
    w = [s.shape[0] * s.shape[1] for s in srcs]
    a = float(np.average(alphas, weights=w)) if alphas else 0.0
    if a <= 1e-3:
        return outs                                   # model already kept enough
    return [np.clip(o.astype(np.float32) + a * D, 0, 255).astype(np.uint8)
            for o, D in zip(outs, Ds)]


def dihedral(a, k):
    if k & 4: a = np.transpose(a, (1, 0, 2))
    if k & 1: a = a[::-1]
    if k & 2: a = a[:, ::-1]
    return np.ascontiguousarray(a)


def undihedral(a, k):
    if k & 2: a = a[:, ::-1]
    if k & 1: a = a[::-1]
    if k & 4: a = np.transpose(a, (1, 0, 2))
    return np.ascontiguousarray(a)


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


# potrace's corner threshold. This is NOT a cosmetic knob: at potrace's default
# of 1.0 the tracer treats almost every corner as a curve, which turned
# DATA/gui/sideex.bmp's SQUARE icon ring into a circle and its straight stem into
# an S-bend. Measured shape error (traced coverage box-downscaled vs the source
# mask) falls monotonically as it drops — but 0 means "all corners, no curves",
# which throws away the de-jagging this function exists for. 0.6 is where genuine
# right angles survive while an organic hull (air.zeppelin) still traces smooth:
#   asset                alphamax 1.0    0.6    0.0
#   gui/sideex.bmp           0.0228  0.0135  0.0126   (square ring -> circle at 1.0)
#   air.zeppelin.bmp         0.0139  0.0128  0.0109   (looks smooth at all three)
DEJAG_ALPHAMAX = float(os.environ.get("GSR_DEJAG_ALPHAMAX", "0.6"))


def dejag_alpha(a, scale=SCALE, alphamax=None, turd=2):
    """Turn a 1-bit color-key silhouette into a SMOOTH, continuous, anti-aliased
    edge (not a blurred staircase). potrace vectorises the low-res mask into
    Bezier curves — deliberately replacing the pixel staircase with smooth, but
    corner-aware, outlines — and we rasterise that at `scale`x with coverage AA.
    `a` is a 0/255 uint8 mask; returns a `scale`x uint8 alpha. Falls back to
    LANCZOS if potrace/rsvg are unavailable or error out."""
    if alphamax is None:
        alphamax = DEJAG_ALPHAMAX
    h, w = a.shape
    H, W = h * scale, w * scale
    m = a > 127
    if not m.any():
        return np.zeros((H, W), np.uint8)
    if m.all():
        return np.full((H, W), 255, np.uint8)
    try:
        with tempfile.TemporaryDirectory() as td:
            pbm, svg, png = td + "/m.pbm", td + "/m.svg", td + "/m.png"
            Image.fromarray(np.where(m, 0, 255).astype(np.uint8), "L").save(pbm)  # sprite=black
            # NO --tight: it crops the SVG to the shape bbox, so rsvg then rescales a
            # small part up to the full canvas (a nose/engine balloons into a stretched
            # blob). Keep the full-image coordinate frame so position/scale stay 1:1.
            _run(["potrace", pbm, "-s", "-o", svg,
                  f"--alphamax={alphamax}", f"--turdsize={turd}"])
            _run(["rsvg-convert", "-w", str(W), "-h", str(H), "-b", "white", "-o", png, svg])
            g = np.asarray(Image.open(png).convert("L"))
        out = (255 - g).astype(np.uint8)                 # black shape -> opaque
        if out.shape != (H, W):
            out = resize_exact(out, H, W)
        # Safety: if the vectorised silhouette drifts too far from the original
        # coverage (tiny/thin sprites potrace can't fit well), keep the faithful
        # LANCZOS edge instead — a correct shape beats a smooth-but-wrong one.
        oc = m.mean(); nc = (out > 127).mean()
        if oc > 0 and not (0.8 <= nc / oc <= 1.25):
            return lanczos_alpha(a, scale)
        return out
    except Exception:
        return lanczos_alpha(a, scale)

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
        self.md = md
        # FP16 so conv/matmul use the RDNA3 WMMA cores. But DAT2's attention blocks
        # overflow half precision and return all-NaN: the plausibility guard then
        # quietly swapped in LANCZOS for EVERY image, so GSR_MODEL=4x-UltraSharpV2
        # looked like it worked while doing no AI upscaling at all. Probe once at
        # load, and keep a runtime guard in _fwd for content-dependent overflow.
        self.half = (self.dev == "cuda") and os.environ.get("GSR_FP32", "0") != "1"
        if self.half:
            md.model.half()
            if not self._finite_in_fp16():
                print("[gsr] FP16 output is non-finite for this model -> using FP32",
                      flush=True)
                md.model.float()
                self.half = False
        # Sprites are all different sizes, so each is a fresh conv shape. cudnn.benchmark
        # (MIOpen exhaustive autotune per shape) then costs ~1-2s EVERY image and never
        # amortises — the dominant cost. Off by default: use immediate/heuristic kernels
        # (runtime is negligible on these tiny tensors anyway). Set GSR_BENCHMARK=1 for the
        # rare same-shape-heavy workload.
        torch.backends.cudnn.benchmark = os.environ.get("GSR_BENCHMARK", "0") == "1"
        # Max model-INPUT side before an image is cut into tiles.
        #
        # Counter-intuitively, spending VRAM here makes things SLOWER, measured on
        # the same 5 large level backgrounds with a fresh process each time:
        #   tile=2048 (nothing tiles)  300s      tile=512  68s      tile=256  35s
        # and for a single 512x512 asset, untiled 48s vs ~13s tiled. Big tensors
        # fall off whatever fast path MIOpen picks for the small uniform shapes, so
        # the tiled route wins comfortably. Do not "optimise" this upward.
        self.tile = int(os.environ.get("GSR_TILE", "512"))
        # Context kept around each tile. RealPLKSR uses partial LARGE kernels (up
        # to 17x17), so 16px of context is not enough — measured against an untiled
        # reference on a 512x512 asset, worst-case tile-boundary error:
        #   pad=16 -> 63/255    pad=32 -> 56/255    pad=64 -> 32/255
        # for +24% time. 16 was leaving faint seams on the big level art.
        self.pad = int(os.environ.get("GSR_PAD", "64"))
        # Input pixels per forward pass. These sprites are tiny and the model is
        # small, so throughput is dominated by per-launch latency, not FLOPs —
        # a bigger batch is close to free. Sized from the card rather than the old
        # hardcoded 2M, with _fwd_split as the safety net if the estimate is off.
        self.batch_px = int(os.environ.get("GSR_BATCH_PX", "0")) or self._vram_budget()
        # Cheap correctness check on every output. This started as a workaround for
        # a 15%-of-images garbage rate that turned out to be self-inflicted (an
        # expandable_segments allocator setting we no longer set — see Dockerfile);
        # with that fixed the observed rate is ~0. It stays as a guard, because a
        # silently-wrong tile would otherwise be baked into the shipped archive.
        # A correct 4x result, box-downscaled to the source, is ~identical to it
        # (err ~1-3/255); garbage differs wildly (err ~40-160). On failure we flush
        # VRAM + retry, then fall back to LANCZOS.
        # 25: measured legitimate worst case is 15.2 (assets/DATA/enemy/alien/worm1b.bmp,
        # deterministic across runs — a genuinely hard image, not corruption), real
        # garbage is 40+. 15 was too tight and needlessly LANCZOS'd worm1b.
        self.garbage_thresh = float(os.environ.get("GSR_GARBAGE_THRESH", "25"))
        self.retries = self.fallbacks = 0
        # Detail preservation — see the block comment above. prescale=2 is the
        # sweet spot: it fixes the structure loss for 4x the model pixels, where
        # prescale=3 costs 9x for a marginal further gain. ensemble>1 aggregates
        # the dihedral transforms (median) to cancel direction-dependent smearing;
        # it costs another Nx and prescale already removes most of that artifact,
        # so it is opt-in rather than default.
        self.prescale = max(1, int(os.environ.get("GSR_PRESCALE", "2")))
        # Which filter does the prescale. NEAREST, measured against lanczos/bicubic
        # in lab.py: those resamplers ring, and the model amplifies the overshoot
        # into crosshatch noise in dark areas (retention overshoots to ~1.05 and
        # correlation drops). NEAREST invents nothing, and the block edges it does
        # introduce are averaged straight back out by the BOX-downscale by p.
        self.prefilter = os.environ.get("GSR_PREFILTER", "nearest").lower()
        self.ensemble = max(1, int(os.environ.get("GSR_ENSEMBLE", "1")))
        self.inject = os.environ.get("GSR_INJECT", "1") == "1"
        # How much extra low-frequency error (0-255 mae) a candidate may carry
        # before it is rejected regardless of how much structure it preserves.
        self.fidelity_margin = float(os.environ.get("GSR_FIDELITY_MARGIN", "1.5"))
        self.picked = [0, 0]                          # [plain, prescaled]
        # Skip the prescaled pass when the plain one already kept the structure.
        # Measured: prescale=2 costs ~80% of all GPU time (40 images: 17.9s plain
        # only vs 47.2s computing both), so not computing it where it cannot help
        # is the largest saving available.
        self.shortcut_ret = float(os.environ.get("GSR_SHORTCUT_RET", "0.95"))
        self.shortcut = 0
        # Shape bucketing (see DIM_LADDER). GSR_BUCKET=0 disables it, which is
        # only useful for A/B-ing the effect — it is a large loss.
        self.bucket = os.environ.get("GSR_BUCKET", "1") == "1"
        # NHWC. MIOpen prefers it for many convolutions; whether it actually wins
        # on gfx1100 for this model is measured, not assumed.
        self.channels_last = os.environ.get("GSR_CHANNELS_LAST", "0") == "1"
        if self.channels_last:
            md.model.to(memory_format=torch.channels_last)
        name = self.torch.cuda.get_device_name(0) if self.dev == "cuda" else "CPU"
        print(f"[gsr] model={os.path.basename(path)} dev={self.dev} half={self.half} "
              f"gpu={name} tile={self.tile} batch_px={self.batch_px/1e6:.1f}M "
              f"bucket={self.bucket} prescale={self.prescale} ensemble={self.ensemble} "
              f"inject={self.inject}", flush=True)

    def empty(self):
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()

    def _plausible(self, src, out):
        """A real 4x SR, downscaled back, matches the source in the low frequencies."""
        h, w = src.shape[:2]
        down = np.asarray(Image.fromarray(out, "RGB").resize((w, h), Image.BOX), np.float32)
        return np.abs(down - src.astype(np.float32)).mean() < self.garbage_thresh

    _BYTES_PER_INPUT_PX = 768      # measured peak for the default model, fp16

    def _vram_budget(self):
        """How many input pixels to push through the model at once.

        Peak activation scales with input pixels: a few dozen feature channels
        live at input resolution, plus the 4x output. Spend most of the free VRAM
        and let the OOM backoff handle a bad estimate on an unusual model."""
        if self.dev != "cuda":
            return 2_000_000
        try:
            free, _total = self.torch.cuda.mem_get_info()
        except Exception:
            return 2_000_000
        return max(2_000_000, int(free * 0.6 / self._BYTES_PER_INPUT_PX))

    def _is_oom(self, e):
        oom = getattr(self.torch.cuda, "OutOfMemoryError", None)
        return ((oom is not None and isinstance(e, oom))
                or "out of memory" in str(e).lower())

    def _fwd_split(self, t):
        """_fwd, halving the batch on VRAM exhaustion, so the budget above can be
        aggressive without risking a hard failure mid-run."""
        try:
            return self._fwd(t)
        except Exception as e:
            if not self._is_oom(e) or t.shape[0] < 2:
                raise
            self.empty()
            n = t.shape[0] // 2
            print(f"[gsr] VRAM exhausted at batch {t.shape[0]} -> splitting", flush=True)
            return self.torch.cat([self._fwd_split(t[:n]), self._fwd_split(t[n:])])

    def _finite_in_fp16(self):
        torch = self.torch
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1, 3, 64, 64, generator=g).to(self.dev).half()
        with torch.inference_mode():
            return bool(torch.isfinite(self.md(x)).all())

    # Shape ladder. Every sprite is a different size, so without this the model
    # sees a brand-new tensor shape almost every call and MIOpen re-selects
    # kernels each time. Measured on 24 forwards of equal total pixels, fresh
    # process: all-same-shape 0.042s/image, all-different-shape 1.263s/image — a
    # 30x penalty for churn alone. Rounding up to a handful of shapes trades a
    # little wasted compute on the padding for that 30x.
    # Fine steps for small inputs, coarse for large: padding a 10x18 sprite cell
    # up to 128x128 is not a border effect, it is mostly invented content, and the
    # model's large kernels pull it in (measured: mean error 7.7/255 on
    # man.space/walk.bmp). Chosen over the alternatives on the real asset
    # distribution — 11304 model inputs, 1555 distinct shapes unbucketed:
    #   multiples of 64   89 shapes  1.11x px  worst dim blowup 16x
    #   multiples of 32  207 shapes  1.05x px  worst  8x
    #   this ladder      254 shapes  1.05x px  worst  2x   (p95 1.41x)
    # Note those are (H,W) counts. MIOpen keys on the FULL tensor shape, so the
    # batch dimension multiplies them: 254 spatial x the batch buckets actually
    # used works out to 617 distinct shapes over 4767 model calls. Still ~2.5x
    # fewer than the 1555 unbucketed, and the warm-up is front-loaded.
    DIM_LADDER = (8, 16, 24, 32, 40, 48, 56, 64,
                  96, 128, 160, 192, 224, 256,
                  320, 384, 448, 512, 576, 640)
    BATCH_BUCKETS = (1, 2, 4, 8, 16, 32)

    def _bucket_dim(self, v):
        if not self.bucket:
            return v
        for b in self.DIM_LADDER:
            if b >= v:
                return b
        return v                                  # off the top of the ladder: exact

    def _bucket_batch(self, hb, wb):
        """Largest batch bucket whose padded tensor still fits the VRAM budget."""
        cap = max(1, int(self.batch_px / max(hb * wb, 1)))
        fits = [b for b in self.BATCH_BUCKETS if b <= cap]
        return max(fits) if fits else 1

    def _pad_to_bucket(self, x):
        """Round (N,C,H,W) up to a ladder shape. Returns (padded, original_dims)."""
        n, _c, h, w = x.shape
        hb, wb = self._bucket_dim(h), self._bucket_dim(w)
        nb = next((b for b in self.BATCH_BUCKETS
                   if b >= n and b * hb * wb <= self.batch_px), n) if self.bucket else n
        if (nb, hb, wb) == (n, h, w):
            return x, None
        F = self.torch.nn.functional
        # Replicate, never zeros: a hard black frame would bleed into the real
        # border pixels through the model's large kernels.
        if (hb, wb) != (h, w):
            x = F.pad(x, (0, wb - w, 0, hb - h), mode="replicate")
        if nb > n:
            x = self.torch.cat([x, x[-1:].expand(nb - n, -1, -1, -1)], 0)
        return x, (n, h, w)

    def _fwd(self, t):
        torch = self.torch
        with torch.inference_mode():
            x = t.to(self.dev)
            x, orig = self._pad_to_bucket(x)
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            y = self.md(x.half() if self.half else x.float())
            if self.half and not torch.isfinite(y).all():
                # Content-dependent FP16 overflow the load-time probe missed.
                # Downgrade permanently rather than per-call: a model that
                # overflows once will keep doing it, and silently degrading to
                # LANCZOS is worse than being slower.
                print("[gsr] non-finite FP16 output -> switching to FP32", flush=True)
                self.md.model.float()
                self.half = False
                y = self.md(x.float())
            if orig is not None:                  # drop the bucket padding
                n, h, w = orig
                y = y[:n, :, :h * SCALE, :w * SCALE]
            return y.float().clamp_(0, 1).cpu()

    def _batch_same_size(self, arrs):
        """arrs: list of HxWx3 uint8 (identical H,W) -> list of 4H x 4W x3 uint8."""
        torch = self.torch
        t = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous().float() / 255.0
        out = []
        # Chunk on a batch-bucket boundary (sized against the PADDED dims), so
        # every full chunk is already a ladder shape and only the last one needs
        # padding up. Keeps VRAM bounded and the shape count small.
        h, w = arrs[0].shape[:2]
        per = self._bucket_batch(self._bucket_dim(h), self._bucket_dim(w))
        for i in range(0, t.shape[0], per):
            y = self._fwd_split(t[i:i + per])
            y = (y.permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
            out.extend(list(y))
        return out

    def _tiled_one(self, arr):
        """Tile, shrinking the tile permanently if even one tile won't fit."""
        try:
            return self._tiled_impl(arr)
        except Exception as e:
            if not self._is_oom(e) or self.tile <= 128:
                raise
            self.empty()
            self.tile = max(128, self.tile // 2)
            print(f"[gsr] VRAM exhausted while tiling -> tile={self.tile}", flush=True)
            return self._tiled_one(arr)

    def _tiled_impl(self, arr):
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

    def _up_many_guarded(self, arrs):
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

    _FILTERS = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC,
                "bilinear": Image.BILINEAR}

    def _prescale_one(self, a, p):
        if self.prefilter == "nearest":
            return np.repeat(np.repeat(a, p, 0), p, 1)
        h, w = a.shape[:2]
        f = self._FILTERS.get(self.prefilter, Image.LANCZOS)
        return np.asarray(Image.fromarray(a, "RGB").resize((w * p, h * p), f), np.uint8)

    def _model_4x(self, arrs, p):
        """Guarded 4x from the model, at prescale p, with the ensemble correction."""
        ins = [self._prescale_one(a, p) for a in arrs] if p > 1 else arrs

        if self.ensemble > 1:
            # Direction-dependent smearing is not equivariant to the dihedral
            # group, so the median across orientations cancels it (mean would just
            # average toward mush). Each orientation is a separate batched pass.
            acc = [[undihedral(o, k) for o in
                    self._up_many_guarded([dihedral(a, k) for a in ins])]
                   for k in range(min(self.ensemble, 8))]
            outs = [np.clip(np.median(np.stack(t).astype(np.float32), 0), 0, 255)
                    .astype(np.uint8) for t in zip(*acc)]
        else:
            outs = self._up_many_guarded(ins)

        if p > 1:                                   # back down to exactly 4x
            outs = [np.asarray(Image.fromarray(o, "RGB").resize(
                        (a.shape[1] * SCALE, a.shape[0] * SCALE), Image.BOX), np.uint8)
                    for a, o in zip(arrs, outs)]
        return outs

    @staticmethod
    def _score(src, out4):
        """(low-frequency fidelity, fine-structure retention) of a candidate."""
        s = src.astype(np.float32)
        d = box_down(out4.astype(np.float32))
        return (float(np.abs(d - s).mean()),
                grad_energy(d) / max(grad_energy(s), 1e-6))

    def _pick(self, srcs, cands):
        """Choose between the plain and prescaled candidates, once per image.

        Prescaling is a large win on detailed art but a small loss on smooth
        gradients (glow/flare sprites) and heavily-dithered flats, where there is
        no fine structure at risk and the model just over-reads the enlarged
        noise — measured over a 90-asset sweep: 36 improved, 4 regressed. So the
        choice is made per image instead of globally: take the best structure
        retention, but reject any candidate whose low-frequency fidelity is
        materially worse than the best on offer (that is the failure mode the
        regressions showed — e.g. Atarix's green LCD drifting olive).

        This is the "context-aware" idea done by measurement rather than by
        classifying objects: no labels, no per-class models, no seams.

        Scored over all the image's cells at once, not per cell: the cells are
        consecutive animation frames, and switching approach mid-animation would
        make the texture pop between frames."""
        agg = []
        w = [s.shape[0] * s.shape[1] for s in srcs]
        for cand in cands:
            sc = [self._score(s, o) for s, o in zip(srcs, cand)]
            agg.append((float(np.average([m for m, _ in sc], weights=w)),
                        float(np.average([r for _, r in sc], weights=w))))
        best_mae = min(m for m, _ in agg)
        ok = [i for i, (m, _) in enumerate(agg) if m <= best_mae + self.fidelity_margin]
        return min(ok, key=lambda i: abs(agg[i][1] - 1.0))

    def _needs_prescale(self, srcs, plain):
        """Is the expensive prescaled pass worth running for this image?

        Prescaling exists to repair GEOMETRY the model rewrites when features fall
        below its noise floor — a square ring coming out circular. If the plain
        result already retains the source's structure there is nothing to repair,
        and the detail injection covers the remaining texture. Since that pass is
        ~80% of GPU time, skipping it when it cannot help is the biggest available
        saving that does not trade away quality."""
        w = [s.shape[0] * s.shape[1] for s in srcs]
        ret = float(np.average([self._score(s, o)[1] for s, o in zip(srcs, plain)],
                               weights=w))
        if ret >= self.shortcut_ret:
            self.shortcut += 1
            return False
        return True

    def up_many(self, arrs):
        """Upscale a whole image's cells to exactly 4x, preserving fine structure.

        Every decision below is made once for the whole call — which is one image
        — so all of a sheet's frames are treated identically."""
        cands = [self._model_4x(arrs, 1)]
        if self.prescale > 1 and self._needs_prescale(arrs, cands[0]):
            cands.append(self._model_4x(arrs, self.prescale))
        k = self._pick(arrs, cands) if len(cands) > 1 else 0
        self.picked[k] += 1
        outs = cands[k]
        return inject_many(arrs, outs) if self.inject else outs

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
def enc_bmp32(rgba):
    """32-bit BGRA BMP carrying real alpha, with a BITMAPV4HEADER.

    Why not just Pillow's BMP writer: it emits a bare 40-byte BITMAPINFOHEADER
    with BI_RGB, which declares no alpha mask at all. Readers then have to guess
    whether the 4th byte means anything — SDL2 guesses with a heuristic, and most
    image viewers guess "no". A BITMAPV4HEADER states the RGBA masks explicitly,
    so both the game and an image editor agree.

    This replaces the previous trick of writing TGA bytes to a .bmp path. That
    worked only because CRXTexture::Load falls back to IMG_LoadTGA_RW when
    IMG_Load_RW fails to sniff the format — but it meant none of the sprites
    would open in an image viewer. The engine keeps the alpha either way: its
    colour-key pass masks alpha out of the comparison (and $0xffffff) and only
    overwrites exact key matches, leaving every other pixel's alpha untouched.
    """
    h, w = rgba.shape[:2]
    # BMP scanlines run bottom-up and store BGRA; rows are already 4-byte aligned.
    data = np.ascontiguousarray(rgba[::-1, :, [2, 1, 0, 3]]).tobytes()
    dib = struct.pack(
        "<IiiHHIIiiII" "IIII" "I" "36x" "III",
        108, w, h, 1, 32, 3, len(data), 2835, 2835, 0, 0,      # 3 = BI_BITFIELDS
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,   # R, G, B, A masks
        0x73524742,                                        # 'sRGB' (LCS_sRGB)
        0, 0, 0)                                           # gamma R/G/B (unused)
    off = 14 + len(dib)
    return struct.pack("<2sIHHI", b"BM", off + len(data), 0, 0, off) + dib + data


def enc(img, fmt):
    b = io.BytesIO()
    if fmt == "JPEG":
        img.save(b, "JPEG", quality=94, subsampling=1)
    elif fmt == "BMP32":
        return enc_bmp32(np.asarray(img.convert("RGBA"), np.uint8))
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
                # magenta is a 1-bit color-key silhouette -> de-jag into a smooth
                # continuous edge when AA is on; rgba already has soft model alpha.
                a4 = dejag_alpha(alpha) if (AA and kind == "magenta") else lanczos_alpha(alpha)
                a_cv[Y0:Y0 + ch, X0:X0 + cw] = resize_exact(a4, ch, cw)

    # --- finalize to the game's container/format ---
    if kind == "magenta":
        if AA:
            # Smooth-alpha output. Match the container to the path's extension so
            # the shipped art actually opens in a viewer/editor: a .bmp path gets
            # a real 32-bit BMP with an explicit alpha mask, a .tga path stays
            # TGA. (Previously every one of these was TGA bytes, including those
            # written to .bmp paths, which no image viewer would open.)
            rgba = Image.fromarray(np.dstack([rgb_cv, a_cv]), "RGBA")
            if rel.lower().endswith(".tga"):
                return enc(rgba, "TGA"), "TGA"
            return enc(rgba, "BMP32"), "BMP"
        rgb_cv[a_cv < 128] = MAGENTA                 # legacy 1-bit color-key BMP
        return enc(Image.fromarray(rgb_cv, "RGB"), "BMP"), "BMP"
    if kind == "rgba":
        rgba = Image.fromarray(np.dstack([rgb_cv, a_cv]), "RGBA")
        if rel.lower().endswith(".bmp"):
            return enc(rgba, "BMP32"), "BMP"
        return enc(rgba, "TGA"), "TGA"
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
    ap.add_argument("--shard", default="",
                    help="i/N — process only every Nth image, for parallel workers. "
                         "The cache is one file per asset, so shards never collide.")
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
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
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
    if up and sum(up.picked):
        pl, pr = up.picked
        print(f"[gsr] prescale skipped as unnecessary on {up.shortcut} images", flush=True)
        print(f"[gsr] candidate chosen per image: plain={pl} prescaled={pr} "
              f"({100 * pr / max(pl + pr, 1):.0f}% prescaled)", flush=True)

    if args.no_pack:
        print("[gsr] --no-pack: hd.dat not written", flush=True); return
    from jngdat import pack
    out = config.OUT_DAT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = pack(files, out)
    print(f"[gsr] DONE {len(files)} files -> {out} ({n/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
