#!/usr/bin/env python3
"""Full HD overlay build using realesrgan BATCH mode (model loaded once, GPU reused).
Preprocess all images -> one GPU batch -> postprocess -> pack hd.dat.

Every image becomes exactly 4x (paired with the /4 binary patch). Transparency:
  magenta -> bleed+alpha RGBA through model -> re-key to exact-magenta BMP
  gray/rgba/opaque -> mode-appropriate; full-screen 4:3 UI -> widescreen 4268x2400 JPEG
Tiny (<=8px) images bypass the model (LANCZOS via upscale_sprite). Cached per model.
"""
import os, sys, io, shutil, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageFilter
from jngdat import pack
from upscale import classify, _bleed, MAGENTA, upscale_sprite
import config

E = config.ASSETS
RE = config.UPSCALER
MODELS = config.MODELS_DIR
MODEL = config.MODEL
CACHE = config.CACHE_DIR
BIN = os.path.join(config.BUILD_DIR, "batch_in"); BOUT = os.path.join(config.BUILD_DIR, "batch_out")
LOGICAL = config.LOGICAL; SCALE = config.SCALE
IMG_EXT = (".bmp", ".tga", ".jpg", ".gif")
FS_DIRS = ("DATA/menu/screen/", "DATA/menu/screen2006/"); FS_FILES = ("DATA/menu/failed.jpg",)

def is_fullscreen(rel, size):
    if rel in FS_FILES: return size == (800, 600)
    return any(rel.lower().startswith(d.lower()) for d in FS_DIRS) and size == (800, 600)

def enc(img, fmt):
    b = io.BytesIO()
    if fmt == "JPEG": img.save(b, "JPEG", quality=94, subsampling=1)
    elif fmt == "PNG": img.save(b, "PNG")
    else: img.save(b, "BMP")
    return b.getvalue()

def compose_widescreen(up, logical):
    LW, LH = logical
    cover = up.copy()
    cw, ch = LW, int(up.height * LW / up.width)
    if ch < LH: ch, cw = LH, int(up.width * LH / up.height)
    cover = cover.resize((cw, ch)).crop(((cw-LW)//2, (ch-LH)//2, (cw-LW)//2+LW, (ch-LH)//2+LH))
    cover = cover.filter(ImageFilter.GaussianBlur(96))
    fw = int(up.width * LH / up.height)
    cover.paste(up.resize((fw, LH)), ((LW-fw)//2, 0))
    return cover

def main():
    todo = []
    for dp, _, fs in os.walk(E):
        for fn in fs:
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                todo.append(os.path.relpath(os.path.join(dp, fn), E).replace("\\", "/"))
    todo.sort()
    print(f"images: {len(todo)}  model={MODEL}", flush=True)

    files = {}; need = []
    for rel in todo:
        cp = os.path.join(CACHE, rel)
        if os.path.exists(cp):
            files[rel] = open(cp, "rb").read()
        else:
            need.append(rel)
    print(f"cached: {len(files)}  to-process: {len(need)}", flush=True)

    shutil.rmtree(BIN, ignore_errors=True); os.makedirs(BIN)
    shutil.rmtree(BOUT, ignore_errors=True); os.makedirs(BOUT)
    meta = {}; tiny = []; t0 = time.time()
    for i, rel in enumerate(need):
        im = Image.open(os.path.join(E, rel)); w, h = im.size
        kind = classify(im); fs = is_fullscreen(rel, (w, h)); fmt0 = im.format
        if min(w, h) <= 8:
            tiny.append((rel, kind, fmt0)); continue
        if kind == "magenta":
            rgb = np.array(im.convert("RGB")); tr = np.all(rgb == MAGENTA, axis=-1)
            inp = Image.fromarray(_bleed(rgb, tr)).convert("RGBA")
            inp.putalpha(Image.fromarray(np.where(tr, 0, 255).astype("uint8")))
        elif kind == "rgba": inp = im.convert("RGBA")
        else: inp = im.convert("RGB")
        fn = f"{i:05d}.png"; inp.save(os.path.join(BIN, fn))
        meta[fn] = (rel, kind, fs, (w, h), fmt0)
    print(f"preprocessed {len(meta)} (+{len(tiny)} tiny) in {time.time()-t0:.0f}s; batching...", flush=True)

    if meta:
        t = time.time()
        subprocess.run([RE, "-i", BIN, "-o", BOUT, "-n", MODEL, "-s", "4", "-g", "0", "-m", MODELS],
                       check=True)
        print(f"GPU batch {len(meta)} imgs in {time.time()-t:.0f}s; postprocessing...", flush=True)

    for fn, (rel, kind, fs, (w, h), fmt0) in meta.items():
        up = Image.open(os.path.join(BOUT, fn))
        if up.size != (4*w, 4*h): up = up.resize((4*w, 4*h), Image.LANCZOS)
        if fs:
            data = enc(compose_widescreen(up.convert("RGB"), (4*LOGICAL[0], 4*LOGICAL[1])), "JPEG")
        elif kind == "magenta":
            arr = np.array(up.convert("RGBA")); rgb = arr[..., :3].copy()
            rgb[arr[..., 3] < 128] = MAGENTA
            data = enc(Image.fromarray(rgb, "RGB"), "BMP")
        elif kind == "gray": data = enc(up.convert("L"), "BMP")
        elif kind == "rgba": data = enc(up.convert("RGBA"), "PNG")
        else: data = enc(up.convert("RGB"), "JPEG" if fmt0 == "JPEG" else "BMP")
        cp = os.path.join(CACHE, rel); os.makedirs(os.path.dirname(cp), exist_ok=True)
        open(cp, "wb").write(data); files[rel] = data

    for rel, kind, fmt0 in tiny:                     # LANCZOS, magenta-safe
        up, fmt = upscale_sprite(Image.open(os.path.join(E, rel)), scale=4, model=MODEL)
        data = enc(up, fmt)
        cp = os.path.join(CACHE, rel); os.makedirs(os.path.dirname(cp), exist_ok=True)
        open(cp, "wb").write(data); files[rel] = data

    shutil.rmtree(BIN, ignore_errors=True); shutil.rmtree(BOUT, ignore_errors=True)
    out = config.OUT_DAT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = pack(files, out)
    print(f"\nDONE {len(files)} files -> {out} ({n/1e6:.1f}MB) total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
