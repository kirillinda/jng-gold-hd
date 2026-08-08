#!/usr/bin/env python3
"""Full-archive quality scan: every overlaid image, box-downscaled back to its
source size, must match the source. Alpha-aware — magenta/keyed pixels are
excluded, since the overlay legitimately replaces them with a real alpha edge.

A correct 4x upscale scores ~1-3/255. Corruption scores 40+. This catches the
sub-threshold corruption that the in-pipeline guard (threshold 25) lets through.
"""
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import config
from jngdat import DatArchive

MAGENTA = (255, 0, 255)
arc = DatArchive(config.OUT_DAT)
print(f"hd.dat: {arc.count} entries, {os.path.getsize(config.OUT_DAT)/1e6:.1f} MB", flush=True)

rows, missing, errs = [], 0, 0
for e in arc.entries:
    rel = e.name.replace("\\", "/")
    src = os.path.join(config.ASSETS, rel)
    if not os.path.exists(src):
        missing += 1
        continue
    try:
        o = Image.open(src)
        ow, oh = o.size
        u = Image.open(io.BytesIO(arc.read(e)))
        if u.size != (4 * ow, 4 * oh):      # fullscreen refits etc.
            continue
        # Build a validity mask from the SOURCE: skip color-keyed pixels.
        o_rgb = np.asarray(o.convert("RGB"), np.uint8)
        keyed = np.all(o_rgb == MAGENTA, axis=-1)
        if o.mode == "RGBA":
            keyed |= np.asarray(o, np.uint8)[..., 3] < 128
        valid = ~keyed
        if valid.sum() < 16:
            continue
        down = np.asarray(u.convert("RGB").resize((ow, oh), Image.BOX), np.float32)
        err = float(np.abs(down - o_rgb.astype(np.float32))[valid].mean())
        rows.append((err, rel))
    except Exception as ex:
        print(f"  ERR {rel}: {ex}", flush=True)
        errs += 1

rows.sort(reverse=True)
e = np.array([r[0] for r in rows])
print(f"\nscanned {len(rows)} paired images ({missing} unpaired, {errs} errors)")
print(f"  median={np.median(e):.2f}  p99={np.percentile(e,99):.2f}  max={e.max():.2f}")
print(f"  >=40 (corruption):     {(e>=40).sum()}")
print(f"  >=25 (guard threshold):{(e>=25).sum()}")
print(f"  >=15:                  {(e>=15).sum()}")
print("\nworst 15:")
for err, rel in rows[:15]:
    print(f"  {err:7.2f}  {rel}")
print("\nSCAN CLEAN" if (e >= 40).sum() == 0 else "\nSCAN FOUND CORRUPTION")
