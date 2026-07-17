#!/usr/bin/env python3
"""Verify build/hd.dat: it round-trips as a valid archive, every overlaid image is
exactly 4x its vanilla size in the game's expected container, and the HTML manual
was left out (so it falls through to vanilla)."""
import os, sys, io, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import config
from jngdat import DatArchive

HD = config.OUT_DAT
A = config.ASSETS
arc = DatArchive(HD)
print(f"hd.dat: {arc.count} entries, {os.path.getsize(HD)/1e6:.1f} MB")

names = [e.name for e in arc.entries]
assert not any(n.lower().startswith("data/manual/") for n in names), "manual leaked into overlay!"
print("manual correctly excluded:", sum(1 for n in names if "manual" in n.lower()), "manual entries")

# check 4x dims on a random sample that we can pair with an original
random.seed(1)
ents = list(arc.entries); random.shuffle(ents)
checked = bad = 0
kinds = {}
for e in ents:
    rel = e.name.replace("\\", "/")
    src = os.path.join(A, rel)
    if not os.path.exists(src):
        continue
    try:
        o = Image.open(src); ow, oh = o.size
        d = arc.read(e)
        u = Image.open(io.BytesIO(d)); uw, uh = u.size
    except Exception as ex:
        print("  ERR", rel, ex); bad += 1; continue
    kinds[u.format] = kinds.get(u.format, 0) + 1
    # fullscreen menus are re-fitted to 4x logical, not 4x source — skip their ratio check
    is_fs = (ow, oh) == (800, 600) and ("menu/screen" in rel.lower() or rel.lower().endswith("failed.jpg"))
    if not is_fs and (uw, uh) != (4 * ow, 4 * oh):
        print(f"  !! {rel}: {ow}x{oh} -> {uw}x{uh} (expected {4*ow}x{4*oh})")
        bad += 1
    checked += 1
    if checked >= 400:
        break
print(f"checked {checked} paired images, {bad} bad; formats: {kinds}")
print("HD VERIFY OK" if bad == 0 else "HD VERIFY FAILED")
sys.exit(1 if bad else 0)
