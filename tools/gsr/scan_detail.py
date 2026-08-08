#!/usr/bin/env python3
"""Rank every cached asset by how much SMALL STRUCTURE the upscaler destroyed.

scan_quality.py answers "is the output corrupt?" (low-frequency fidelity). This
answers a different and subtler question: "did the rivets survive?"

Box-downscale the 4x output back to the source grid and compare GRADIENT MAPS,
not pixels. A faithful upscale reproduces the source's fine structure, so its
downscaled gradient map matches the source's. A model that erased 3px rivets and
resynthesised a smooth surface still scores a near-perfect pixel MAE — the mean
colour is right — but its gradient energy collapses. That collapse is exactly the
"bolts became blobs / isometric went flat" complaint, made measurable.

  ret  gradient-energy retention.  1.0 = structure survived, 0.6 = 40% erased.
  cor  correlation with the source gradient map. Catches the other failure:
       detail that is present but in the WRONG PLACE (hallucinated texture).

  python tools/gsr/scan_detail.py [--cache DIR] [--top 40] [--csv out.csv]
"""
import os, sys, argparse, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import config

MAGENTA = (255, 0, 255)


def grad(g):
    gy, gx = np.gradient(g.astype(np.float32))
    return np.abs(gy) + np.abs(gx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        config.REPO, "upscaled_gsr", "4x-UltraSharpV2_Lite_aa"))
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    rows = []
    for dp, _, fs in os.walk(args.cache):
        for fn in fs:
            if fn.endswith(".fmt"):
                continue
            up = os.path.join(dp, fn)
            rel = os.path.relpath(up, args.cache).replace("\\", "/")
            src = os.path.join(config.ASSETS, rel)
            if not os.path.exists(src):
                continue
            try:
                o = Image.open(src)
                w, h = o.size
                u = Image.open(up)
                if u.size != (4 * w, 4 * h):
                    continue                       # widescreen refits etc.
                orgb = np.asarray(o.convert("RGB"), np.uint8)
                keyed = np.all(orgb == MAGENTA, axis=-1)
                if o.mode == "RGBA":
                    keyed |= np.asarray(o, np.uint8)[..., 3] < 128
                valid = ~keyed
                if valid.sum() < 256:
                    continue
                d = np.asarray(u.convert("RGB").resize((w, h), Image.BOX), np.uint8)
                gs = grad(np.asarray(Image.fromarray(orgb).convert("L")))
                gd = grad(np.asarray(Image.fromarray(d).convert("L")))
                a, b = gs[valid], gd[valid]
                if a.mean() < 1.0:                 # flat art: nothing to preserve
                    continue
                ret = float(b.mean() / a.mean())
                cor = float(np.corrcoef(a, b)[0, 1])
                mae = float(np.abs(d.astype(np.float32)
                                   - orgb.astype(np.float32))[valid].mean())
                rows.append((ret, cor, mae, float(a.mean()), w, h, rel))
            except Exception as ex:
                print(f"  ERR {rel}: {ex}", flush=True)

    rows.sort(key=lambda r: r[0])
    ret = np.array([r[0] for r in rows])
    cor = np.array([r[1] for r in rows])
    print(f"\n{len(rows)} images  cache={args.cache}")
    print(f"  retention  median={np.median(ret):.3f}  p10={np.percentile(ret,10):.3f}"
          f"  min={ret.min():.3f}")
    print(f"  correlation median={np.median(cor):.3f}  p10={np.percentile(cor,10):.3f}")
    print(f"  images losing >30% of their fine structure: {(ret<0.70).sum()}")
    print(f"  images losing >50%:                         {(ret<0.50).sum()}")
    print(f"\nworst {args.top} (most structure destroyed):")
    print(f"  {'ret':>5} {'cor':>5} {'mae':>6} {'detail':>6}  size        asset")
    for r, c, m, e, w, h, rel in rows[:args.top]:
        print(f"  {r:5.3f} {c:5.3f} {m:6.2f} {e:6.1f}  {w:4d}x{h:<5d}  {rel}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["ret", "cor", "mae", "src_detail", "w", "h", "rel"])
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
