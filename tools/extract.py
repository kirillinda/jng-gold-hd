#!/usr/bin/env python3
"""Extract every file from the game's .dat archives into assets/.

The mod ships the unpacked assets in the repo, but this lets you regenerate them
from your own copy of the game (or after a game update). Order matters: update.dat
is an override overlay that patches a couple of files in jng.dat, so it is applied
second (its entries win on lookup — see jngdat.DatArchive / the game's loader).
"""
import os, sys, json, hashlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jngdat import DatArchive
import config

def main():
    manifest = []
    ext_counts = collections.Counter()
    for datname in ("jng.dat", "update.dat"):
        path = os.path.join(config.GAME_DIR, datname)
        if not os.path.exists(path):
            print(f"! missing {path} — set JNG_GAME_DIR to your install", file=sys.stderr)
            sys.exit(1)
        arc = DatArchive(path)
        print(f"{datname}: {arc.count} entries")
        for e in arc.entries:
            data = arc.read(e)
            dst = os.path.join(config.ASSETS, e.name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
            ext_counts[os.path.splitext(e.name)[1].lower() or "(none)"] += 1
            manifest.append({"archive": datname, "name": e.name, "size": e.uncomp,
                             "sha1": hashlib.sha1(data).hexdigest()})
    with open(os.path.join(config.ASSETS, "_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=0)
    print(f"\nextracted {len(manifest)} files to {config.ASSETS}")
    for ext, n in ext_counts.most_common():
        print(f"  {n:5d}  {ext}")

if __name__ == "__main__":
    main()
