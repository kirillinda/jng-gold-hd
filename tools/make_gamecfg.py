#!/usr/bin/env python3
"""Emit a Game.cfg (to stdout) preconfigured for Hor+ widescreen.

Starts from the game's own Game.cfg.default and injects the logical render
resolution (Width/Height). The engine renders into glOrtho(0,Width,Height,0),
so setting Width to 16:9 of the native 600 height gives widescreen without
distortion; the window can be any size. See docs/HOW_IT_WORKS.md.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

W, H = config.LOGICAL
inject = f"\n# --- HD mod: Hor+ widescreen logical resolution ---\nWidth {W}\nHeight {H}\nBPP 32\n"

src = os.path.join(config.GAME_DIR, "Game.cfg.default")
base = open(src).read() if os.path.exists(src) else "windowed 1\nratio43 0\nvsync 1\ndetail_level 4\n"

if "ratio43" in base:                      # place it right after the aspect flag
    out = base.replace("ratio43 0", "ratio43 0" + inject, 1).replace("ratio43 1", "ratio43 0" + inject, 1)
else:
    out = base + inject
sys.stdout.write(out)
