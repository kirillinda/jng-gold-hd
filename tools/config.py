#!/usr/bin/env python3
"""Shared configuration / paths for the JnG Gold HD mod tools.

Everything is derived from the repository location so the project is portable;
each value can be overridden with an environment variable. Defaults assume the
Linux Steam build of Jets'n'Guns Gold (v1.308 ST) at its standard install path.
"""
import os

# Repository root = parent of this tools/ directory.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the game is installed (needed to read the original .dat archives and to
# patch/install). Override with JNG_GAME_DIR for a non-standard Steam library.
GAME_DIR = os.environ.get(
    "JNG_GAME_DIR",
    os.path.expanduser("~/.local/share/Steam/steamapps/common/JnG Gold"),
)

# Unpacked original assets (checked into the repo so they can be edited directly).
ASSETS = os.environ.get("JNG_ASSETS", os.path.join(REPO, "assets"))

# Upscale model name (a *.param/*.bin pair under MODELS_DIR). This is the model
# the released mod was built with; build.sh downloads it if absent.
MODEL = os.environ.get("HD_MODEL", "4x_NMKD-Siax_200k")

# realesrgan-ncnn-vulkan binary + its models directory (downloaded by build.sh).
UPSCALER = os.environ.get("JNG_UPSCALER", os.path.join(REPO, "tools/upscaler/realesrgan-ncnn-vulkan"))
MODELS_DIR = os.environ.get("JNG_MODELS", os.path.join(REPO, "tools/upscaler/models"))

# Per-model upscale cache (regenerable; git-ignored) and build scratch/output.
CACHE_DIR = os.path.join(REPO, "upscaled", MODEL)
BUILD_DIR = os.environ.get("JNG_BUILD", os.path.join(REPO, "build"))

# Where the finished overlay is written. build.sh copies it into dist/ and the game.
OUT_DAT = os.environ.get("JNG_OUT_DAT", os.path.join(BUILD_DIR, "hd.dat"))

# --- Mod parameters -------------------------------------------------------
# Hor+ widescreen logical resolution: native height (600), width widened to 16:9.
LOGICAL = (1067, 600)
# Upscale factor. MUST match the binary patch (which divides texture dims by 4).
SCALE = 4
