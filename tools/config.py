#!/usr/bin/env python3
"""Shared configuration / paths for the JnG Gold HD mod tools.

Everything is derived from the repository location so the project is portable;
each value can be overridden with an environment variable. Cross-platform: the
Windows and Linux Steam builds of Jets'n'Guns Gold (v1.308 ST) are both
supported, and the OS-specific defaults (game path, executable name, upscaler
binary name) are selected automatically below.
"""
import os, sys

IS_WINDOWS = (os.name == "nt")

# Repository root = parent of this tools/ directory.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The game executable's basename differs per platform (PE vs ELF). Override with
# JNG_GAME_EXE if needed.
GAME_EXE = os.environ.get("JNG_GAME_EXE", "jng_gold.exe" if IS_WINDOWS else "jng_gold")


def _default_game_dir():
    if IS_WINDOWS:
        # Standard Steam library; other drives (e.g. a D: library) are common,
        # so callers usually set JNG_GAME_DIR. This is only the fallback.
        pf = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        return os.path.join(pf, "Steam", "steamapps", "common", "JnG Gold")
    return os.path.expanduser("~/.local/share/Steam/steamapps/common/JnG Gold")


# Where the game is installed (needed to read the original .dat archives and to
# patch/install). Override with JNG_GAME_DIR for a non-standard Steam library.
GAME_DIR = os.environ.get("JNG_GAME_DIR", _default_game_dir())

# Unpacked original assets (checked into the repo so they can be edited directly).
ASSETS = os.environ.get("JNG_ASSETS", os.path.join(REPO, "assets"))

# Upscale model name (a *.param/*.bin pair under MODELS_DIR). This is the model
# the released mod was built with; the builder downloads it if absent.
MODEL = os.environ.get("HD_MODEL", "4x_NMKD-Siax_200k")

# realesrgan-ncnn-vulkan binary (+ its models directory), fetched by the builder.
# The binary has a .exe suffix on Windows.
_UPSCALER_BIN = "realesrgan-ncnn-vulkan.exe" if IS_WINDOWS else "realesrgan-ncnn-vulkan"
UPSCALER = os.environ.get("JNG_UPSCALER", os.path.join(REPO, "tools", "upscaler", _UPSCALER_BIN))
MODELS_DIR = os.environ.get("JNG_MODELS", os.path.join(REPO, "tools", "upscaler", "models"))

# Per-model upscale cache (regenerable; git-ignored) and build scratch/output.
CACHE_DIR = os.path.join(REPO, "upscaled", MODEL)
BUILD_DIR = os.environ.get("JNG_BUILD", os.path.join(REPO, "build"))

# Where the finished overlay is written. The builder copies it into dist/ and the game.
OUT_DAT = os.environ.get("JNG_OUT_DAT", os.path.join(BUILD_DIR, "hd.dat"))

# --- Mod parameters -------------------------------------------------------
# Hor+ widescreen logical resolution: native height (600), width widened to 16:9.
LOGICAL = (1067, 600)
# Upscale factor. MUST match the binary patch (which divides texture dims by 4).
SCALE = 4
