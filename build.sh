#!/usr/bin/env bash
#
# build.sh — one-shot builder for the Jets'n'Guns Gold HD + widescreen mod.
#
# Usage:
#   ./build.sh [MODEL_NAME]
#
#   MODEL_NAME   Optional. Name of a realesrgan-ncnn model (a MODEL.param/MODEL.bin
#                pair) under tools/upscaler/models/. If omitted, the default model
#                the mod was released with (4x_NMKD-Siax_200k) is downloaded and used.
#
# Environment overrides:
#   JNG_GAME_DIR   Path to the game install (default: the standard Steam location).
#
# What it does:
#   1. sets up a Python venv with Pillow + numpy
#   2. downloads realesrgan-ncnn-vulkan (Vulkan GPU upscaler) if missing
#   3. resolves / downloads the upscale model
#   4. upscales every asset 4x and packs the HD override archive  (build/hd.dat)
#   5. binary-patches the game executable                          (build/jng_gold)
#        - patch_hd.py:         draw 4x art at its original on-screen size
#        - patch_widescreen.py: kill the leftover hardcoded 800x600 gameplay bounds
#   6. assembles two deliverables under dist/:
#        dist/mod-dropin/    only the changed files (+ install/uninstall scripts)
#        dist/patched-game/  a full, ready-to-run copy of the patched game
#
# Assumes the Linux Steam build of Jets'n'Guns Gold, version 1.308 ST.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

GAME_DIR="${JNG_GAME_DIR:-$HOME/.local/share/Steam/steamapps/common/JnG Gold}"
# HD art backend:
#   gsr  (default) — modern GAN super-resolution (spandrel/DAT2/RealPLKSR) in a
#                    ROCm Docker container; crisp, sheet-aware. See tools/gsr/.
#   ncnn           — the original realesrgan-ncnn-vulkan path (soft/"soapy").
HD_BACKEND="${HD_BACKEND:-gsr}"
MODEL="${1:-4x_NMKD-Siax_200k}"
DEFAULT_MODEL="4x_NMKD-Siax_200k"
UPSC_DIR="tools/upscaler"
MODELS_DIR="$UPSC_DIR/models"
VENV="tools/venv"
PY="$VENV/bin/python"

log(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

[ -d "$GAME_DIR" ] || die "game not found at '$GAME_DIR' (set JNG_GAME_DIR)"
[ -f "$GAME_DIR/jng_gold" ] || die "'$GAME_DIR/jng_gold' missing — is this the Linux build?"

# 1. Python environment ------------------------------------------------------
log "Python venv + dependencies"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r tools/requirements.txt

# 2-3. ncnn upscaler binary + model — only for the legacy ncnn backend -------
if [ "$HD_BACKEND" = ncnn ]; then
  if [ ! -x "$UPSC_DIR/realesrgan-ncnn-vulkan" ]; then
    log "Downloading realesrgan-ncnn-vulkan"
    mkdir -p "$UPSC_DIR"
    url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
    curl -sL -o /tmp/realesrgan.zip "$url"
    unzip -o -q /tmp/realesrgan.zip -d "$UPSC_DIR"
    chmod +x "$UPSC_DIR/realesrgan-ncnn-vulkan"
  fi
  if [ ! -f "$MODELS_DIR/$MODEL.param" ]; then
    if [ "$MODEL" != "$DEFAULT_MODEL" ]; then
      die "model '$MODEL' not found in $MODELS_DIR (drop the .param/.bin there, or omit for the default)"
    fi
    log "Downloading default model $DEFAULT_MODEL"
    base="https://github.com/upscayl/custom-models/raw/main/models"
    for ext in param bin; do
      curl -sL -o "$MODELS_DIR/$DEFAULT_MODEL.$ext" "$base/$DEFAULT_MODEL.$ext"
    done
  fi
  log "Using ncnn model: $MODEL"
fi

# 4. Unpack the game's assets (from YOUR copy) if not already present --------
if [ -z "$(ls -A assets/DATA 2>/dev/null)" ]; then
  log "Unpacking assets from your game into assets/"
  JNG_GAME_DIR="$GAME_DIR" "$PY" tools/extract.py
fi

# 5. Build the HD override archive ------------------------------------------
if [ "$HD_BACKEND" = gsr ]; then
  log "Upscaling assets and packing build/hd.dat (GSR: ROCm Docker + GAN model)"
  GSR_MODEL="${GSR_MODEL:-4x-UltraSharpV2_Lite}" tools/gsr/run.sh
else
  log "Upscaling assets and packing build/hd.dat (ncnn, this uses the GPU)"
  HD_MODEL="$MODEL" JNG_GAME_DIR="$GAME_DIR" "$PY" tools/build_batch.py
fi

# 6. Patch the game binary ---------------------------------------------------
# Always patch from the STOCK binary. If the mod is already installed, the
# original is preserved as jng_gold.orig; patching an already-patched binary
# would fail the safety check in patch_hd.py.
SRC_BIN="$GAME_DIR/jng_gold"
[ -f "$GAME_DIR/jng_gold.orig" ] && SRC_BIN="$GAME_DIR/jng_gold.orig"
log "Patching game binary ($SRC_BIN) -> build/jng_gold"
"$PY" tools/patch_hd.py "$SRC_BIN" "build/jng_gold.hd"
"$PY" tools/patch_widescreen.py "build/jng_gold.hd" "build/jng_gold"
rm -f "build/jng_gold.hd"

# 7. Assemble deliverables ---------------------------------------------------
log "Assembling dist/"
DROPIN="dist/mod-dropin"; FULL="dist/patched-game"
rm -rf "$DROPIN" "$FULL"; mkdir -p "$DROPIN" "$FULL"

# ws.dat: the level defs whose coordinates were authored for an 800-wide screen, re-authored
# for the target Width. Kept OUT of hd.dat and listed first, so changing the resolution is a
# one-second rebuild rather than re-upscaling 1930 images. Both it and Game.cfg take Width
# from tools/config.py, so they always agree.
log "Building build/ws.dat (resolution-dependent defs)"
"$PY" tools/make_widescreen_defs.py build/ws.dat

# Data.ini loads the overlays first (first match wins) and the widescreen config.
printf 'data_file = ws.dat\ndata_file = hd.dat\ndata_file = update.dat\ndata_file = jng.dat\n' > "$DROPIN/Data.ini"
"$PY" tools/make_gamecfg.py > "$DROPIN/Game.cfg"
cp build/hd.dat build/ws.dat build/jng_gold "$DROPIN/"
cp tools/install.sh tools/uninstall.sh "$DROPIN/"; chmod +x "$DROPIN"/*.sh
cat > "$DROPIN/README.txt" <<EOF
Jets'n'Guns Gold HD + Widescreen — drop-in mod
Copy these files into your game folder:
  $GAME_DIR
run ./install.sh from inside that folder (backs up originals), or copy manually:
  jng_gold  hd.dat  ws.dat  Data.ini  Game.cfg
Uninstall with ./uninstall.sh (restores the .orig backups).
EOF

# Full ready-to-run copy of the patched game. Copy the stock game, excluding any
# dev/backup artifacts a modded install may contain, then overlay the patched files.
cp -a "$GAME_DIR/." "$FULL/"
rm -f "$FULL"/*.orig "$FULL"/*.bak "$FULL/jng_gold_hd" "$FULL/hd_test.dat" \
      "$FULL/game.log" "$FULL/steam_appid.txt"
rm -rf "$FULL/screenshots" "$FULL/saves"
cp -f "$DROPIN/jng_gold" "$DROPIN/hd.dat" "$DROPIN/ws.dat" "$DROPIN/Data.ini" "$DROPIN/Game.cfg" "$FULL/"

log "Done."
echo "  dist/mod-dropin/    -> copy into your game folder (or run its install.sh)"
echo "  dist/patched-game/  -> a complete, ready-to-run patched game"
