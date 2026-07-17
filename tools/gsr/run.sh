#!/usr/bin/env bash
#
# run.sh — build the ROCm GPU-upscaler image and produce build/hd.dat from the
# unpacked assets/, using the modern GAN super-resolution pipeline (spandrel).
#
# Isolation: all GPU/ML work happens inside a Docker container built on the
# official ROCm 7.x PyTorch base — nothing is installed on the host. The repo is
# bind-mounted so the container reads assets/ and writes build/hd.dat in place.
#
# Resumable: every image is cached under upscaled_gsr/<model>/, so if the run is
# interrupted (e.g. a GPU hiccup) just run this again — it skips what's done. The
# worker also self-heals: it detects and repairs any corrupt tile the ROCm conv
# path emits (retry, then LANCZOS) — see build_hd_gsr.py — so no garbage is packed.
#
# Usage:
#   tools/gsr/run.sh
# Env:
#   GSR_MODEL   model basename under tools/gsr/models/  (default: 4x-UltraSharpV2_Lite)
#               4x-UltraSharpV2       — DAT2, max quality, slower (attention)
#               4x-UltraSharpV2_Lite  — RealPLKSR, crisp + fast on gfx1100 (default)
#   HF_TOKEN    HuggingFace token, used only if a model file is missing.
#   GSR_IMAGE   docker image tag (default: jng-gsr:rocm7)
#   GSR_CHUNK   if set, process at most this many new images per container run
#               (a wrapper for a fresh GPU context; only needed if your card
#               degrades/crashes under long sustained ROCm load).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GSR_DIR="$REPO/tools/gsr"
MODEL="${GSR_MODEL:-4x-UltraSharpV2_Lite}"
IMAGE="${GSR_IMAGE:-jng-gsr:rocm7}"
HF_REPO="Kim2091/UltraSharpV2"
CACHE="$REPO/upscaled_gsr/$MODEL"

log(){ printf '\n\033[1;35m[gsr] %s\033[0m\n' "$*"; }
die(){ printf '\033[1;31m[gsr] error: %s\033[0m\n' "$*" >&2; exit 1; }

# 1. Model weights present? (fetch once, git-ignored) ------------------------
mkdir -p "$GSR_DIR/models"
mf="$GSR_DIR/models/$MODEL.safetensors"
if [ ! -f "$mf" ]; then
  log "Fetching model $MODEL from HuggingFace ($HF_REPO)"
  auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  curl -fSL "${auth[@]}" -o "$mf" \
    "https://huggingface.co/$HF_REPO/resolve/main/$MODEL.safetensors" \
    || die "model download failed (set HF_TOKEN, or drop $MODEL.safetensors into $GSR_DIR/models/)"
fi

# 2. Assets unpacked? --------------------------------------------------------
[ -n "$(ls -A "$REPO/assets/DATA" 2>/dev/null)" ] || die "assets/ empty — run tools/extract.py first"

# 3. Build the image ---------------------------------------------------------
log "Building $IMAGE (ROCm 7.x + PyTorch + spandrel)"
docker build -t "$IMAGE" "$GSR_DIR"

# --device kfd/dri exposes the GPU to ROCm; the process must be in the host's
# render/video groups (pass numeric GIDs — the names don't exist in the image).
# MIOpen db -> /tmp (its permission-fixup fails on a bind mount).
RENDER_GID="$(getent group render | cut -d: -f3)"; VIDEO_GID="$(getent group video | cut -d: -f3)"
run_worker(){   # extra args -> build_hd_gsr.py
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    ${RENDER_GID:+--group-add "$RENDER_GID"} ${VIDEO_GID:+--group-add "$VIDEO_GID"} \
    --security-opt seccomp=unconfined --shm-size=8g \
    -v "$REPO":/work -w /work \
    -e GSR_MODEL="$MODEL" ${GSR_TILE:+-e GSR_TILE="$GSR_TILE"} -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -e MIOPEN_USER_DB_PATH=/tmp/miopen -e MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen \
    "$IMAGE" python tools/gsr/build_hd_gsr.py "$@"
}

# 4. Upscale every asset 4x (GPU) -------------------------------------------
count(){ find "$CACHE" -type f ! -name '*.fmt' 2>/dev/null | wc -l; }
total="$(cd "$REPO" && find assets -type f \( -iname '*.bmp' -o -iname '*.tga' -o -iname '*.jpg' \
        -o -iname '*.jpeg' -o -iname '*.gif' \) ! -path 'assets/DATA/manual/*' | wc -l)"
if [ -n "${GSR_CHUNK:-}" ]; then           # opt-in: fresh context per chunk
  prev=-1
  while c="$(count)"; [ "$c" -lt "$total" ]; do
    log "cache $c / $total (chunk $GSR_CHUNK)"
    [ "$c" -eq "$prev" ] && die "no progress last chunk — check the GPU/logs"
    prev="$c"; run_worker --no-pack --max-new "$GSR_CHUNK" || log "chunk crashed — resuming"
  done
else                                        # default: one pass (re-run to resume)
  log "Upscaling on the GPU (model=$MODEL, $total images)"
  run_worker --no-pack
fi

c="$(count)"
[ "$c" -ge "$total" ] || die "generation incomplete ($c / $total) — the GPU may have hiccuped; run this again to resume from cache"

# 5. Pack hd.dat (everything cached now; no GPU needed) ----------------------
log "Packing build/hd.dat"
run_worker

log "hd.dat written to build/hd.dat"
