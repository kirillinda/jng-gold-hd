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
# interrupted just run this again — it skips what's done. The worker also verifies
# every output and repairs anything implausible (retry, then LANCZOS), so no
# garbage can be packed — see build_hd_gsr.py.
#
# Usage:
#   tools/gsr/run.sh
# Env:
#   GSR_MODEL   model basename under tools/gsr/models/  (default: 4x-UltraSharpV2_Lite)
#               4x-UltraSharpV2       — DAT2, max quality, slower (attention, and
#                                       runs in FP32: it overflows FP16)
#               4x-UltraSharpV2_Lite  — RealPLKSR, crisp + fast on gfx1100 (default)
#   HF_TOKEN    HuggingFace token, used only if a model file is missing.
#   GSR_IMAGE   docker image tag (default: jng-gsr:rocm7)
#   GSR_GPU     HIP device index to pin to (default: auto-detected gfx1100 dGPU,
#               so an APU's integrated graphics is never used).
#   GSR_CHUNK   if set, process at most this many new images per container run
#               (each chunk gets a fresh HIP context; not needed normally).
#   GSR_WORKERS parallel shards over the asset list (default 6). The cache is one
#               file per asset so shards never collide. Full-build sweep from an
#               empty cache: 3 workers 226s (62% GPU busy), 4 -> 188s (74%),
#               6 -> 164s (86%), 8 -> 175s (contention). Set 1 to serialise.
#
# MIOpen runs in FAST find mode: kernels are chosen by heuristic instead of by
# benchmarking candidates on-device per tensor shape. Measured on identical
# 40-image work, fresh process each: hybrid find ~95s (even WITH a tuned user db
# present - the find results don't survive process restart in this MIOpen build),
# FAST 10.8-12.5s. The heuristic kernels cost nothing measurable at steady state
# here, and FAST needs no persistent database, so cold and warm builds are now
# the same speed. tools/gsr/.miopen is still mounted (git-ignored): FAST will use
# any find-db entries that do exist, and it keeps MIOpen's lockfiles out of /tmp.
# Override with GSR_FIND_MODE (1=normal, 3=hybrid, 5=dynamic-hybrid) to re-tune.
#
# Detail preservation (see the block comment in build_hd_gsr.py — these exist
# because the model, being trained to remove noise, was erasing 2-4px rivets,
# bolts and baked-in lettering and inventing smooth surfaces over them):
#   GSR_PRESCALE  NEAREST-upscale each image this much before the model and
#                 BOX-downscale after (default 2, 1 disables). Costs P^2 in model
#                 pixels. Both the plain and prescaled results are produced and
#                 the better one is kept PER IMAGE, so this is a ceiling, not a
#                 forced setting.
#   GSR_INJECT    1 (default) — add the source's own fine structure back on top
#                 of the model output, at a strength solved per image so the
#                 result's gradient energy matches the source's. 0 disables.
#   GSR_ENSEMBLE  aggregate N dihedral orientations (median) to cancel
#                 direction-dependent smearing. Default 1 (off); 8 costs 8x and
#                 the prescale already removes most of that artifact.
#   GSR_FP32      1 forces full precision (normally auto-detected per model).
#
# Throughput / tiling:
#   GSR_TILE      max model-INPUT side before an image is tiled (default 512).
#                 Raising this to use more VRAM makes it SLOWER, not faster —
#                 measured 300s vs 68s vs 35s for tile 2048/512/256 on the same
#                 large level art. Leave it alone.
#   GSR_PAD       context kept around each tile (default 64). RealPLKSR's large
#                 kernels need more than the old 16, which left faint seams.
#   GSR_BATCH_PX  input pixels per forward pass (default: sized from free VRAM).
#                 Mostly matters for sheets with many same-sized frames, which
#                 then go through in one batch. Backs off automatically on OOM.
#   GSR_BUCKET    1 (default) — round every model input up to a fixed ladder of
#                 shapes (254 spatial; 617 counting the batch dim, vs 1555
#                 unbucketed) so MIOpen stops re-selecting kernels per sprite
#                 size. Measured on 24 forwards of equal total pixels: same shape
#                 0.042s each, all-different shapes 1.263s each — a 30x penalty
#                 for shape churn alone. Costs ~5% wasted pixels on padding and
#                 changes output by at most 2/255 (fp16 noise). 0 only for A/B.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GSR_DIR="$REPO/tools/gsr"
MODEL="${GSR_MODEL:-4x-UltraSharpV2_Lite}"
IMAGE="${GSR_IMAGE:-jng-gsr:rocm7}"
HF_REPO="Kim2091/UltraSharpV2"
GSR_AA="${GSR_AA:-1}"                       # de-jagged sprite silhouettes (potrace)
GSR_WORKERS="${GSR_WORKERS:-6}"             # parallel shards (see the note above)
# NB: the `|| :` matters — under `set -e` a failing $(...) in an assignment kills
# the script, so plain `[ ... ] && echo` made every GSR_AA=0 run die silently.
CACHE="$REPO/upscaled_gsr/$MODEL$([ "$GSR_AA" = 1 ] && echo _aa || :)"

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

# MIOpen scratch (user db + lockfiles). With FAST find (see header) no state is
# required here — kernels are picked heuristically — but the dir keeps whatever
# MIOpen does write out of the container's ephemeral /tmp.
mkdir -p "$GSR_DIR/.miopen"

# 3. Build the image ---------------------------------------------------------
log "Building $IMAGE (ROCm 7.x + PyTorch + spandrel)"
docker build -t "$IMAGE" "$GSR_DIR"

# --device kfd/dri exposes the GPU to ROCm; the process must be in the host's
# render/video groups (pass numeric GIDs — the names don't exist in the image).
RENDER_GID="$(getent group render | cut -d: -f3)"; VIDEO_GID="$(getent group video | cut -d: -f3)"

# Pin the workload to the discrete GPU. /dev/dri exposes every render node, so a
# machine with an APU (this one: Ryzen Raphael gfx1036) presents a second ROCm
# agent. Find the gfx1100 KFD node and pin to it, so we can never dispatch onto
# integrated graphics — which also shares system RAM with the desktop.
GSR_GPU="${GSR_GPU:-$(python3 - <<'PY' 2>/dev/null || echo 0
import glob, re
for p in sorted(glob.glob('/sys/class/kfd/kfd/topology/nodes/*/properties')):
    if re.search(r'^gfx_target_version 110000$', open(p).read(), re.M):
        # HIP indexes GPU agents in KFD node order, skipping the CPU node.
        n = int(p.split('/')[-2])
        gpus = [int(q.split('/')[-2]) for q in sorted(glob.glob('/sys/class/kfd/kfd/topology/nodes/*/properties'))
                if 'gfx_target_version 0\n' not in open(q).read()]
        print(sorted(gpus).index(n)); break
else:
    print(0)
PY
)}"
run_worker(){   # extra args -> build_hd_gsr.py
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    ${RENDER_GID:+--group-add "$RENDER_GID"} ${VIDEO_GID:+--group-add "$VIDEO_GID"} \
    --security-opt seccomp=unconfined --shm-size=8g \
    -v "$REPO":/work -w /work \
    -e GSR_MODEL="$MODEL" -e GSR_AA="$GSR_AA" ${GSR_TILE:+-e GSR_TILE="$GSR_TILE"} \
    ${GSR_BATCH_PX:+-e GSR_BATCH_PX="$GSR_BATCH_PX"} ${GSR_PAD:+-e GSR_PAD="$GSR_PAD"} \
    ${GSR_BUCKET:+-e GSR_BUCKET="$GSR_BUCKET"} ${GSR_DEJAG_ALPHAMAX:+-e GSR_DEJAG_ALPHAMAX="$GSR_DEJAG_ALPHAMAX"} \
    ${GSR_PRESCALE:+-e GSR_PRESCALE="$GSR_PRESCALE"} \
    ${GSR_INJECT:+-e GSR_INJECT="$GSR_INJECT"} \
    ${GSR_ENSEMBLE:+-e GSR_ENSEMBLE="$GSR_ENSEMBLE"} \
    ${GSR_FIDELITY_MARGIN:+-e GSR_FIDELITY_MARGIN="$GSR_FIDELITY_MARGIN"} \
    ${GSR_FP32:+-e GSR_FP32="$GSR_FP32"} \
    -e HIP_VISIBLE_DEVICES="$GSR_GPU" \
    -e MIOPEN_USER_DB_PATH=/work/tools/gsr/.miopen \
    -e MIOPEN_CUSTOM_CACHE_DIR=/work/tools/gsr/.miopen \
    -e MIOPEN_FIND_MODE="${GSR_FIND_MODE:-2}" \
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
  log "Upscaling on the GPU (model=$MODEL, $total images, $GSR_WORKERS worker(s))"
  if [ "$GSR_WORKERS" -gt 1 ]; then
    # The cache is one file per asset, so shards never collide and a crashed
    # worker just leaves its share uncached for the next run to pick up.
    pids=()
    for i in $(seq 0 $((GSR_WORKERS - 1))); do
      run_worker --no-pack --shard "$i/$GSR_WORKERS" & pids+=($!)
    done
    ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
    [ "$ok" = 1 ] || log "a worker exited non-zero — re-run to resume from cache"
  else
    run_worker --no-pack
  fi
fi

c="$(count)"
[ "$c" -ge "$total" ] || die "generation incomplete ($c / $total) — run this again to resume from cache"

# 5. Pack hd.dat (everything cached now; no GPU needed) ----------------------
log "Packing build/hd.dat"
run_worker

log "hd.dat written to build/hd.dat"
