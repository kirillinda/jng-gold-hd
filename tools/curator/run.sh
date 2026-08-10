#!/usr/bin/env bash
# Launch the curator GUI (FLUX-assisted manual touch-up of GSR-upscaled
# assets). Serves http://localhost:7860 — open it in a browser.
#
# Prerequisites (all local, nothing is committed):
#   - the GSR cache:      upscaled_gsr/4x-UltraSharpV2_Lite_aa/  (tools/gsr/run.sh)
#   - extracted assets:   assets/
#   - FLUX.1-dev:         lab-flux/flux-dev/        (diffusers layout)
#   - the two LoRAs:      lab-flux/lora/ultrasharp-v{2,3}-flux.safetensors
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${CURATOR_IMAGE:-jng-curator:latest}"
PORT="${CURATOR_PORT:-7860}"

docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || docker build -t "$IMAGE" "$REPO/tools/curator"

RENDER_GID="$(getent group render | cut -d: -f3 || true)"
VIDEO_GID="$(getent group video | cut -d: -f3 || true)"

# Pin to the discrete gfx1100 card, never the APU (same logic as tools/gsr).
GSR_GPU="${GSR_GPU:-$(python3 - <<'PY' 2>/dev/null || echo 0
import glob, re
for p in sorted(glob.glob('/sys/class/kfd/kfd/topology/nodes/*/properties')):
    if re.search(r'^gfx_target_version 110000$', open(p).read(), re.M):
        n = int(p.split('/')[-2])
        gpus = [int(q.split('/')[-2]) for q in sorted(glob.glob('/sys/class/kfd/kfd/topology/nodes/*/properties'))
                if 'gfx_target_version 0\n' not in open(q).read()]
        print(sorted(gpus).index(n)); break
else:
    print(0)
PY
)}"

exec docker run --rm --name jng-curator \
  --device=/dev/kfd --device=/dev/dri \
  ${RENDER_GID:+--group-add "$RENDER_GID"} ${VIDEO_GID:+--group-add "$VIDEO_GID"} \
  --security-opt seccomp=unconfined --shm-size=8g \
  -v "$REPO":/work -w /work \
  -p "127.0.0.1:$PORT:7860" \
  -e HIP_VISIBLE_DEVICES="$GSR_GPU" \
  -e MIOPEN_USER_DB_PATH=/work/tools/gsr/.miopen \
  -e MIOPEN_CUSTOM_CACHE_DIR=/work/tools/gsr/.miopen \
  -e MIOPEN_FIND_MODE=2 \
  "$IMAGE" uvicorn --app-dir tools/curator server:app --host 0.0.0.0 --port 7860
