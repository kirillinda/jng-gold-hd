#!/usr/bin/env bash
# Overlay the curated picks (lab-curator/chosen/) onto the GSR cache so the
# next pack step ships them. Re-run after any full cache rebuild — a rebuild
# regenerates the plain GSR files and silently drops the curated ones.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CHOSEN="$REPO/lab-curator/chosen"
CACHE="$REPO/${CURATOR_CACHE:-upscaled_gsr/4x-UltraSharpV2_Lite_aa}"

[ -d "$CHOSEN" ] || { echo "nothing curated yet ($CHOSEN missing)"; exit 1; }
[ -d "$CACHE" ] || { echo "cache missing: $CACHE"; exit 1; }

n=$(find "$CHOSEN" -type f | wc -l)
rsync -a --info=name "$CHOSEN/" "$CACHE/"
echo "applied $n curated file(s) onto $CACHE"
echo "now rebuild hd.dat (pack step) to ship them"
