#!/usr/bin/env bash
# Install the HD mod into the current directory (run from your JnG Gold folder,
# or from the mod-dropin folder pointed at the game). Backs up originals to *.orig.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAME="${1:-$PWD}"
[ -f "$GAME/jng_gold" ] || { echo "Run from your JnG Gold folder, or pass its path: ./install.sh /path/to/JnG Gold"; exit 1; }
for f in jng_gold Data.ini Game.cfg; do
  [ -f "$GAME/$f" ] && [ ! -f "$GAME/$f.orig" ] && cp "$GAME/$f" "$GAME/$f.orig" && echo "backed up $f -> $f.orig"
done
for f in jng_gold hd.dat ws.dat Data.ini Game.cfg; do
  cp "$HERE/$f" "$GAME/$f" && echo "installed $f"
done
echo "Done. Launch normally (Steam). Uninstall with ./uninstall.sh"
