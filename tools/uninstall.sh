#!/usr/bin/env bash
# Restore the stock game: put back the *.orig backups and remove the overlay.
set -euo pipefail
GAME="${1:-$PWD}"
for f in jng_gold Data.ini Game.cfg; do
  [ -f "$GAME/$f.orig" ] && mv -f "$GAME/$f.orig" "$GAME/$f" && echo "restored $f"
done
rm -f "$GAME/hd.dat" && echo "removed hd.dat"
echo "Stock game restored."
