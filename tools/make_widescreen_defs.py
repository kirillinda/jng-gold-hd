#!/usr/bin/env python3
"""Re-author the level defs whose coordinates assume an 800-wide screen.

`patch_widescreen.py` fixes the engine's hardcoded 800s. But plenty of 4:3 assumptions live
in the game's *data*: the title screen is built out of level defs, and every level's ambient
particle field declares its spawn rectangle in screen space. Those place things in a
coordinate space where the screen is 800 units wide, so widening the viewport strands them.

Scope: the intro defs under `DATA/level/intro_1/`, plus the `[FIELD]` spawn rects in all 70
level files that declare one. Nothing global is touched (see WHY NOT `ld` below) — the rest
of the game's movement data is left exactly as authored.

The coordinates come in three flavours, and using the wrong transform on the wrong one is
how you break things, so they're handled explicitly rather than by a blanket regex:

  * ABSOLUTE screen positions (`absolute = 1` entries, `absolute_direct` path nodes, text
    `pos`). What to do depends on what the thing is anchored to:
      - centred composition  -> x += (Width-800)//2   [the logo, the ESC prompt]
      - right-edge anchored  -> x += (Width-800)       [the right-hand credits panel]
      - left-edge anchored   -> unchanged              [the left-hand credits panel]
  * SPAWN OFFSETS, relative to the spawn point — which `ParseNextStarter` correctly puts at
    the RIGHT EDGE (x = Width). An offset that was really measuring "off the LEFT edge" via
    a big negative number has to grow with the screen: x -= (Width-800).
  * The `-1` x sentinel (`pos = -1, 270`) — the engine centres those itself. Left alone.

What each fix addresses, all measured on a 1067x600 logical screen:

1. LOGO OFF-CENTRE. The logo is two enemy sprites (`jng_l1`/`jng_r1`, 120px each) whose
   paths end on absolute x=300 and x=420 — the pair straddles 800/2. They still landed on
   400 at 1067, i.e. 133px left of centre.  Fix: +133 -> 433 / 553 (verified in memory:
   both halves rest at exactly 433.00 / 553.00).
   NOT proportional scaling (x *= Width/800 -> 400/560): that pulls the halves 40px apart
   and tears the logo in half. The gap between them is a sprite width, not a screen
   fraction. Additive keeps them adjacent and preserves the composition's offset exactly.

2. LOGO HALVES ARRIVE AT DIFFERENT TIMES. Both spawn relative to the right edge:
   `jng_l pos = -950` and `jng_r pos = +70`. At 800 that's start -150 / 870 -> both travel
   450 -> they meet together. At 1067 it's start 117 (on-screen!) / 1137 -> 316 vs 584, so
   the left half lands early and it looks broken. `jng_l`'s offset is really "150px off the
   LEFT edge", so it must grow with the screen: -950 - 267 = -1217 -> start -150, travel
   583 vs 584. Symmetric again.

3. JETS VANISH MID-SCREEN. Enemies that fly across are removed when their *path* runs out
   (`CEnemy::NextNodePos` clears the alive flag), not when they cross an edge. The jets use
   the global straight-line path `ld` (reach 1000) at `path_scale = 100`, spawning at
   Width+100:
       800  screen: spawn  900 - 1000 = -100  (just off-screen, correct)
       1067 screen: spawn 1167 - 1000 = +167  (dies in mid-air)
   Measured under gdb: 7/7 jets despawned at x=167.4 — matches exactly. Fix: scale the
   intro jet's `path_scale` so its reach is Width+200, restoring the ±100 margins.
   Re-measured after the fix: 7/7 despawn at x=-103.

4. CREDITS / ESC PROMPT MISPLACED. The credits are two 150px panels alternating sides every
   29s: `titles_1` hugs the left edge (x=20), `titles_2` sits 50px off the right edge
   (x=600 at 800). At 1067 the right-hand one stranded mid-screen. Kept edge-anchored
   (titles_1 unchanged, titles_2 -> 867). The ESC prompt is a centred composition -> +133.

5. PARTICLES SPAWN MID-SCREEN (game-wide). Every level's ambient particle `[FIELD]` spawns
   from a screen-space rect — see fix_particle_fields(). Verified in the intro: 721/803
   spawns now land in 1067..1087 (was 800..820). The engine's `init_count` fill already
   spreads across the real Width, so only the ongoing spawn strip needed moving.

WHY NOT just lengthen the `ld` path (for 3): `ld` is the engine's *universal* straight-line
path — **1686 call sites** — and `path_scale` is used as a direction vector against it
(`0,100` = missiles straight up, `-100,100` = 45' diagonals, `100,40` = shallow dives).
`ld`'s `direct = -1000,-1000` only reads as "reach 1000" when the Y scale is 0. Changing its
dx would silently re-angle every diagonal and every missile in the game. Scaling one
level-local behavior is the correct blast radius.

Output: {archive_path: bytes} to merge into an overlay listed before jng.dat
(first-match-wins), so the originals are never modified.
"""
import re, os, sys, glob

BASE_W = 800            # the width these defs were authored against
LD_REACH = 1000         # horizontal reach of the global `ld` path at path_scale = 100
SPAWN_MARGIN = 100      # intro trigger bigjet_side: pos = 100, 0  (spawns at Width + 100)
EXIT_MARGIN = 100       # vanilla died at x = -100; keep that margin

PATH_DEF = "DATA/level/intro_1/path.txt"
BEHAVIOR_DEF = "DATA/level/intro_1/behavior.txt"
TRIGGER_DEF = "DATA/level/intro_1/trigger.txt"

CROSS_BEHAVIORS = {"bigjet_side"}       # intro behaviors that must fly clear of the screen

# (trigger, entry) -> how that entry's `pos` x must move. Entry None = every entry in it.
# Anything not listed is deliberately left alone (`jng_r`'s +70 and `bigjet_side`'s +100 are
# genuine off-the-right-edge margins and stay correct as-is).
CENTRED, ANCHOR_RIGHT, FROM_LEFT_EDGE = "centred", "anchor_right", "from_left_edge"
TRIGGER_RULES = {
    ("jng", "jng_l"):        FROM_LEFT_EDGE,   # -950 is really "150px off the LEFT edge"
    ("titles_2", None):      ANCHOR_RIGHT,     # right-hand credits panel
    ("esc_to_menu", None):   CENTRED,          # "PRESS 'ESC' TO ENTER MAIN MENU"
}


def _shift(x: int, rule: str, width: int) -> int:
    extra = width - BASE_W
    if rule == CENTRED:
        return x + extra // 2
    if rule == ANCHOR_RIGHT:
        return x + extra
    if rule == FROM_LEFT_EDGE:
        return x - extra
    raise ValueError(rule)


def recentre_absolute_paths(text: str, width: int) -> str:
    """`absolute[_direct] = x, y[, ...]` in the intro paths -> centred composition."""
    if width == BASE_W:
        return text

    def sub(m):
        nums = [n.strip() for n in m.group(2).split(",")]
        nums[0] = str(_shift(int(nums[0]), CENTRED, width))
        return f"{m.group(1)}={','.join(nums)}"

    # [ \t]* not \s* — \s* would swallow the newline and glue the next line on.
    return re.sub(r'\b(absolute_direct|absolute)[ \t]*=[ \t]*(-?\d+(?:[ \t]*,[ \t]*-?\d+)*)',
                  sub, text, flags=re.I)


def rescale_cross_behaviors(text: str, width: int) -> str:
    """Set path_scale.x on CROSS_BEHAVIORS so `ld`'s reach becomes Width + 200."""
    if width == BASE_W:
        return text
    scale = round((width + SPAWN_MARGIN + EXIT_MARGIN) / LD_REACH * 100)
    lines, cur = text.split("\n"), None
    for i, ln in enumerate(lines):
        m = re.match(r'\s*\[\s*behavior\s*=\s*([^\]]+)\]', ln, re.I)
        if m:
            cur = m.group(1).strip().lower()
            continue
        if cur in CROSS_BEHAVIORS:
            m = re.match(r'([ \t]*)path_scale([ \t]*=[ \t]*)(-?\d+)([ \t]*,.*)$', ln, re.I)
            if m:
                lines[i] = f"{m.group(1)}path_scale{m.group(2)}{scale}{m.group(4)}"
    return "\n".join(lines)


def fix_triggers(text: str, width: int) -> str:
    """Apply TRIGGER_RULES to each `pos = x, ...` inside its (trigger, entry) block."""
    if width == BASE_W:
        return text
    lines, trig, entry = text.split("\n"), None, None
    for i, ln in enumerate(lines):
        m = re.match(r'\s*\[\s*trigger\s*=\s*([^\]]+)\]', ln, re.I)
        if m:
            trig, entry = m.group(1).strip().lower(), None
            continue
        m = re.match(r'\s*<\s*(?:enemy|text)\s*=\s*([^>]+)>', ln, re.I)
        if m:
            entry = m.group(1).strip().lower()
            continue
        m = re.match(r'([ \t]*)pos([ \t]*=[ \t]*)(-?\d+)(.*)$', ln, re.I)
        if not m:
            continue
        rule = TRIGGER_RULES.get((trig, entry), TRIGGER_RULES.get((trig, None)))
        x = int(m.group(3))
        if rule is None or x == -1:       # -1 = engine centres it itself; never touch
            continue
        lines[i] = f"{m.group(1)}pos{m.group(2)}{_shift(x, rule, width)}{m.group(4)}"
    return "\n".join(lines)


def fix_particle_fields(text: str, width: int):
    """Move each `[FIELD]`'s particle spawn `rect` into the wider screen.

    A FIELD emits ambient particles (starfield, rain, snow) from one or more spawn
    rectangles, given in screen space. Two patterns exist across the 70 level files, and
    both are broken at 16:9 -- this is the "particles spawn nowhere near the right side"
    report:

      x1 >= 800                         a narrow strip at/beyond the old RIGHT edge, which
      rect = 800, 64, 820, 600   (x49)  stars/rain drift left out of. At 1067 that strip
      rect = 850,  0, 860, 600   (x35)  sits at ~75% across, so particles visibly pop into
                                        mid-air. The shop levels use 850 (a 50px margin)
                                        rather than 800 — same pattern, so key off `>= 800`
                                        and preserve each one's margin.
                                        -> shift x by (Width-800): 1067..1087 / 1117..1127

      x1 == 0 and x2 == 800             a FULL-WIDTH strip just above the screen; rain
      rect = 0, -20, 800, -10    (x12)  falls from it. At 1067 it only covers 0..800, so
                                        the right 267px would get no rain at all.
                                        -> stretch to the new width: 0, -20, 1067, -10

    A FIELD can carry both (level_grass's rain enters from the top *and* the right, since
    the level also scrolls). Anything matching neither pattern is left alone and reported;
    `level_zog`'s `rect = 650, 0, 660, 520` is the only one, and 650 is mid-screen even at
    800, so there is no defensible way to reinterpret it.
    """
    extra = width - BASE_W
    if extra == 0:
        return text, []
    lines, in_field, skipped = text.split("\n"), False, []
    for i, ln in enumerate(lines):
        if re.match(r'\s*\[', ln):
            in_field = bool(re.match(r'\s*\[\s*FIELD', ln, re.I))
            continue
        m = re.match(r'([ \t]*)rect([ \t]*=[ \t]*)(-?\d+)[ \t]*,[ \t]*(-?\d+)[ \t]*,'
                     r'[ \t]*(-?\d+)[ \t]*,[ \t]*(-?\d+)(.*)$', ln, re.I)
        if not (in_field and m):
            continue
        x1, y1, x2, y2 = (int(m.group(g)) for g in (3, 4, 5, 6))
        if x1 >= BASE_W:                       # spawn strip at/beyond the old right edge
            x1, x2 = x1 + extra, x2 + extra    # (keeps each strip's own margin)
        elif x1 == 0 and x2 == BASE_W:         # full-width strip above/below the screen
            x2 = width
        else:
            skipped.append(f"rect = {x1}, {y1}, {x2}, {y2}")
            continue
        lines[i] = f"{m.group(1)}rect{m.group(2)}{x1}, {y1}, {x2}, {y2}{m.group(7)}"
    return "\n".join(lines), skipped


def build(assets_dir: str, width: int) -> dict:
    """-> {archive_name: bytes} for the overlay."""
    files, skipped = {}, []
    for rel, fn in ((PATH_DEF, recentre_absolute_paths),
                    (BEHAVIOR_DEF, rescale_cross_behaviors),
                    (TRIGGER_DEF, fix_triggers)):
        text = open(os.path.join(assets_dir, rel), errors="replace").read()
        files[rel] = fn(text, width).encode("latin1")

    # Particle fields are game-wide: every level file that declares one.
    for path in sorted(glob.glob(os.path.join(assets_dir, "DATA/level/*/lvl*.txt"))):
        text = open(path, errors="replace").read()
        if "[FIELD" not in text.upper():
            continue
        new, skip = fix_particle_fields(text, width)
        rel = os.path.relpath(path, assets_dir).replace(os.sep, "/")
        skipped += [f"{rel}: {s}" for s in skip]
        if new != text:
            files[rel] = new.encode("latin1")
    build.skipped = skipped
    return files


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    from jngdat import pack

    width = config.LOGICAL[0]
    files = build(config.ASSETS, width)
    if "--print" in sys.argv:
        for name, data in files.items():
            print(f"=== {name}  (Width={width}) ===\n{data.decode('latin1')}")
        return
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.BUILD_DIR, "ws.dat")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = pack(files, out)
    print(f"ws.dat: {len(files)} intro defs re-authored for Width={width} -> {out} ({n} bytes)")
    for name in files:
        print(f"  {name}")


if __name__ == "__main__":
    main()
