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

GAME-WIDE SWEEP (all 70 levels): beyond the intro and the particle fields, every
level's spawn tables carry the same three 800-isms.  `tools/scan_levels.py` simulates
every starter -> trigger -> enemy -> behavior chain at 800 and at the target Width
(see its docstring for the engine semantics) and this module turns its findings into
line edits, re-scanning after each round until the sweep comes back clean:

  LPOP    left-entry spawn offsets (pos <= -736 relative to the right edge): the
          enemy spawned off the LEFT edge at 800 but pops in on-screen at 1067; the
          same fix restores <on_screen_left> activations that only fire when the
          spawn x is < 0.                              -> pos.x -= (Width-800)
  RSTAGE  `absolute = 1` staging at/past the old right edge (x in 780..1100), now
          visibly on-screen.  Screen-absolute regardless of caller (verified in
          CTrigger::Execute), so the shift is always safe.  -> pos.x += (Width-800)
  VANISH  the final path segment's reach was tuned to cross an 800-wide screen; the
          enemy is removed mid-air when the node list runs out (CEnemy::NextNodePos).
          -> raise that segment's path_scale.x so the death keeps its authored
          edge margin: leftward movers die at the same x as at 800, rightward movers
          at the same distance past the (moved) right edge.  If the segment inherited
          its scale from an earlier event, a new path_scale line is inserted BEFORE
          the path line (scale must be current when SetPath computes the first node).
  LIMIT   path_limit right-bounds in 780..1100 (kill/turn boxes that extended past
          the old right edge)                          -> x += (Width-800)

Fixes are computed per FILE (behaviors/triggers are shared across levels), applied to
whole re-authored copies shipped in the overlay.  Chains the simulator cannot model
(hero-hunting, rotated or cyclically reassigned paths) are marked fuzzy: they are
still fixed when the final segment is a plain straight exit, and reported either way.

Output: {archive_path: bytes} to merge into an overlay listed before jng.dat
(first-match-wins), so the originals are never modified.
"""
import re, os, sys, glob, math

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


# ---------------------------------------------------------------------------
# game-wide sweep (driven by tools/scan_levels.py)
# ---------------------------------------------------------------------------
POS_X_RE = re.compile(r'(pos\s*=\s*)(-?\d+)', re.I)
SCALE_X_RE = re.compile(r'(path_scale\s*=\s*)(-?\d+)', re.I)
LIMIT_RE = re.compile(r'(path_limit\s*=\s*)(.+)$', re.I)
TIMER_RE = re.compile(r'(<\s*on_timer\s*=\s*)(\d+)', re.I)
STARTER_RE = re.compile(r'(t\.\S+\s*=\s*)(-?\d+)', re.I)


def _apply_edits(text: str, edits) -> str:
    """edits: {line_no: ("replace", fn) | ("insert_before", str)}, 0-based lines."""
    lines = text.split("\n")
    for ln in sorted(edits, reverse=True):
        op, arg = edits[ln]
        if op == "replace":
            lines[ln] = arg(lines[ln])
        else:
            lines.insert(ln, arg)
    return "\n".join(lines)


def _shift_pos_x(delta):
    """Shift a pos line's x -- and, for 4-value random boxes (x1,y1,x2,y2), both xs."""
    def sub(ln):
        m = POS_X_RE.search(ln)
        head, vals = ln[:m.start(2)], ln[m.start(2):].split(",")
        vals[0] = str(int(vals[0].strip()) + delta)
        if len(vals) >= 4:
            try:
                vals[2] = f" {int(vals[2].strip()) + delta}"
            except ValueError:
                pass
        return head + ",".join(vals)
    return sub


def _set_scale(new_x, new_y=None):
    """Rewrite path_scale's x (and, for angle-preserving extension, y)."""
    def sub(ln):
        m = SCALE_X_RE.search(ln)
        head, vals = ln[:m.start(2)], ln[m.start(2):].split(",")
        vals[0] = str(new_x)
        if new_y is not None and len(vals) >= 2:
            vals[1] = f" {new_y}"
        return head + ",".join(vals)
    return sub


def _shift_limit(extra, lo, hi):
    def sub(ln):
        m = LIMIT_RE.search(ln)
        vals = [v.strip() for v in m.group(2).split(",")]
        out = []
        for v in vals:
            try:
                n = int(v)
                out.append(str(n + extra) if lo <= n <= hi else v)
            except ValueError:
                out.append(v)
        return ln[:m.start(2)] + ", ".join(out)
    return sub


# Hand-curated on_timer retimes the simulator cannot derive (rotated/sinus paths it
# can only track fuzzily).  Each entry is safe under BOTH hypotheses about the timer:
# if it fires after the exit it extends an invisible cleanup, if it IS the despawn it
# restores the crossing.  (t_new = t + (Width-800) / cruise speed.)
#   ussi: the huge mothership flyby in level_ussi -- sinus path at cruise ~60 px/s,
#   10.75 s was tuned to an 800-wide pass; 267 extra px need ~4.45 s more.
MANUAL_RETIMES = [
    ("data/enemy/special.ussi/ussi_b.txt", "ussi", 10750,
     lambda extra: 10750 + int(math.ceil(extra / 60.0 * 100) * 10)),
]


def _sort_starters(text: str) -> str:
    """Keep a starter file sorted by x -- LoadStarter REJECTS the whole level if any
    starter's x is below its predecessor's, and shifting a subset of starters by
    +(Width-800) can make one overtake a later unshifted line.  Only the `t.` lines
    are reordered (stable, so equal-x lines keep their authored order); comments and
    blanks stay where they are."""
    lines = text.split("\n")
    idxs, entries = [], []
    for i, ln in enumerate(lines):
        m = re.match(r'\s*t\.\S+\s*=\s*(-?\d+)', ln)
        if m:
            idxs.append(i)
            entries.append((int(m.group(1)), ln))
    entries.sort(key=lambda p: p[0])
    for i, (_x, ln) in zip(idxs, entries):
        lines[i] = ln
    return "\n".join(lines)


def sweep_levels(assets_dir: str, width: int, base_files: dict,
                 rounds: int = 6) -> tuple:
    """Generic game-wide pass.  -> ({archive_name: bytes}, report dict).

    base_files: the intro/FIELD overlay from build(), already re-authored; the scan
    runs on top of it so those fixes are neither re-flagged nor re-applied."""
    import scan_levels as SL
    extra = width - SL.BASE_W
    fixed = {}                    # lower rel -> text (this sweep's edits)
    display = {}                  # lower rel -> archive_name (on-disk case)
    done = set()                  # finding identities already fixed in a prior round
    base_lower = {k.replace("\\", "/").lower(): v.decode("latin1")
                  for k, v in base_files.items()}
    report = dict(applied={}, timer_risk=[], unfixed=[], fuzzy_fixed=[])

    if extra == 0:
        return {}, report

    for _round in range(rounds):
        gd = SL.GameData(assets_dir, overlay={**base_lower, **fixed})
        findings, _seen = SL.scan(gd, width)
        edits = {}                # rel -> {line: op}, staged for this round

        def stage(rel, line, op, tag):
            if rel.startswith("data/level/intro_1/") \
                    and not rel.endswith("starter.txt"):
                return            # the intro defs are hand-authored above;
                                  # starter timing is not covered by them
            f_edits = edits.setdefault(rel, {})
            if line in f_edits:
                return            # first writer wins; identical for shared chains
            f_edits[line] = op
            report["applied"].setdefault(tag, set()).add(rel)

        leftovers = []
        for f in findings:
            cls = f["cls"]
            # the intro's trigger/behavior/path defs are hand-authored by the
            # rules above; the scan may re-flag values those rules moved (e.g.
            # the right-anchored credits panel).  Starter timing (SSTART) is not
            # covered by them and stays in scope.
            if cls != "SSTART" and ("intro_1" in (f.get("src") or "")
                                    or "intro_1" in (f.get("bsrc") or "")):
                continue
            ident = (cls, f["src"], f.get("trigger"), f.get("entry"),
                     f.get("bsrc"), f.get("behavior"), f.get("prop_line"))
            if ident in done and cls != "VANISH":
                # a shifted pos can land back in the detection band (e.g. -800 ->
                # -1067 spawns exactly at x = 0); one shift is always the full fix,
                # so a repeat flag on an edited entry is self-inflicted.  VANISH is
                # exempt: its fix may legitimately need refining once the round's
                # spawn shifts have landed.
                continue
            if cls in ("LPOP", "RSTAGE") and f.get("prop_line") is not None:
                if cls == "LPOP" and not f.get("starter"):
                    # relative pos on a trigger nothing starter-drives (weapon/child
                    # spawns are parent-relative): not ours to move
                    leftovers.append(f)
                    continue
                done.add(ident)
                stage(f["src"], f["prop_line"],
                      ("replace", _shift_pos_x(-extra if cls == "LPOP" else extra)),
                      cls)
            elif cls == "SSTART":
                done.add(ident)
                stage(f["src"], f["prop_line"],
                      ("replace", lambda ln, d=extra:
                       STARTER_RE.sub(lambda m: f"{m.group(1)}{int(m.group(2)) + d}",
                                      ln, count=1)), cls)
            elif cls == "LIMIT":
                done.add(ident)
                stage(f["src"], f["prop_line"],
                      ("replace", _shift_limit(extra, SL.SUS_LO, SL.SUS_HI)), cls)
            elif cls == "VANISH":
                if _stage_vanish_fix(f, extra, stage, report):
                    done.add(ident)
                else:
                    leftovers.append(f)
            elif cls == "CONFLICT":
                leftovers.append(f)

        if os.environ.get("WS_SWEEP_DEBUG"):
            from collections import Counter
            print(f"round {_round}: "
                  f"{Counter(f['cls'] for f in findings).most_common()} "
                  f"-> edits in {len(edits)} files", file=sys.stderr)
        if not edits:
            report["unfixed"] = leftovers
            break
        for rel, f_edits in edits.items():
            cur = fixed.get(rel) or base_lower.get(rel) or gd.text(rel)
            new = _apply_edits(cur, f_edits)
            if rel.endswith("starter.txt"):
                new = _sort_starters(new)
            fixed[rel] = new
            real = gd._real.get(rel)
            display[rel] = (os.path.relpath(real, assets_dir).replace(os.sep, "/")
                            if real else rel)
    else:
        raise RuntimeError("level sweep did not converge")

    # hand-curated retimes (see MANUAL_RETIMES)
    gd = SL.GameData(assets_dir, overlay={**base_lower, **fixed})
    for rel, bname, t_old, t_fn in MANUAL_RETIMES:
        for d in gd.defs(rel) or []:
            if d.kind != "behavior" or d.name != bname.lower():
                continue
            for e in d.entries:
                if e.kind == "on_timer" and e.name == str(t_old):
                    cur = fixed.get(rel) or gd.text(rel)
                    t_new = t_fn(extra)
                    fixed[rel] = _apply_edits(cur, {e.line: (
                        "replace", lambda ln, t=t_new:
                        TIMER_RE.sub(lambda m: f"{m.group(1)}{t}", ln, count=1))})
                    real = gd._real.get(rel)
                    display[rel] = (os.path.relpath(real, assets_dir)
                                    .replace(os.sep, "/") if real else rel)
                    report["applied"].setdefault("TIMEKILL", set()).add(rel)

    out = {display[rel]: text.encode("latin1") for rel, text in fixed.items()}
    return out, report


def _stage_vanish_fix(f, extra, stage, report) -> bool:
    """Turn one VANISH finding into a path_scale or on_timer edit.
    False if not fixable."""
    ot, ob = f["outW"], f["out800"]

    if f.get("child") and ot.death_kind != "path-exhausted":
        # child chains (spawned by a parent) only get reach extensions; their
        # timers may pace scripted sequences and are never retimed
        return False

    if ot.death_kind == "timekill":
        # a timer-despawn tuned to the 800 crossing: give it the extra travel time
        ev, vel = ot.death_event, abs(ot.death_vel)
        if ev is None or not vel:
            return False
        if f["direction"] == "left":
            delta_x = ot.death_x - ob.death_x
        else:
            delta_x = (ob.death_x + extra) - ot.death_x
        if delta_x <= 0:
            return False
        t_new = int(math.ceil((float(ev.name) + delta_x * 1000.0 / vel) / 10) * 10)
        stage(ot.death_bsrc, ev.line,
              ("replace", lambda ln, t=t_new:
               TIMER_RE.sub(lambda m: f"{m.group(1)}{t}", ln, count=1)),
              "TIMEKILL")
        if f.get("fuzzy"):
            report["fuzzy_fixed"].append(f)
        return True

    if not ot.segments:
        return False
    seg = ot.segments[-1]
    units, sx = seg.path_units_x, seg.sx
    if f.get("child") and ot.vx > 0 and f.get("t800") is not None:
        # if living longer would newly reach one of the chain's own timers, the
        # extension would change a scripted visual (e.g. the endgame explosion
        # hiding on its part_state timer instead of its path end): leave it
        d = (ot.death_x - ob.death_x) if f["direction"] == "left" \
            else (ob.death_x + extra) - ot.death_x
        if any(f["t800"] < tt <= f["t800"] + d / ot.vx + 0.01
               for tt in ot.timeline_t):
            return False
    if not units or not sx:
        return False
    if f["direction"] == "left":
        delta_reach = ot.death_x - ob.death_x              # extend leftward by this
    else:
        delta_reach = (ob.death_x + extra) - ot.death_x    # extend rightward
    if delta_reach <= 0:
        return False
    scale_val = sx * 100.0
    new_mag = int(math.ceil(abs(scale_val) + delta_reach * 100.0 / abs(units)))
    new_x = new_mag if scale_val > 0 else -new_mag
    # extend diagonals ALONG their line: scale y by the same factor so the exit
    # angle is preserved (x-only extension would flatten the trajectory)
    sy_val = seg.sy * 100.0
    new_y = (int(round(sy_val * new_mag / abs(scale_val)))
             if sy_val and abs(scale_val) > 0 else None)

    rel = seg.bsrc or f["bsrc"]      # handoffs can end in a different file
    if seg.scale_prop is not None and seg.scale_event is seg.event:
        stage(rel, seg.scale_prop.line, ("replace", _set_scale(new_x, new_y)),
              "VANISH")
    else:
        stage(rel, seg.path_prop.line,
              ("insert_before",
               f"    path_scale = {new_x}, {new_y if new_y is not None else int(round(sy_val))}"),
              "VANISH")
    if f.get("fuzzy"):
        report["fuzzy_fixed"].append(f)
    # would the longer life let a timer fire that never fired at 800?
    if ot.vx > 0 and f.get("t800") is not None:
        t_new = f["t800"] + delta_reach / ot.vx
        risky = [tt for tt in ot.timeline_t if f["t800"] < tt <= t_new + 0.01]
        if risky:
            report["timer_risk"].append((f, risky))
    return True


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
    swept, report = sweep_levels(config.ASSETS, width, files)
    overlap = set(files) & set(swept)
    assert not overlap, f"sweep touched hand-authored files: {overlap}"
    files.update(swept)

    if "--print" in sys.argv:
        for name, data in files.items():
            print(f"=== {name}  (Width={width}) ===\n{data.decode('latin1')}")
        return
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.BUILD_DIR, "ws.dat")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = pack(files, out)
    print(f"ws.dat: {len(files)} defs re-authored for Width={width} -> {out} ({n} bytes)")
    for tag, rels in sorted(report["applied"].items()):
        print(f"  sweep {tag}: {len(rels)} files")
    if report["fuzzy_fixed"]:
        print(f"  sweep: {len(report['fuzzy_fixed'])} fuzzy chains fixed on their "
              f"final straight segment (listed by scan_levels.py)")
    for f, risky in report["timer_risk"]:
        print(f"  TIMER-RISK {f['behavior']} [{f['bsrc']}]: events at {risky} s "
              f"newly reachable")
    for f in report["unfixed"]:
        print(f"  UNFIXED {f['cls']} {f.get('trigger')}/{f.get('behavior')} "
              f"[{f.get('bsrc') or f.get('src')}]")


if __name__ == "__main__":
    main()
