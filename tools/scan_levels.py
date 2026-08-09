#!/usr/bin/env python3
"""Find (and let make_widescreen_defs.py fix) every 4:3 screen-space assumption in the
game's level data.

For each level this walks lvl.txt's includes (trigger_file / behavior_file / path_file /
enemy_set / starter_file), then simulates every starter -> trigger -> enemy -> behavior
chain twice -- once on the 800-wide screen the data was authored against, once on the
target Width -- and reports every chain whose *qualitative* outcome differs:

  LPOP     spawns off the LEFT edge at 800 (pos <= ~-800 relative to the right edge)
           but visibly on-screen at the target width; also covers the on_screen_left
           activation stall: those behaviors fire at spawn only when spawn x < 0
  RSTAGE   absolute staging at/just beyond the old right edge (x >= 800) that the wider
           screen now shows: `absolute = 1` trigger entries and absolute path nodes
  VANISH   the path runs out (CEnemy::NextNodePos clears the alive flag when the node
           list ends) while still on-screen at the target width, off-screen at 800
  LIMIT    a path_limit carrying an absolute screen x in the suspicious 780..1100 band
  CONFLICT a trigger entry that would get an LPOP edit but is ALSO spawned from a
           behavior (for relative pos, trigger.X = dx, dy makes the offset
           parent-relative, not right-edge-relative), so the edit cannot be applied
           blindly.  `absolute = 1` entries are exempt: the engine uses their pos as
           screen coordinates regardless of the caller (CTrigger::Execute 0x8083e13
           skips the base add), and `absolute = 2` (world coords, scroll-adjusted)
           is never touched.

Engine semantics baked in here (from the disassembly, see docs/HOW_IT_WORKS.md):
  * ParseNextStarter fires when scrollX + Width >= starter.x and spawns at x = Width
    (+ the trigger entry's pos offset); `absolute = 1` entries use pos as screen coords.
  * CEnemy::SetDestinationPointByNode: relative nodes (node/direct) are deltas from the
    sprite's CURRENT position, scaled per-axis by path_scale/100; absolute[_direct]
    nodes jump to screen coords; hero_hunt* track the player; a `loop` node (type 9)
    restarts the path.
  * When the node list is exhausted the enemy is silently removed -- ON-SCREEN if the
    scaled reach ends there.  This is the mid-air-vanish class.
  * speed = vx, vy in px/s per axis, -1 = keep current.
  * `ground >= 1` anchors the sprite to the world: it drifts left with the level's
    Scroll_speed on top of any path motion.
  * `behavior = X` inside an event hands the enemy off to behavior X (timers restart).
  * `part_state.0 = 0` in an <on_timer> hides the main part -- the designers' idiom
    for a TIME-based despawn, tuned to how long an 800-wide crossing takes (e.g. the
    level-1 habitats: 22 s at scroll 40 ~ spawn at 900, gone at x=20).  On a wider
    screen the same timer fires a third of the way across.  VANISH covers these too;
    the fix is retiming that on_timer (see make_widescreen_defs.py).

The 800-wide reference simulation always runs on PRISTINE data while the target-width
simulation runs on the overlay under repair, so "fixed" means "matches the authored
800 margins" and the fix loop terminates exactly there.

The simulator is deliberately conservative: anything player-relative (hero_hunt),
rotated (path_angle/path_rotation) or cyclic makes the position an interval / marks the
chain fuzzy, and fuzzy chains are reported for eyes-on review even when auto-fixed.
"""
import os, re, sys, glob, argparse
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE_W = 800
SPRITE_W = 64                 # nominal sprite width for death-visibility margins
HALF_W = 32                   # half of it, for spawn visibility (pos is the centre)
SUS_LO, SUS_HI = 780, 1100    # "off-screen-right staging at 800" band
LEFT_ANCHOR = -736            # pos <= this (spawn x <= 64 at 800) counts as a left entry
CHILD_STAGE = 200             # worst-case parent staging offset for child chains
                              # (the shop target spawner sits at Width + 200)

# ---------------------------------------------------------------------------
# generic def-file parsing
# ---------------------------------------------------------------------------
SEC_RE = re.compile(r'^\s*\[\s*([a-z_0-9]+)\s*=\s*([^\]]+?)\s*\]', re.I)
ENT_RE = re.compile(r'^\s*<\s*([a-z_0-9.]+)\s*=\s*([^>]+?)\s*>', re.I)
KEY_RE = re.compile(r'^\s*([a-z_0-9.]+)\s*=\s*(.*?)\s*$', re.I)

Prop = namedtuple("Prop", "key vals line")               # line = 0-based index
Entry = namedtuple("Entry", "kind name props line")      # <kind = name>
Def = namedtuple("Def", "kind name props entries line")  # [kind = name]


def parse_defs(text):
    """-> list of Def.  props preserve order, duplicates and source line numbers."""
    defs, cur, ent = [], None, None
    for i, raw in enumerate(text.split("\n")):
        line = raw.split("//")[0]
        m = SEC_RE.match(line)
        if m:
            cur = Def(m.group(1).lower(), m.group(2).strip().lower(), [], [], i)
            ent = None
            defs.append(cur)
            continue
        if cur is None:
            continue
        m = ENT_RE.match(line)
        if m:
            ent = Entry(m.group(1).lower(), m.group(2).strip(), [], i)
            cur.entries.append(ent)
            continue
        m = KEY_RE.match(line)
        if m and not line.lstrip().startswith("#"):
            key = m.group(1).lower()
            vals = [v.strip() for v in m.group(2).split(",")]
            (ent.props if ent is not None else cur.props).append(Prop(key, vals, i))
    return defs


def nums(vals):
    out = []
    for v in vals:
        try:
            out.append(float(v))
        except ValueError:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# game data loading (with an in-memory overlay for verifying generated fixes)
# ---------------------------------------------------------------------------
class GameData:
    def __init__(self, assets, overlay=None):
        self.assets = assets
        self.overlay = {k.lower(): v for k, v in (overlay or {}).items()}
        self._real = {}          # lowercase rel path -> real path
        for root, _dirs, files in os.walk(assets):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, assets).replace(os.sep, "/")
                self._real[rel.lower()] = p
        self._cache = {}

    def canon(self, ref):
        """-> lowercase forward-slash rel path, or None if the file doesn't exist."""
        rel = ref.replace("\\", "/").lower().lstrip("/")
        return rel if (rel in self._real or rel in self.overlay) else None

    def text(self, ref):
        rel = self.canon(ref)
        if rel is None:
            return None
        if rel in self.overlay:
            return self.overlay[rel]
        return open(self._real[rel], errors="replace").read()

    def defs(self, ref):
        rel = self.canon(ref)
        if rel is None:
            return None
        key = (rel, rel in self.overlay and hash(self.overlay[rel]))
        if key not in self._cache:
            self._cache[key] = parse_defs(self.text(rel))
        return self._cache[key]


INCLUDE_RE = re.compile(
    r'^\s*(path_file|trigger_file|behavior_file|enemy_set|starter_file)\s*=\s*(\S+)',
    re.I | re.M)


class Level:
    """One level's resolved namespace: triggers, behaviors, paths, starters."""
    def __init__(self, gd, lvl_path):
        self.name = os.path.basename(os.path.dirname(lvl_path))
        self.triggers, self.behaviors, self.paths = {}, {}, {}
        self.trigger_src, self.behavior_src, self.path_src = {}, {}, {}
        self.starters = []
        self.missing = []
        text = open(lvl_path, errors="replace").read()
        m = re.search(r'^\s*Scroll_speed\s*=\s*(-?\d+)', text, re.I | re.M)
        self.scroll = float(m.group(1)) if m else 40.0
        for m in INCLUDE_RE.finditer(text):
            kind, val = m.group(1).lower(), m.group(2).strip()
            if kind == "enemy_set":
                refs = [(val + "_t.txt", "trigger"), (val + "_b.txt", "behavior")]
            elif kind == "starter_file":
                self._load_starters(gd, val)
                continue
            else:
                refs = [(val, {"path_file": "path", "trigger_file": "trigger",
                               "behavior_file": "behavior"}[kind])]
            for ref, _want in refs:
                ds = gd.defs(ref)
                if ds is None:
                    self.missing.append(ref)
                    continue
                rel = gd.canon(ref)
                for d in ds:
                    if d.kind == "trigger":
                        self.triggers[d.name] = d
                        self.trigger_src[d.name] = rel
                    elif d.kind == "behavior":
                        self.behaviors[d.name] = d
                        self.behavior_src[d.name] = rel
                    elif d.kind == "path":
                        self.paths[d.name] = d
                        self.path_src[d.name] = rel

    def _load_starters(self, gd, ref):
        text = gd.text(ref)
        if text is None:
            self.missing.append(ref)
            return
        self.starter_src = gd.canon(ref)
        for i, line in enumerate(text.split("\n")):
            m = re.match(r'\s*t\.(\S+?)\s*=\s*(-?\d+)\s*,\s*(-?\d+)', line)
            if m:
                self.starters.append((m.group(1).lower(),
                                      int(m.group(2)), int(m.group(3)), i))


# ---------------------------------------------------------------------------
# kinematic simulation
# ---------------------------------------------------------------------------
REL_NODES = {"node", "direct"}
ABS_NODES = {"absolute", "absolute_direct"}
HERO_NODES = {"hero_hunt", "hero_hunt_x", "hero_hunt_y", "hero_pos"}

Outcome = namedtuple(
    "Outcome",
    "spawn_x death death_x death_t reason fuzzy notes osl_at_spawn segments "
    "timeline_t vx death_kind death_event death_bsrc death_vel rand_speeds")

Segment = namedtuple(
    "Segment",
    # the state that produced the final (or any) path assignment; bsrc is the file
    # of the behavior whose event assigned it (handoffs can cross files)
    "path_name path_units_x sx sy scale_prop scale_event path_prop event start_t "
    "bsrc")

MAX_HANDOFFS = 8


class Sim:
    """Simulate one behavior chain from a spawn point at screen width W.

    x is an interval [lo, hi] so hero-relative motion degrades gracefully.  The
    interval is tracked in the WORLD frame; ground-anchored sprites additionally
    drift left with the level scroll, applied when positions are read out."""

    def __init__(self, level, width):
        self.lv, self.W = level, width

    def run(self, behavior_name, spawn_x, t_limit=180.0, rand_speed=None):
        """rand_speed: simulate the specific on_random speed branch (enemies pick a
        random cruise speed at spawn -- each branch can die a different way: slow
        ones hit timer-despawns, fast ones exhaust their path).  None = the slowest
        branch (pessimistic for timers); scan() re-runs per branch."""
        st = dict(xlo=float(spawn_x), xhi=float(spawn_x), t=0.0, vx=100.0,
                  sx=1.0, sy=0.0, scale_prop=None, scale_event=None,
                  nodes=[], ni=0, ground=False, fuzzy=False, notes=[],
                  segments=[], times=[], death=None, rand_speed=rand_speed,
                  rand_speeds=set())
        first = self.lv.behaviors.get(behavior_name)
        osl = (first is not None and spawn_x < 0
               and any(e.kind == "on_screen_left" for e in first.entries))

        beh, visited = behavior_name, []
        while beh is not None and len(visited) < MAX_HANDOFFS:
            if beh in visited:
                st["notes"].append(f"behavior cycle at {beh}")
                break
            visited.append(beh)
            beh = self._run_one(beh, st, t_limit)

        d = st["death"]
        drift = self.lv.scroll if st["ground"] else 0.0

        if d is not None:
            kind, x, t, ev, bsrc, path_speed = d
            # signed screen-frame velocity at death: path motion in the direction
            # of the final segment, minus the leftward world scroll for ground units
            seg = st["segments"][-1] if st["segments"] else None
            dirn = 0.0
            if seg and seg.path_units_x * seg.sx:
                dirn = 1.0 if seg.path_units_x * seg.sx > 0 else -1.0
            vel = dirn * path_speed - drift
            return Outcome(spawn_x, True, x - drift * t, t, kind, st["fuzzy"],
                           st["notes"], osl, st["segments"], st["times"], st["vx"],
                           kind, ev, bsrc, vel, st["rand_speeds"])
        return Outcome(spawn_x, False, None, None, "alive", st["fuzzy"],
                       st["notes"], osl, st["segments"], st["times"], st["vx"],
                       None, None, None, 0.0, st["rand_speeds"])

    def _run_one(self, bname, st, t_limit):
        """Run one behavior of the chain.  Returns the next behavior or None."""
        b = self.lv.behaviors.get(bname)
        notes = st["notes"]
        if b is None:
            notes.append(f"behavior {bname} missing")
            st["fuzzy"] = True
            return None
        bsrc = self.lv.behavior_src.get(bname)
        st["cur_bsrc"] = bsrc
        t0 = st["t"]

        timeline = [(t0, e) for e in b.entries if e.kind == "on_init"]
        rand_speeds, rand_handoffs = [], []
        for e in b.entries:
            if e.kind == "on_timer":
                try:
                    timeline.append((t0 + float(e.name) / 1000.0, e))
                except ValueError:
                    pass
                continue
            if e.kind == "on_random":
                for p in e.props:
                    n = nums(p.vals)
                    if p.key == "speed" and n and n[0] is not None and n[0] >= 0:
                        rand_speeds.append(n[0])
                    elif p.key == "behavior":
                        rand_handoffs.append(p.vals[0].lower())
            if e.kind in ("on_node", "on_cycle", "on_random", "on_global",
                          "if_global", "on_total_timer", "on_screen_left",
                          "on_screen_right", "on_limit_left", "on_limit_right",
                          "on_hit", "on_anim_end"):
                if any(p.key == "path" for p in e.props):
                    st["fuzzy"] = True
                    notes.append(f"path set in <{e.kind}> (not simulated)")
        timeline.sort(key=lambda p: p[0])
        st["times"] += [w for w, _ in timeline]
        if rand_speeds:
            st["rand_speeds"].update(rand_speeds)
            if st["rand_speed"] is not None and st["rand_speed"] in rand_speeds:
                st["vx"] = st["rand_speed"]           # simulate this exact branch
            else:
                # default: the slowest branch (pessimistic for timer-despawns)
                st["vx"] = min([st["vx"]] + rand_speeds)

        for when, e in timeline:
            if self._advance_to(st, when):
                return None
            for p in e.props:
                n = nums(p.vals)
                if p.key == "path":
                    self._set_path(p, e, st, notes)
                elif p.key == "path_scale" and n and n[0] is not None:
                    st["sx"] = n[0] / 100.0
                    if len(n) > 1 and n[1] is not None:
                        st["sy"] = n[1] / 100.0
                    st["scale_prop"], st["scale_event"] = p, e
                elif p.key == "speed" and n and n[0] is not None and n[0] >= 0:
                    st["vx"] = n[0]
                elif p.key in ("path_angle", "path_rotation") and any(
                        x not in (0, None) for x in n):
                    st["fuzzy"] = True
                    if f"{p.key} used" not in notes:
                        notes.append(f"{p.key} used")
                elif p.key == "ground" and n and n[0] not in (0, None):
                    st["ground"] = True
                elif p.key == "part_state.0" and n and n[0] == 0 and when > t0:
                    # the timer-despawn idiom: hide the main part
                    x = (st["xlo"] + st["xhi"]) / 2
                    spd = st["vx"] if st["ni"] < len(st["nodes"]) else 0.0
                    st["death"] = ("timekill", x, st["t"], e, bsrc, spd)
                    return None
                elif p.key == "behavior":
                    return p.vals[0].lower()          # handoff; timers restart

        # deterministic-enough random handoff: every branch goes the same place
        if rand_handoffs and len(set(rand_handoffs)) == 1:
            st["t"] += 0.3
            return rand_handoffs[0]
        if rand_handoffs:
            st["fuzzy"] = True
            notes.append(f"diverging on_random handoffs in {bname}")
            return None

        # no more events: run the current path to its end
        self._advance_to(st, st["t"] + t_limit)
        return None

    def _set_path(self, p, e, st, notes):
        pname = p.vals[0].lower()
        pdef = self.lv.paths.get(pname)
        if pdef is None:
            notes.append(f"path {pname} missing")
            st["fuzzy"] = True
            st["nodes"], st["ni"] = [], 0
        else:
            st["nodes"], st["ni"] = self._path_nodes(pdef, notes), 0
            st["segments"].append(Segment(
                pname, sum(nx for k, nx, _ in st["nodes"] if k == "rel"),
                st["sx"], st["sy"], st["scale_prop"], st["scale_event"],
                p, e, st["t"], st.get("cur_bsrc")))

    def _advance_to(self, st, t_stop):
        """Follow the current path until t_stop.  True if the path ran out."""
        xlo, xhi, ni, t, died = self._advance(
            st["nodes"], st["ni"], st["xlo"], st["xhi"], st["vx"], st["sx"],
            st["t"], t_stop, st["notes"])
        st["xlo"], st["xhi"], st["ni"], st["t"] = xlo, xhi, ni, t
        if died:
            seg = st["segments"][-1] if st["segments"] else None
            # direction/fix data are derived from the final segment by the caller
            st["death"] = ("path-exhausted", (xlo + xhi) / 2, t,
                           seg.event if seg else None, None, 0.0)
        return died

    def _path_nodes(self, pdef, notes):
        out = []
        for p in pdef.props:
            n = nums(p.vals)
            if p.key in REL_NODES and n and n[0] is not None:
                out.append(("rel", n[0], n[1] if len(n) > 1 else 0))
            elif p.key in ABS_NODES and n and n[0] is not None:
                out.append(("abs", n[0], n[1] if len(n) > 1 else 0))
            elif p.key in HERO_NODES:
                out.append(("hero", 0, 0))
            elif p.key == "loop":
                out.append(("loop", 0, 0))
        return out

    def _advance(self, nodes, ni, xlo, xhi, vx, sx, t, t_stop, notes):
        """Follow nodes until t_stop.  ->  xlo, xhi, ni, t, died."""
        if not nodes:
            return xlo, xhi, ni, t_stop, False
        while t < t_stop:
            if ni >= len(nodes):
                return xlo, xhi, ni, t, True          # exhausted -> removed
            kind, nx, ny = nodes[ni]
            if kind == "loop":                        # looping path never exhausts
                return xlo, xhi, 0, t_stop, False
            if kind == "hero":                        # player-relative: anywhere
                return 0.0, float(self.W), ni + 1, t_stop, False
            if kind == "abs":
                dlo = dhi = float(nx)
            else:
                dlo, dhi = xlo + nx * sx, xhi + nx * sx
            dist = max(abs(dlo - xlo), abs(dhi - xhi), abs(ny))
            if vx <= 0:
                return xlo, xhi, ni, t_stop, False    # not moving; stuck/vertical
            dt = dist / vx if dist else 0.0
            if t + dt > t_stop and dt > 0:
                frac = (t_stop - t) / dt
                return (xlo + (dlo - xlo) * frac, xhi + (dhi - xhi) * frac,
                        ni, t_stop, False)
            xlo, xhi, t = dlo, dhi, t + dt
            ni += 1
        return xlo, xhi, ni, t, False


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------
def entry_pos(entry):
    """-> (behavior, pos_x, pos_prop, abs_mode) of an <enemy> trigger entry.
    abs_mode: 0 = relative to the spawn base, 1 = screen coords, 2 = world coords."""
    beh, px, pp, absolute = None, 0.0, None, 0
    for p in entry.props:
        if p.key == "behavior":
            beh = p.vals[0].lower()
        elif p.key == "pos":
            n = nums(p.vals)
            if n and n[0] is not None:
                px, pp = n[0], p
        elif p.key == "absolute":
            n = nums(p.vals)
            absolute = int(n[0]) if n and n[0] is not None else 0
    return beh, px, pp, absolute


def classify_vanish(ob, ot, target_w):
    """Compare deaths (path exhaustion or timer-despawn) edge-relatively.
    -> 'left'/'right'/None.

    Leftward movers die against the fixed left edge (x = 0), so the 800 death x is
    directly the authored margin.  Rightward movers die against the RIGHT edge, which
    moves with Width, so their margin is death_x - Width.  A chain is broken when its
    death is (near-)visible at the target width with a margin meaningfully worse than
    the authored one.  ob comes from the PRISTINE 800 sim, so a fixed chain compares
    equal and stops matching."""
    if not (ob.death and ot.death) or ob.death_x is None or ot.death_x is None:
        return None
    if abs(ob.death_x - ot.death_x) < 1.0:
        # same screen position at both widths: the resting point is pinned (absolute
        # nodes / absolute spawn) -- a composition choice, not a width-dependent drift
        return None
    if ot.death_kind == "timekill":
        leftward = ot.death_vel < 0
    else:
        seg = ot.segments[-1] if ot.segments else None
        if seg is None:
            return None
        reach = seg.path_units_x * seg.sx
        if reach == 0:
            return None
        leftward = reach < 0
    if leftward:       # the target edge (x = 0) does not move
        # ob.death_x <= SPRITE_W: only treat the death as a crossing cleanup when
        # the AUTHORED death hugged the left edge.  A death deep on-screen at 800
        # (balloonjets bursting, turnbacks the sim can't follow) is masked by
        # motion this x-only model does not track -- leave it alone.
        if ot.death_x > -SPRITE_W and ot.death_x > ob.death_x + 5 \
                and ob.death_x <= SPRITE_W:
            return "left"
        return None
    m800 = ob.death_x - BASE_W
    mW = ot.death_x - target_w
    if mW < m800 - 5 and ot.death_x < target_w + SPRITE_W:
        return "right"
    return None


def trigger_spawn_kind(level, trig):
    """'ground' | 'air' | 'control' -- how a starter-fired trigger manifests.

    ground: every enemy entry's behavior anchors to the world (ground >= 1 at
    init) or spawns world/screen-absolute -- its position is a place, not an
    entry point, so the starter x must NOT be shifted.
    control: no enemy entries at all (text, scroll-speed, movie bars) -- pure
    timing, shifting restores the vanilla scroll position.
    air: anything else -- flies in from the spawn point, shift to keep the
    right-edge entry."""
    if trig is None:
        return "control"
    kinds = set()
    for e in trig.entries:
        if e.kind != "enemy":
            continue
        beh, _px, _pp, absolute = entry_pos(e)
        if absolute:
            kinds.add("ground")
            continue
        b = level.behaviors.get(beh) if beh else None
        grounded = False
        if b is not None:
            for ev in b.entries:
                if ev.kind != "on_init":
                    continue
                for p in ev.props:
                    n = nums(p.vals)
                    if p.key == "ground" and n and n[0] not in (0, None):
                        grounded = True
        kinds.add("ground" if grounded else "air")
    if not kinds:
        return "control"
    return "ground" if kinds == {"ground"} else "air"


def behavior_called_triggers(level):
    """Names of triggers invoked from behaviors (trigger.X = dx, dy): parent-relative."""
    out = set()
    for b in level.behaviors.values():
        for e in b.entries:
            for p in e.props:
                if p.key.startswith("trigger."):
                    out.add(p.key.split(".", 1)[1].lower())
    return out


_PRISTINE = {}


def scan(gd, target_w, verbose=False):
    """-> findings (list of dicts), chains (dedup map). Pure analysis, no edits.

    The 800-wide reference sim runs on pristine (un-overlaid) data so that fixed
    chains compare equal to their authored margins and stop matching."""
    if gd.assets not in _PRISTINE:
        _PRISTINE[gd.assets] = GameData(gd.assets)
    gdp = _PRISTINE[gd.assets]
    findings, seen = [], {}
    site_seen = set()          # global (file, line) fix-site dedup across levels
    lvls = sorted(glob.glob(os.path.join(gd.assets, "DATA/level/*/lvl*.txt")))
    for lp in lvls:
        lv = Level(gd, lp)
        lvb = Level(gdp, lp)
        sim_b, sim_t = Sim(lvb, BASE_W), Sim(lv, target_w)
        beh_called = behavior_called_triggers(lv)
        starter_trigs = {name for name, _x, _y, _l in lv.starters}

        # ---- SSTART: starters in the [800, Width) band ------------------
        # ParseNextStarter spawns at starter.x - scrollX (verified: 0x808485b..
        # 0x808486a).  A starter with x in [800, Width) fired at the right edge
        # at 800 but fires at t=0 ON-SCREEN at the target width.  Air chains and
        # control triggers get x += extra (same scroll position as vanilla);
        # ground chains stay (their spawn IS a world position, and the wider
        # screen legitimately shows that strip of world at level start).
        for name, sx_, _sy, sline in lv.starters:
            if not (BASE_W <= sx_ < target_w):
                continue
            key = (lv.starter_src, sline)
            if key in site_seen:
                continue
            site_seen.add(key)
            trig = lv.triggers.get(name)
            kind = trigger_spawn_kind(lv, trig)
            if kind == "ground":
                continue
            findings.append(dict(cls="SSTART", level=lv.name, trigger=name,
                                 x=sx_, prop_line=sline, src=lv.starter_src,
                                 kind=kind))

        for tname, trig in sorted(lv.triggers.items()):
            src = lv.trigger_src[tname]
            for i, e in enumerate(trig.entries):
                if e.kind != "enemy":
                    continue
                beh, px, pp, absolute = entry_pos(e)
                chain = (src, tname, i)
                fresh = chain not in seen
                if fresh:
                    info = dict(levels={lv.name}, behavior=beh,
                                starter=tname in starter_trigs, findings=[],
                                scrolls=set())
                    seen[chain] = info
                else:
                    info = seen[chain]
                    info["levels"].add(lv.name)
                    # a chain is starter-driven if ANY level starter-drives it
                    if tname in starter_trigs and not info["starter"]:
                        info["starter"] = True
                        for g in info["findings"]:
                            g["starter"] = True
                # timer-despawn outcomes depend on the level's scroll speed, and
                # behavior names resolve to DIFFERENT files per level (the shop
                # target skins), so the kinematic sim runs once per (chain,
                # scroll, resolution); statics once per chain
                kin_key = (lv.scroll, lv.behavior_src.get(beh) if beh else None)
                kin_fresh = kin_key not in info["scrolls"]
                info["scrolls"].add(kin_key)
                common = dict(level=lv.name, trigger=tname, entry=f"{e.name}#{i}",
                              behavior=beh, src=src,
                              bsrc=lv.behavior_src.get(beh),
                              starter=tname in starter_trigs)

                # ---- spawn-position classes (static, once per chain) ----
                # Both tests compare what the value DOES at the two widths, so a
                # value this sweep has already shifted stops matching (idempotent).
                cls = None
                if absolute == 1 and SUS_LO <= px < target_w:
                    # right-edge staging, now on-screen; screen-absolute
                    # regardless of caller (absolute = 2 is world-anchored: never touch)
                    cls = "RSTAGE"
                elif absolute == 0 and px <= LEFT_ANCHOR \
                        and target_w + px > -HALF_W and BASE_W + px <= -HALF_W:
                    # off the left edge at 800, visible at the target width; the
                    # 800-side guard also makes a shifted value stop matching
                    # (its target spawn equals the authored 800 spawn, <= -32).
                    # Parent-relative when behavior-called -> only fixable if the
                    # right-edge interpretation is the one actually used
                    cls = "CONFLICT" if tname in beh_called else "LPOP"
                if cls and fresh:
                    g = dict(cls=cls, x=px,
                             wanted="LPOP" if cls == "CONFLICT" else None,
                             prop_line=pp.line if pp else None, **common)
                    findings.append(g)
                    info["findings"].append(g)

                # ---- kinematic classes ----------------------------------
                # Starter-driven chains spawn at the right edge.  Behavior-called
                # (child) chains spawn at their parent -- worst case ALSO the
                # right edge (shop target spawners, rear-fired waves), so they
                # use the same spawn model, but only path-exhaustion fixes are
                # allowed for them (see sweep): extending a leftward crosser past
                # the left edge is invisible, while retiming a child's timer
                # could disturb scripted boss sequences.
                if beh and kin_fresh and absolute == 0 \
                        and (tname in starter_trigs or tname in beh_called):
                    # the 800 reference uses the PRISTINE pos for this entry
                    bpx = px
                    btrig = lvb.triggers.get(tname)
                    if btrig is not None:
                        bents = [x for x in btrig.entries if x.kind == "enemy"]
                        ents = [x for x in trig.entries if x.kind == "enemy"]
                        if len(bents) == len(ents):
                            k = ents.index(e)
                            _bb, bpx, _bp, _ba = entry_pos(bents[k])
                    # child chains spawn at their parent, worst case staged past
                    # the right edge; same margin on both sides keeps the
                    # comparison edge-relative and the fix amount exact
                    stage_off = 0 if tname in starter_trigs else CHILD_STAGE
                    ob = sim_b.run(beh, BASE_W + bpx + stage_off)
                    ot = sim_t.run(beh, target_w + px + stage_off)
                    if ob.osl_at_spawn and not ot.osl_at_spawn and cls is None \
                            and fresh and tname in starter_trigs:
                        findings.append(dict(cls="LPOP", x=px, stall=True,
                                             prop_line=pp.line if pp else None,
                                             **common))
                    # enemies that pick a random cruise speed at spawn can die a
                    # different way per branch (slow -> timer, fast -> path end):
                    # simulate every branch, not just the pessimistic default
                    pairs = [(ob, ot, None)]
                    for s in sorted(ot.rand_speeds):
                        pairs.append(
                            (sim_b.run(beh, BASE_W + bpx + stage_off, rand_speed=s),
                             sim_t.run(beh, target_w + px + stage_off, rand_speed=s),
                             s))
                    for ob_i, ot_i, s in pairs:
                        v = classify_vanish(ob_i, ot_i, target_w)
                        if not v:
                            continue
                        # one finding per (death mechanism, fix site) is enough --
                        # the site is (file, line), deduped across ALL levels
                        seg_i = ot_i.segments[-1] if ot_i.segments else None
                        site = ("VANISH", ot_i.death_kind,
                                ot_i.death_bsrc if ot_i.death_kind == "timekill"
                                else (seg_i.bsrc if seg_i else None),
                                ot_i.death_event.line if ot_i.death_event else
                                (seg_i.path_prop.line if seg_i else None))
                        if site in site_seen:
                            continue
                        site_seen.add(site)
                        findings.append(dict(
                            cls="VANISH", death800=ob_i.death_x,
                            deathW=ot_i.death_x, t800=ob_i.death_t,
                            tW=ot_i.death_t, fuzzy=ot_i.fuzzy, direction=v,
                            branch_speed=s,
                            child=tname not in starter_trigs,
                            notes="; ".join(dict.fromkeys(ot_i.notes)),
                            out800=ob_i, outW=ot_i, **common))

        # ---- absolute path nodes staged past the old right edge ---------
        for pname, pdef in sorted(lv.paths.items()):
            for p in pdef.props:
                n = nums(p.vals)
                if p.key in ABS_NODES and n and n[0] is not None \
                        and SUS_LO <= n[0] <= SUS_HI:
                    key = (lv.path_src[pname], pname, p.line)
                    if key in seen:
                        seen[key]["levels"].add(lv.name)
                        continue
                    seen[key] = dict(levels={lv.name})
                    findings.append(dict(cls="ABSNODE", level=lv.name, path=pname,
                                         x=n[0], prop_line=p.line,
                                         src=lv.path_src[pname]))

        # ---- path_limit values in the suspicious band --------------------
        for bname, bdef in sorted(lv.behaviors.items()):
            for e in bdef.entries:
                for p in e.props:
                    if p.key != "path_limit":
                        continue
                    n = nums(p.vals)
                    if any(x is not None and SUS_LO <= x <= SUS_HI for x in n):
                        key = (lv.behavior_src[bname], bname, p.line)
                        if key in seen:
                            seen[key]["levels"].add(lv.name)
                            continue
                        seen[key] = dict(levels={lv.name})
                        findings.append(dict(cls="LIMIT", level=lv.name,
                                             behavior=bname, vals=p.vals,
                                             prop_line=p.line,
                                             src=lv.behavior_src[bname]))
    return findings, seen


def audit_timers(gd, min_ms=3000):
    """Every long timer-despawn site (<on_timer> + part_state.0 = 0) in every level's
    scope, with the overlay's current value.  -> [(src, behavior, T_orig, T_now)]"""
    if gd.assets not in _PRISTINE:
        _PRISTINE[gd.assets] = GameData(gd.assets)
    gdp = _PRISTINE[gd.assets]
    sites = {}
    for lp in sorted(glob.glob(os.path.join(gd.assets, "DATA/level/*/lvl*.txt"))):
        lv = Level(gd, lp)
        lvb = Level(gdp, lp)
        for bname, bdef in lv.behaviors.items():
            src = lv.behavior_src[bname]
            bb = lvb.behaviors.get(bname)
            for e in bdef.entries:
                if e.kind != "on_timer":
                    continue
                if not any(p.key == "part_state.0" and nums(p.vals)[:1] == [0.0]
                           for p in e.props):
                    continue
                try:
                    t_now = int(e.name)
                except ValueError:
                    continue
                t_orig = t_now
                if bb is not None:
                    for eb in bb.entries:
                        if eb.kind == "on_timer" and eb.line == e.line:
                            try:
                                t_orig = int(eb.name)
                            except ValueError:
                                pass
                if t_orig >= min_ms:
                    sites[(src, bname, e.line)] = (src, bname, t_orig, t_now)
    return sorted(sites.values(), key=lambda s: -s[2])


def print_report(findings, seen, target_w):
    by_cls = {}
    for f in findings:
        by_cls.setdefault(f["cls"], []).append(f)
    print(f"target width {target_w}: "
          + ", ".join(f"{c}:{len(v)}" for c, v in sorted(by_cls.items())) or "clean")
    for cls in ("LPOP", "RSTAGE", "SSTART", "VANISH", "ABSNODE", "LIMIT",
                "CONFLICT"):
        rows = by_cls.get(cls, [])
        if not rows:
            continue
        print(f"\n== {cls}: {len(rows)}")
        for f in rows:
            if cls == "SSTART":
                extra = f"x={f['x']} ({f['kind']})"
            elif cls in ("LPOP", "RSTAGE", "CONFLICT"):
                extra = (f"pos.x={f['x']:.0f}" + (" STALL-only" if f.get("stall") else "")
                         + ("" if f.get("starter") else " (not starter-driven here)"))
            elif cls == "VANISH":
                extra = (f"{f.get('direction', '?'):5} "
                         f"death 800:{f['death800'] and round(f['death800'])} "
                         f"W:{f['deathW'] and round(f['deathW'])}"
                         + (" FUZZY" if f.get("fuzzy") else ""))
                if f.get("notes"):
                    extra += f"  ({f['notes']})"
            elif cls == "ABSNODE":
                extra = f"path={f['path']} x={f['x']:.0f}"
            else:
                extra = f"path_limit = {', '.join(f['vals'])}"
            print(f"  {f['level']:16} {f.get('trigger') or f.get('behavior') or f.get('path'):26} "
                  f"{str(f.get('entry') or ''):20} {extra}   [{f.get('bsrc') or f['src']}]")


def main():
    import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=config.LOGICAL[0])
    ap.add_argument("--timers", action="store_true",
                    help="audit every long timer-despawn site instead of scanning")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    gd = GameData(config.ASSETS)
    if args.timers:
        for src, bname, t_orig, t_now in audit_timers(gd):
            mark = f" -> {t_now}" if t_now != t_orig else ""
            print(f"  {t_orig:>7}ms{mark:>12}  {bname:28} {src}")
        return
    findings, seen = scan(gd, args.width, args.verbose)
    print_report(findings, seen, args.width)


if __name__ == "__main__":
    main()
