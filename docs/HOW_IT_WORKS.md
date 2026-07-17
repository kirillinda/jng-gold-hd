# How it works

A complete technical account of the mod: the engine internals that were reverse-engineered,
the file formats, and exactly why each piece is correct. The bulk was derived from the shipped
(unstripped) 32-bit Linux `jng_gold` binary with `objdump`/`gdb`; the Windows `jng_gold.exe`
specifics (see [§9](#9-windows-vs-linux)) were confirmed with `capstone`/`pefile` and a small
Win32 debugger. Both are the same 32-bit RX engine on SDL2 + OpenGL.

## Contents
1. [The engine at a glance](#1-the-engine-at-a-glance)
2. [The `.dat` archive format](#2-the-dat-archive-format)
3. [The override overlay](#3-the-override-overlay)
4. [Widescreen](#4-widescreen) — incl. [4a. 4:3 assumptions in the *code*](#4a-the-43-assumptions-the-config-change-does-not-fix) and [4b. in the *data*](#4b-the-43-assumptions-that-live-in-the-games-data)
5. [Why HD needs a binary patch](#5-why-hd-needs-a-binary-patch)
6. [The binary patch](#6-the-binary-patch)
7. [Transparency / color-key](#7-transparency--color-key)
8. [The upscaling pipeline](#8-the-upscaling-pipeline)
9. [Windows vs. Linux](#9-windows-vs-linux)

---

## 1. The engine at a glance

Jets'n'Guns Gold uses Rake in Grass's in-house C++ **"RX" engine**: SDL2 for windowing/input,
**fixed-function OpenGL** for drawing, and a custom LZO-compressed archive (`jng.dat`) for
assets. The Linux binary is 32-bit and *not stripped*, so class/method names
(`CRXTexture`, `CRXScreen`, `OpenGLSprite::Draw`, …) are visible, which is what made this
feasible. There is leftover `D3DPOOL`/`D3DXVECTOR2` naming — it's a Windows/Direct3D game
ported to Linux behind a thin abstraction.

The game draws in a **pixel-space orthographic projection**:

```
glOrtho(0, Width, Height, 0, -1, 1)      // top-left origin, 1 unit == 1 pixel
```

`Width`/`Height` come from `Game.cfg` (defaults 800×600). There is **no fixed internal
framebuffer** — the logical coordinate system *is* the configured resolution, and a viewport
maps it onto the window (optionally letterboxed to keep the aspect ratio via `ratio43`).

---

## 2. The `.dat` archive format

Reader/writer: [`tools/jngdat.py`](../tools/jngdat.py).

```
Header (16 bytes)
  u32 magic            (per-file, not validated)
  u32 count            number of index entries
  u32 index_offset     absolute file offset of the index
  u32 adler32          adler32 of the first 12 header bytes  ← SEED 0, not the usual 1
Payload region [16, index_offset)
  concatenated, per-file block data (see below)
Index @ index_offset, `count` records:
  u16 namelen                       (includes trailing NUL)
  u8  name[namelen]                 XOR-obfuscated; '\' path separators
  u32 uncompressed_size             XOR-obfuscated
  u32 block_table_offset            XOR-obfuscated
```

**Index obfuscation** (`xorBuf`): a per-entry key starts at `index_offset` and evolves
`key = (key * 0x17BC3) & 0xFFFFFFFF` after each entry; the keystream for word *i* is
`key + i*0x732C2E17`. (The shipped routine only obfuscates the first `len>>2` "units" of each
buffer — a quirk of the original code that we reproduce exactly so round-trips are identical.)

**Per-file storage is block-streamed, not one blob per file.** Each file is a sequence of
**32 KB (0x8000) *uncompressed* blocks**. Its `block_table_offset` points at
`ceil(uncompressed_size / 0x8000)` records of 12 bytes each:

```
u32 absolute_offset    where this block's data is in the file
u32 adler32            per-block checksum (not verified on read)
u16 comp_size          compressed size; 0 = stored raw (read a full 0x8000)
u16 flag
```

Each block's on-disk bytes are **XOR-obfuscated with seed 0** (same keystream), then LZO1X.
To read a block: read `comp_size` bytes → de-XOR (seed 0) → `lzo1x_decompress` to ≤ 0x8000.

Two gotchas the writer must honor (both were real crash bugs during development):

- **Header adler32 uses seed 0**, not the standard `adler32` seed of 1.
- The game reads each compressed block into a **fixed 0x8000 stack buffer**. If a 32 KB chunk
  is near-incompressible (JPEG data), LZO can *expand* it past 0x8000 → buffer overflow →
  crash. The original avoids this by storing such blocks **raw** (`comp_size = 0` → the reader
  loads a full 0x8000). Our packer does the same, padding a partial final block if needed.

LZO is called through the system `liblzo2` via `ctypes` — no compilation, no pip dependency.

---

## 3. The override overlay

`Data.ini` lists the archives to load:

```
data_file = hd.dat        ← added by the mod, loaded FIRST
data_file = update.dat
data_file = jng.dat
```

`InitDataFiles` reads this list and appends every archive's index into one global table.
`_lz_fopen` looks a file up by name with a **first-match-wins** linear scan, so entries from
archives listed earlier win. (`update.dat` is the stock game's own example of this — a tiny
overlay that patches two files in `jng.dat`.) Putting `hd.dat` first means our upscaled files
override the originals **without modifying `jng.dat`**.

---

## 4. Widescreen

Because the logical coordinate space *is* `Width`×`Height`, widescreen is just a config
change — but you can't naïvely raise both. The engine draws art at native pixel size and only
*repositions* the HUD by resolution; it does not scale art. So at, say, `1920×1080` everything
becomes tiny and the ~600px-tall content leaves a black band.

The fix is the standard **Hor+** technique: keep the logical **height at the native 600** and
widen only the **width** to the target aspect (`600 × 16/9 ≈ 1067`). Vertical positions are
unchanged (so gameplay is identical), and you simply see more to the sides. The window can be
any size; the viewport scales `1067×600` onto it. See [`tools/make_gamecfg.py`](../tools/make_gamecfg.py).

(Ultrawide 21:9 would be `Width ≈ 1433`; only the full-screen UI images would need
regenerating at that aspect. Not currently shipped.)

### 4a. The 4:3 assumptions the config change does *not* fix

Setting `Width` is necessary but **not sufficient**: a handful of gameplay sites ignore
`g_Screen` and hardcode the 800×600 the game shipped with. Most of the engine is well
behaved — shot culling (`CShot::Update`), enemy despawn (`CEnemy::Update`), the vertical
kill plane (`CEnemy::UpdateVelocity`), the tilemap visibility rect
(`CRXTileMap::CalculateVisibilityRect`) and the enemy spawn trigger (`ParseNextStarter`)
all read `g_Screen->Width`/`->Height` properly. Five sites do not.

The one that actually breaks the game is in **`CEnemy::Update`**. Enemies are driven by
behavior scripts through numbered events; the parser in `ReadBehaviorEntry` maps the `.txt`
names to ids (`on_screen_left` = 0x1f, `on_screen_right` = 0x20). `CEnemy::Update` fires
them from the sprite's x:

```c
if (spr->x < 0)                                 ParseBehaviorEvent(on_screen_left);
else if (spr->x > (type ? 800 - type->w : 800)) ParseBehaviorEvent(on_screen_right);
//                       ^^^ hardcoded, should be g_Screen->Width
```

`on_screen_left` is fine — 0 is the left edge at any width. `on_screen_right` is not.
`ParseNextStarter` correctly spawns each enemy at the true right edge (it fires when
`scrollX + g_Screen->Width >= starter->x`, so the enemy appears at `x ≈ Width`). On a
1067-wide screen **every enemy is therefore born already past the stale 800 bound and fires
`on_screen_right` on its very first frame**. Measured live under gdb with `type->w = 120`:
vanilla fires at `x > 680`, fixed fires at `x > 947`.

That event means "you reached the right edge, turn back / leave", so the damage is
data-driven and shows up as sprites dying for no reason:

| def | `<on_screen_right>` does | effect at 1067 wide |
| --- | --- | --- |
| `air.torpedo/*_b.txt` | `behavior = torp_exit` | torpedoes fade out on spawn — vanish long before crossing |
| `special.tanker/tanker_b.txt` | `trigger.TANKER_FAILED` | escort mission fails the instant the tanker appears |
| `man/behavior_man.txt`, `special.ogre`, `man.mutant` | `Behavior = MAN_WALK_LEFT` / `OGRE_TURN_L` / `M_TURN_L` | walkers turn back at ~75% across, at an invisible wall |

The other sites are less dramatic: **`GetNearestEnemy`** (homing / auto-aim target
selection) only considers enemies inside a hardcoded 800×600 box, so nothing in the extra
width is targetable; and **`CEnemyPart::ParseCollisions`** saves the `CHECK_POINT` respawn
scroll position as `worldX - 800` instead of `worldX - Width`.

Careful: 800.0f/600.0f/400.0f appear in plenty of *other* places as **speeds** (multiplied
by the frame delta) or as a terminal-velocity clamp — `CEnemy::UpdateAsWreck`,
`CAtarix::Update`, `CHero::UpdateWingLine`, `CScoreNumber::Update`. Those are not screen
bounds and must be left alone. The 800.0f at `.rodata:0x80cf144` is *shared* between real
screen bounds and physics constants, which is why the constant itself can't just be
edited — each site is patched individually.

[`tools/patch_widescreen.py`](../tools/patch_widescreen.py) fixes all five, reading
`g_Screen->Width`/`->Height` at **runtime** (through `ebx`, which GCC pins to the GOT base
`0x80e08f0` in every function here), so no resolution is baked into the binary and any
`Width` in `Game.cfg` stays correct. Four sites are rewritten in place — the replacement is
exactly the size of the original straight-line region, and no branch enters those regions
from outside; the fifth needs 19 bytes it can't borrow, so it calls a small routine placed
in unreferenced `.rodata` alignment padding (`.rodata` is mapped `R E` here). It composes
with `patch_hd.py`, which uses a different cave. **Linux ELF only** — the Windows PE is
stripped and needs its own address survey.

### 4b. The 4:3 assumptions that live in the game's *data*

Fixing the engine isn't enough either: the title screen is built out of level defs, and
every level's ambient particle field declares a spawn rectangle — all in a coordinate space
where the screen is 800 wide. [`tools/make_widescreen_defs.py`](../tools/make_widescreen_defs.py)
re-authors them for the target `Width` and ships them as a tiny separate overlay
(`ws.dat`, ~100KB) listed before `hd.dat`. Keeping them out of the 1.2GB art archive means
changing resolution is a one-second rebuild instead of re-upscaling 1930 images.

Coordinates come in three flavours, and using the wrong transform on the wrong one breaks
things, so each is handled explicitly:

| flavour | transform | why |
| --- | --- | --- |
| absolute, centred composition | `x += (W-800)/2` | the logo, the ESC prompt |
| absolute, right-edge anchored | `x += (W-800)` | the right-hand credits panel, particle spawn strips |
| absolute, left-edge anchored | unchanged | the left-hand credits panel |
| spawn offset relative to the right edge, but really measuring off the **left** edge | `x -= (W-800)` | the logo's left half |
| `pos = -1` | unchanged | engine sentinel — it centres those itself |

The five data bugs, all measured on a 1067×600 logical screen:

1. **Logo off-centre.** The logo is two sprites (`jng_l1`/`jng_r1`, 120px each) whose paths
   end on absolute x=300/420 — straddling 800/2. Fix `+133` → 433/553 (verified in memory:
   both halves rest at exactly 433.00/553.00). *Not* proportional scaling (`x *= W/800` →
   400/560): that pulls the halves 40px apart and tears the logo in half — the gap between
   them is a sprite width, not a screen fraction.

2. **Logo halves arrived at different times.** Both spawn relative to the right edge
   (`jng_l pos = -950`, `jng_r pos = +70`). At 800 that's start −150/870 → both travel 450 →
   they meet together. At 1067: start 117 (on-screen!) / 1137 → 316 vs 584, so the left half
   landed early and it looked broken. `jng_l`'s offset really means "150px off the **left**
   edge" → `-950 - 267 = -1217`. Verified: both halves now land on the same frame (t=682).

3. **Intro jets vanished mid-screen.** Enemies that fly across are removed when their *path*
   runs out (`CEnemy::NextNodePos` clears the alive flag), not when they cross an edge. The
   jets use the global path `ld` (reach 1000) at `path_scale = 100`, spawning at `Width+100`:
   at 800, `900 - 1000 = -100` (just off-screen, correct); at 1067, `1167 - 1000 = +167`.
   Measured under gdb: 7/7 despawned at x=167.4. Fix: scale the *intro jet's* `path_scale` so
   its reach is `Width+200`. Re-measured: 7/7 despawn at −103.

   **Why not just lengthen `ld`:** it's the engine's *universal* straight-line path —
   **1686 call sites** — and `path_scale` is used as a direction vector against it
   (`0,100` = missiles straight up, `-100,100` = 45° diagonals, `100,40` = shallow dives).
   `ld`'s `direct = -1000,-1000` only reads as "reach 1000" when the Y scale is 0. Changing
   its dx would silently re-angle every diagonal and every missile in the game.

4. **Credits / ESC prompt misplaced.** The credits are two 150px panels alternating sides
   every 29s: `titles_1` hugs the left edge (x=20), `titles_2` sits 50px off the right edge
   (x=600). Kept edge-anchored (`titles_2` → 867) to preserve the alternating design; the
   ESC prompt is a centred composition → 553.

5. **Particles spawned mid-screen (game-wide).** Every level's `[FIELD]` emits ambient
   particles (starfield, rain) from screen-space rects. 96 of them across 70 level files, in
   two patterns: a narrow strip at/beyond the old right edge (`800,64,820,600` ×49 and
   `850,0,860,600` ×35 — the shop levels use a 50px margin, so the rule keys off `x1 >= 800`
   and preserves each margin), and a full-width strip above the screen that rain falls from
   (`0,-20,800,-10` ×12 — at 1067 the right 267px would get no rain). Verified in the intro:
   721/803 spawns now land in 1067..1087, was 800..820. `level_zog`'s `650,0,660,520` matches
   neither pattern (650 is mid-screen even at 800), so it's left alone and reported.

---

## 5. Why HD needs a binary patch

`OpenGLSprite::Draw` emits each sprite as a quad whose **on-screen size equals the destination
rectangle**, and the destination is derived from the texture's stored dimensions
(`frame = imageWidth / columns`). UVs are `srcRect / POT_dimensions`. In other words:

> **one texture pixel == one logical unit.** A 4× texture simply draws 4× bigger.

So you cannot add detail by making textures bigger — unless you also tell the engine to treat
them as smaller. That is exactly what the patch does.

A `CRXTexture` stores four dimension fields (offsets found in `CRXTexture::Create`):

```
+0x0  SDL_Surface*      +0xc  image width    +0x14  POT width
+0x4  GL texture id     +0x10 image height   +0x18  POT height
```

Crucially, **`CRXTexture::Upload` uploads to the GPU from the SDL *surface's* own `w/h`**, not
from these members. So the four members are used **only for layout** — sizing, positioning,
and UV normalization — never for the actual GPU upload. That decoupling is what makes the mod
possible.

---

## 6. The binary patch

Patcher: [`tools/patch_hd.py`](../tools/patch_hd.py). It injects, in `CRXTexture::Load`, four
instructions that **divide the four dimension members by 4** (`sar [esi+off],2`):

```
image W /= 4    image H /= 4    POT W /= 4    POT H /= 4
```

We ship every texture at exactly **4×**, so after the ÷4 the engine believes each texture is
its *original* size:

- **Size & position** use image W/H → back to original → sprites are the right size and land
  in the right place.
- **UVs** are `srcRect / POT`. Because ×4 is a power of two, `POT(4·W) == 4·POT(W)` *exactly*,
  so the image-to-POT ratio is unchanged. A source rect expressed in original coordinates,
  divided by the ÷4 POT, maps onto **the full 4× texture** — i.e. GL samples all the extra
  detail. This is why 4 (a power of two, and ≥ the ~3.2× the window scales `1067→3440`) is the
  right factor: the division is exact and the math is lossless.
- The **GPU upload** is untouched (it reads the 4× surface), so the texture really does hold 4×
  the pixels.

Two subtleties, both learned the hard way:

- **Inject *after* the color-key loop.** `CRXTexture::Load` has a loop that keys transparent
  pixels using `+0x14/+0x18` (the POT dims) as its bounds. Dividing the members *before* it ran
  keyed only 1/16 of each sprite (magenta rectangles). The patch is injected at the convergence
  point of both `Load` success paths, *after* keying.
- **Every** image must be 4×. An un-upscaled texture would render at ¼ size; a tiny one could
  divide to 0 and crash (`SIGFPE`). The build guarantees exact-4× coverage of all art.

Mechanically the patch is position-independent: it overwrites a 5-byte `jmp` to a **code cave**
(zero padding after an assert string in `.rodata`, which is in the executable segment), does the
four shifts, replays the displaced instructions, and jumps back. Nothing is relocated, so it's
robust across the exact shipped binary. Fully reversible — keep the `jng_gold.orig` backup.

---

## 7. Transparency / color-key

Most sprites are 8-bit palettized BMPs whose transparent background is **magenta `(255,0,255)`**.
The color key is passed by the caller into `CRXTexture::Load`; the load loop sets any pixel whose
`RGB == key` to fully transparent. FX sprites (fire, explosions; `blend_mode 1`) instead use
**additive blending** (black contributes nothing), and there are grayscale `_w` companion masks.

The pipeline preserves each case (see below). The important constraint: after upscaling, the
transparent region must be **exactly** `(255,0,255)` again, or the key misses and you get magenta
halos. Anti-aliasing/bilinear resizing across a magenta edge produces near-magenta pixels that the
key can't catch — so we never let the upscaler blend *toward* magenta.

---

## 8. The upscaling pipeline

Pipeline: [`tools/upscale.py`](../tools/upscale.py); batch build: [`tools/build_batch.py`](../tools/build_batch.py).

Per sprite, by type:

- **Magenta color-key:** build an alpha mask from the magenta, **bleed** the sprite's edge colors
  outward into the magenta region (so the model/resize never sees magenta to blur), upscale the
  RGBA, then **re-key**: pixels whose upscaled alpha < 128 are set back to exact `(255,0,255)`.
  Output as a 24-bit BMP. Result: clean, halo-free keying.
- **Additive / opaque / grayscale:** upscaled directly in the appropriate mode (black stays black
  for additive FX).
- **Existing RGBA (TGA):** upscaled with its real alpha (the model upscales the alpha channel too).
- **Full-screen 4:3 UI** (menus, comics, loading screens): upscaled and **fitted to the widescreen
  logical size × 4** (`4268×2400`, so ÷4 → `1067×600`), with the 4:3 art centered and a blurred
  cover filling the sides so it reads as full-screen rather than pillar-boxed.
- **Tiny sprites (≤ 8px):** the neural model is unreliable at that size, so they use a
  magenta-safe LANCZOS resize instead (bleed → resize → re-key).

**Performance:** the model runs through `realesrgan-ncnn-vulkan` in **batch mode** — one process
over a whole directory with the model loaded once and the GPU session reused. That is ~24× faster
than invoking it per image (which re-initializes Vulkan and reloads the 64 MB model every time):
~200s for all ~1800 model-eligible images vs. ~40 min. Everything is cached per model under
`upscaled/<model>/`, so changing models or editing a few assets only reprocesses what changed.

---

## 9. Windows vs. Linux

The Windows and Linux Steam builds are the *same* game — Rake in Grass's RX engine on SDL2 +
fixed-function **OpenGL** (the Windows build imports `OPENGL32`/`GLU32`, not Direct3D) — so the
whole approach ports directly. Three things differ, and the tooling handles each automatically
(`patch_hd.py` dispatches on the file magic; `config.py` picks OS-aware defaults).

**The binary patch (PE vs. ELF).** `CRXTexture::Load` has the identical shape on the Windows
`jng_gold.exe`, including the four dimension members at the same offsets (`+0xc/+0x10/+0x14/+0x18`)
and a color-key loop bounded by the POT dims. Only the register assignment and addresses change:
`this` is in **EBX** (not ESI), the injection point — the convergence of both Load success paths,
right after keying — is **`0x408609`**, and the code cave is the `int3` padding at **`0x43ac43`**.
The four instructions become `sar dword [ebx+off], 2`. Same math, same ÷4 invariant.

**PNG is not safe on Windows.** SDL2_image loads PNG through libpng, which it opens *dynamically*;
the shipped game bundles `libjpeg-9.dll` but **no `libpng16-16.dll`**. Handing this SDL2_image a
PNG makes it call through a null libpng pointer and crash (observed as `EIP=0`, an execute fault
at address 0). The Linux system SDL2_image has PNG, which hid this. So RGBA sprites are written as
**uncompressed 32-bit TGA** — byte-identical to the game's own `.tga` art (image type 2, 32 bpp,
descriptor `0x08`) and decoded by SDL2_image's always-present TGA path. `IMG_Load_RW` can't sniff
TGA (it has no magic number), which is exactly why `Load` seeks back and calls `IMG_LoadTGA_RW`
explicitly as a fallback — the path our TGAs take. **No image in `hd.dat` is a PNG.**

**Large-Address-Aware.** HD art is 16× the pixels, so a level's textures need far more transient
memory (the fog tiles alone are 4096×4096 = 64 MB surfaces). A 32-bit process gets a **3 GB** user
address space on 32-bit Linux but only **2 GB** on 64-bit Windows unless the exe is flagged
`IMAGE_FILE_LARGE_ADDRESS_AWARE` — so a level that fits on Linux exhausts the address space on
Windows and `SDL_CreateRGBSurface` starts returning NULL ("Unable to load texture"). The PE patch
sets that flag (one bit in the FILE_HEADER characteristics), lifting the ceiling to 4 GB. This is
the standard large-texture-pack fix and is only applied to the Windows binary.

**Widescreen config location.** The logical `Width`/`Height` (and `ratio43`) live in the
game-folder `Game.cfg` on Linux, but in `Documents\JnGGold\Game.ini` under `[VIDEO]` on Windows,
read via `GetPrivateProfileInt`. The game's config *writer* never emits `Width`/`Height`, so the
values the installer adds persist across runs. `install.ps1` edits that file (and backs it up).
