# How it works

A complete technical account of the mod: the engine internals that were reverse-engineered,
the file formats, and exactly why each piece is correct. Everything here was derived from the
shipped (unstripped) 32-bit `jng_gold` binary with `objdump`/`gdb`.

## Contents
1. [The engine at a glance](#1-the-engine-at-a-glance)
2. [The `.dat` archive format](#2-the-dat-archive-format)
3. [The override overlay](#3-the-override-overlay)
4. [Widescreen](#4-widescreen)
5. [Why HD needs a binary patch](#5-why-hd-needs-a-binary-patch)
6. [The binary patch](#6-the-binary-patch)
7. [Transparency / color-key](#7-transparency--color-key)
8. [The upscaling pipeline](#8-the-upscaling-pipeline)

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
