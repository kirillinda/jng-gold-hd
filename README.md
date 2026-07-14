# Jets'n'Guns Gold — HD + Widescreen Mod

A mod for the **Linux Steam build of Jets'n'Guns Gold (v1.308 ST)** that:

- adds **true 16:9 widescreen** (Hor+ — you see more to the sides, HUD stays correct), and
- renders every sprite, texture and effect from **4× AI-upscaled art** at the correct
  on-screen size, so the 20-year-old graphics look crisp on a modern display.

It is delivered as a **non-destructive overlay** plus a tiny binary patch — the original
`jng.dat` is never touched, and everything is fully reversible.

> ⚠️ You must own Jets'n'Guns Gold. This repo contains **only tooling and docs** — it builds
> everything from your own copy of the game and never redistributes the game's files or art
> (© Rake in Grass). Shared **non-commercially with the developers' blessing**; do not sell it
> or anything built with it.

---

## Quick start

```bash
git clone <this-repo> jng-gold-hd && cd jng-gold-hd
./build.sh                 # downloads the upscaler + default model, builds everything
```

`build.sh` produces two folders under `dist/`:

| Folder | What it is |
| --- | --- |
| `dist/mod-dropin/` | **Only the changed files** — `jng_gold`, `hd.dat`, `Data.ini`, `Game.cfg` + `install.sh`. Copy into your game folder for a drop-in install. |
| `dist/patched-game/` | A **complete, ready-to-run** copy of the patched game. |

Install the drop-in:

```bash
cd dist/mod-dropin
./install.sh "$HOME/.local/share/Steam/steamapps/common/JnG Gold"
```

`install.sh` backs up the originals to `*.orig`; `uninstall.sh` restores them. Then launch
the game from Steam as usual.

### Choosing the upscale model

```bash
./build.sh                      # default: 4x_NMKD-Siax_200k (sharp, detailed)
./build.sh realesrgan-x4plus    # any model present in tools/upscaler/models/
```

If you pass a model name, it must already be in `tools/upscaler/models/` as a
`NAME.param` + `NAME.bin` pair (grab more from
[openmodeldb.info](https://openmodeldb.info) or the
[Upscayl models](https://github.com/upscayl/custom-models)). With no argument, the default
model is downloaded automatically. Results are cached per-model, so switching is cheap.

---

## Requirements

- The Linux Steam build of **Jets'n'Guns Gold 1.308 ST**.
- A **Vulkan-capable GPU** (the upscaler is `realesrgan-ncnn-vulkan`; it runs on AMD/Intel/NVIDIA
  with the system Vulkan driver — no CUDA/ROCm needed). Developed on an AMD RX 7900 XTX.
- `python3`, `curl`, `unzip`, and a working Vulkan loader.

---

## Repository layout

```
build.sh                 one-shot builder (see top of file for options)
tools/
  config.py              paths / parameters (env-overridable)
  jngdat.py              reader + writer for the game's LZO .dat archives
  extract.py             unpack the .dat archives into assets/
  upscale.py             the upscaling pipeline (transparency-aware)
  build_batch.py         upscale every asset 4x and pack build/hd.dat (GPU batch)
  patch_hd.py            binary-patch the game so 4x textures draw at correct size
  make_gamecfg.py        generate a widescreen Game.cfg
  install.sh/uninstall.sh  drop-in (un)installer
assets/                  the game's unpacked art (GENERATED from your own game copy;
                         git-ignored — never redistributed)
docs/HOW_IT_WORKS.md     full technical write-up (engine internals + why the patch works)
```

Regenerable/large/proprietary things (`tools/venv/`, `tools/upscaler/`, `upscaled/` cache,
`build/`, `dist/`, and the game's `assets/` + binaries) are git-ignored. The repo ships only
the tooling and docs — you build everything from **your own** legally-owned copy of the game.

---

## Editing the art yourself

`build.sh` unpacks your game's art into `assets/DATA/...` on first run (or run
`tools/venv/bin/python tools/extract.py` manually). Edit any `.bmp/.jpg/.tga` there (keep the
magenta `255,0,255` background as the transparent key for sprites), then re-run `./build.sh`
— only changed files are re-upscaled and repacked. `assets/` is git-ignored, so your edits
stay local and the game's copyrighted art is never committed.

---

## How it works (short version)

1. **Widescreen** is a config change: the engine renders into a pixel-space
   `glOrtho(0, Width, Height, 0)`, so we set a 16:9 logical resolution at the native
   600px height (`1067×600`). Gameplay and HUD reposition correctly.
2. **The engine draws every texture 1:1** — one texture pixel = one logical unit — so a
   bigger texture would just draw bigger. The **binary patch** makes each loaded texture
   report its size as ¼ of the real (4×) pixels, so sprites keep their original size and
   position while sampling 4× the detail. This is a 4-instruction change in
   `CRXTexture::Load`.
3. **The overlay** (`hd.dat`) is a valid game archive containing 4×-upscaled versions of
   every image; it's listed first in `Data.ini` so it overrides `jng.dat` (first match
   wins) without modifying the original.

The full story — the `.dat` format, the color-key handling, and exactly why the ÷4 patch is
mathematically correct — is in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

---

## Credits

- **Jets'n'Guns Gold** © [Rake in Grass](https://www.rakeingrass.com/).
- Upscaling via [Real-ESRGAN / realesrgan-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN)
  with the community **4x_NMKD-Siax_200k** model.
