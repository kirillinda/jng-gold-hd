# Asset curator

A local web GUI for hand-curating FLUX detail passes on top of the GSR
upscale cache. The verdict from the lab-flux experiments stands: FLUX cannot
run unattended (it rewrites text and iconography at every usable strength),
but confined to a hand-painted mask with a human picking the result, it adds
detail the SR model cannot.

## Workflow

1. `tools/curator/run.sh` → open <http://localhost:7860>.
2. Pick an asset (sidebar, searchable). The original and the GSR upscale are
   shown side by side; paint a mask on either one — it lands in the same
   output-resolution mask either way.
3. Tune parameters if needed (defaults are the values validated in the
   lab-flux test batches), optionally type a prompt (empty = unconditioned),
   press **Generate 6 options** — 3 seeds × ultrasharp-v2 + 3 seeds ×
   ultrasharp-v3 (LoHa, hand-merged).
4. Click an option to select it (the full-size candidate panel shows it
   composited into the sheet; hold the mouse button there to flip to the
   current base). **Save selected** writes the composite; **Use GSR as-is**
   accepts the plain upscale; **Skip** defers.
5. **Multi-stage refinement**: after a save, the middle panel becomes the
   saved result (stage 1, 2, …) and you can paint a new mask on it and
   generate again — refinement passes produce one quick sample per checked
   family, with the seed rolled on every press so each generate gives a new
   variation. **Undo last save** steps back one stage (previous stages are
   kept on disk); options from before a save are marked stale and cannot be
   accidentally composited onto the newer base.
6. Everything persists under `lab-curator/` (gitignored): masks, generated
   batches, per-asset parameters, choices, stage history. Close the browser
   or kill the server any time; it resumes where you left off.
7. `tools/curator/apply.sh` overlays `lab-curator/chosen/` (the latest stage
   of every curated asset) onto the cache; rebuild `hd.dat` after that.
   Re-run apply after any full cache rebuild.

## How the masked pass works

- The mask's bounding box is padded by 48px, grown to ≥320px per side and
  aligned to /16 (FLUX latent constraint); only that crop is diffused
  (img2img over the GSR pixels, composited over gray where transparent).
- The result is blended back **only inside the mask**: Gaussian-feathered
  (`feather` slider) and weighted by the sprite's alpha, so colorkey/alpha
  areas and everything outside the mask stay byte-identical GSR pixels.
- Chosen files are written in the cache's own format (RGBA BMP / TGA /
  JPEG q94 subsampling 1). Caveat: for JPEG the whole file is re-encoded
  once, so untouched regions go through one extra q94 pass.
- Mask painted outside the processed crop box is ignored only in the rare
  case where the box hits the sheet edge and cannot align to /16.

## Moving parts

- `server.py` — FastAPI: asset list, state (`lab-curator/state.json`), mask
  store, generation jobs (single worker thread), compositing, image serving.
- `engine.py` — FLUX.1-dev img2img on the 24GB card: prompt embeddings
  cached with text encoders off-GPU, transformer stored fp8 (layerwise
  casting), v2 LoRA fused / v3 LoHa hand-merged per variant. Prepared
  variants stay resident (active on GPU, up to three parked in RAM/swap)
  so after the first batch a variant switch is a ~6s PCIe swap, not a
  ~1min rebuild — measured 251s → 33s per 6-option batch, byte-identical
  outputs across the park/restore cycle.
- `static/` — the UI (vanilla JS, no build step).
- `Dockerfile` — `jng-gsr:rocm7` + diffusers + fastapi (image
  `jng-curator:latest`, built automatically by `run.sh`).

Model files are expected in `lab-flux/` (see `run.sh` header); none of them,
and none of the outputs, are committed.
