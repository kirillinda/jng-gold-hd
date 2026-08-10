# FLUX img2img engine for the curator server.
#
# Keeps one transformer variant resident on the GPU and swaps it only when the
# requested (LoRA family, scale) changes:
#   v2 = standard kohya LoRA          -> diffusers load_lora_weights + fuse
#   v3 = LyCORIS LoHa (diffusers can't load it) -> hand-merged into the weights:
#        W += scale * (alpha/r) * (W1a@W1b) ⊙ (W2a@W2b), with kohya's fused
#        qkv / linear1 matrices row-sliced onto diffusers' split projections.
# Prompt embeddings are cached per prompt string; the text encoders live on the
# CPU and only visit the GPU while encoding a prompt we haven't seen yet (the
# transformer is dropped first, so both never share the 24GB card).
import gc
import os

import torch
from safetensors.torch import load_file

FLUX = os.environ.get("CURATOR_FLUX", "/work/lab-flux/flux-dev")
LORA_V2 = os.environ.get("CURATOR_LORA_V2",
                         "/work/lab-flux/lora/ultrasharp-v2-flux.safetensors")
LORA_V3 = os.environ.get("CURATOR_LORA_V3",
                         "/work/lab-flux/lora/ultrasharp-v3-flux.safetensors")

D = 3072  # FLUX hidden size


def _targets(mod):
    """kohya LoHa module name -> [(diffusers param name, row_start, row_end)]."""
    if mod.startswith("lora_unet_double_blocks_"):
        rest = mod[len("lora_unet_double_blocks_"):]
        n, rest = rest.split("_", 1)
        b = f"transformer_blocks.{n}"
        table = {
            "img_attn_qkv": [(f"{b}.attn.to_q.weight", 0, D),
                             (f"{b}.attn.to_k.weight", D, 2 * D),
                             (f"{b}.attn.to_v.weight", 2 * D, 3 * D)],
            "txt_attn_qkv": [(f"{b}.attn.add_q_proj.weight", 0, D),
                             (f"{b}.attn.add_k_proj.weight", D, 2 * D),
                             (f"{b}.attn.add_v_proj.weight", 2 * D, 3 * D)],
            "img_attn_proj": [(f"{b}.attn.to_out.0.weight", None, None)],
            "txt_attn_proj": [(f"{b}.attn.to_add_out.weight", None, None)],
            "img_mlp_0": [(f"{b}.ff.net.0.proj.weight", None, None)],
            "img_mlp_2": [(f"{b}.ff.net.2.weight", None, None)],
            "txt_mlp_0": [(f"{b}.ff_context.net.0.proj.weight", None, None)],
            "txt_mlp_2": [(f"{b}.ff_context.net.2.weight", None, None)],
            "img_mod_lin": [(f"{b}.norm1.linear.weight", None, None)],
            "txt_mod_lin": [(f"{b}.norm1_context.linear.weight", None, None)],
        }
        return table[rest]
    if mod.startswith("lora_unet_single_blocks_"):
        rest = mod[len("lora_unet_single_blocks_"):]
        n, rest = rest.split("_", 1)
        b = f"single_transformer_blocks.{n}"
        table = {
            "linear1": [(f"{b}.attn.to_q.weight", 0, D),
                        (f"{b}.attn.to_k.weight", D, 2 * D),
                        (f"{b}.attn.to_v.weight", 2 * D, 3 * D),
                        (f"{b}.proj_mlp.weight", 3 * D, 3 * D + 4 * D)],
            "linear2": [(f"{b}.proj_out.weight", None, None)],
            "modulation_lin": [(f"{b}.norm.linear.weight", None, None)],
        }
        return table[rest]
    return None  # lora_te1_* (CLIP part) — not applied, same as the test batch


def _merge_loha(tr, scale, log):
    sd = load_file(LORA_V3)
    mods = sorted({k.rsplit(".", 1)[0] for k in sd if k.startswith("lora_unet_")})
    params = dict(tr.named_parameters())
    applied = skipped = 0
    for mod in mods:
        tgt = _targets(mod)
        if tgt is None:
            skipped += 1
            continue
        w1a = sd[mod + ".hada_w1_a"].to("cuda", torch.float32)
        w1b = sd[mod + ".hada_w1_b"].to("cuda", torch.float32)
        w2a = sd[mod + ".hada_w2_a"].to("cuda", torch.float32)
        w2b = sd[mod + ".hada_w2_b"].to("cuda", torch.float32)
        alpha = float(sd[mod + ".alpha"])
        r = w1b.shape[0]
        delta = (w1a @ w1b) * (w2a @ w2b) * (alpha / r) * scale
        for name, r0, r1 in tgt:
            p = params[name]
            d = delta if r0 is None else delta[r0:r1]
            assert d.shape == p.shape, (mod, name, d.shape, p.shape)
            p.data.add_(d.to(p.dtype).cpu())
        del w1a, w1b, w2a, w2b, delta
        applied += 1
    torch.cuda.empty_cache()
    assert skipped == 0, f"{skipped} LoHa modules unmapped"
    log(f"v3 LoHa merged ({applied} modules, scale {scale})")


class Engine:
    # Prepared (fused/merged + fp8-cast) transformers are kept resident: the
    # active one on the GPU, the others parked in system RAM, swapped over
    # PCIe in seconds instead of a ~1min rebuild from disk. ~12GB per parked
    # variant; the box has 46GB + a large swap, so three variants (e.g. two
    # scales of one family) stay warm. The text encoders (~10GB) are loaded
    # lazily and dropped during first-time builds so the transient bf16 load
    # (~24GB) doesn't stack on top of everything at once.
    MAX_CACHE = 3

    def __init__(self):
        from diffusers import FluxImg2ImgPipeline
        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            FLUX, transformer=None, text_encoder=None, text_encoder_2=None,
            torch_dtype=torch.bfloat16)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.to("cuda")
        # Text encoders stay OFF the pipe except while encoding: diffusers
        # derives its execution device from registered components, and
        # CPU-parked encoders would drag latent prep onto the CPU.
        self._te = self._te2 = None
        self._embeds = {}
        self._cache = {}     # (family, scale) -> prepared transformer
        self.variant = None  # key of the variant currently on the GPU

    def _park(self):
        tr = self.pipe.transformer
        if tr is not None:
            tr.to("cpu")
            self.pipe.transformer = None
        self.variant = None
        gc.collect()
        torch.cuda.empty_cache()

    def encode(self, prompt, log=print):
        if prompt in self._embeds:
            return self._embeds[prompt]
        self._park()
        if self._te is None:
            log("loading text encoders")
            from transformers import CLIPTextModel, T5EncoderModel
            self._te = CLIPTextModel.from_pretrained(
                FLUX, subfolder="text_encoder", torch_dtype=torch.bfloat16)
            self._te2 = T5EncoderModel.from_pretrained(
                FLUX, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
        log("encoding prompt")
        p = self.pipe
        p.text_encoder, p.text_encoder_2 = self._te, self._te2
        p.text_encoder.to("cuda")
        p.text_encoder_2.to("cuda")
        with torch.inference_mode():
            emb, pooled, _ = p.encode_prompt(
                prompt=prompt, prompt_2=None, device="cuda",
                max_sequence_length=256)
        p.text_encoder.to("cpu")
        p.text_encoder_2.to("cpu")
        p.text_encoder = p.text_encoder_2 = None
        self._embeds[prompt] = (emb.detach(), pooled.detach())
        gc.collect()
        torch.cuda.empty_cache()
        return self._embeds[prompt]

    def ensure(self, family, scale, log=print):
        want = (family, round(float(scale), 3))
        if self.variant == want:
            return
        self._park()
        tr = self._cache.pop(want, None)
        if tr is None:
            # RAM guard: fresh bf16 load (~24GB) + parked variant (~12GB) +
            # text encoders (~10GB) would exceed this box. Prompt embeddings
            # stay cached, so the encoders are rarely needed again.
            if self._cache and self._te is not None:
                self._te = self._te2 = None
                gc.collect()
            while len(self._cache) >= self.MAX_CACHE:
                self._cache.pop(next(iter(self._cache)))
                gc.collect()
            from diffusers import FluxTransformer2DModel
            log(f"building transformer ({family} ×{scale}) — first use, "
                "about a minute")
            tr = FluxTransformer2DModel.from_pretrained(
                FLUX, subfolder="transformer", torch_dtype=torch.bfloat16)
            self.pipe.transformer = tr
            if scale > 0:
                if family == "v2":
                    self.pipe.load_lora_weights(LORA_V2)
                    self.pipe.fuse_lora(lora_scale=float(scale))
                    self.pipe.unload_lora_weights()
                    log(f"v2 LoRA fused (scale {scale})")
                elif family == "v3":
                    _merge_loha(tr, float(scale), log)
            tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                        compute_dtype=torch.bfloat16)
        else:
            log(f"swapping in cached transformer ({family} ×{scale})")
            self.pipe.transformer = tr
        tr.to("cuda")
        self._cache[want] = tr  # re-insert = most recently used
        self.variant = want

    def generate(self, image, prompt, strength, steps, guidance, seed):
        emb, pooled = self._embeds[prompt]
        g = torch.Generator("cpu").manual_seed(int(seed))
        with torch.inference_mode():
            return self.pipe(
                prompt_embeds=emb.to("cuda"), pooled_prompt_embeds=pooled.to("cuda"),
                image=image, width=image.width, height=image.height,
                strength=float(strength), guidance_scale=float(guidance),
                num_inference_steps=int(steps), generator=g).images[0]
