# GPU super-resolution image for the JnG Gold HD mod.
#
# Fresh build on the official ROCm 7.x PyTorch base so the gfx1100 (RX 7900 XTX,
# RDNA3) matrix kernels (hipBLASLt / MIOpen / rocWMMA) are the tuned ROCm-7 ones.
# The neural upscaler runs in FP16 so conv/matmul dispatch to the RDNA3 WMMA
# units; on gfx1100 Flash-Attention is NOT a win (aotriton flash is often slower
# than the math path), so the default conv-only model (RealPLKSR) avoids the
# attention bottleneck entirely — see build_hd_gsr.py.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Do NOT set HSA_OVERRIDE_GFX_VERSION here. The RX 7900 XTX is natively
    # gfx1100 so it is a no-op for the dGPU — but it is applied to EVERY visible
    # agent, which makes the Ryzen Raphael iGPU (gfx1036) also claim to be
    # gfx1100. ROCm then happily enumerates it and would load RDNA3 code objects
    # onto RDNA2 hardware. run.sh pins HIP_VISIBLE_DEVICES to the dGPU instead.
    #
    # Do NOT set PYTORCH_HIP_ALLOC_CONF=expandable_segments:True either. It looks
    # right for our thousands of differently-sized sprites, but on ROCm 7.2.4 its
    # VA remapping is broken under exactly that access pattern: measured 14.95%
    # of outputs came back as garbage (NaN/Inf, and even boolean-mask .sum()
    # returning negative counts) vs 0.26% with it off, same GPU, same run length.
    # It also disturbs other clients of the card — this box drives its display
    # from the same 7900 XTX. The default caching allocator is correct here.
    #
    # MIOpen: allow kernel auto-tuning + persistent cache (faster convs, WMMA).
    MIOPEN_FIND_MODE=3 \
    MIOPEN_USER_DB_PATH=/work/.miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/work/.miopen

# liblzo2 gives jngdat.py a zero-Python-dep .dat (de)compressor; lzallright is
# the portable fallback. Pillow/numpy for image work.
# liblzo2: jngdat .dat codec. potrace + librsvg2-bin: de-jag sprite silhouettes
# (vectorise the 1-bit color-key mask into smooth curves, rasterise with AA).
RUN apt-get update && apt-get install -y --no-install-recommends \
        liblzo2-2 potrace librsvg2-bin \
    && rm -rf /var/lib/apt/lists/*

# spandrel loads the SR model weights (DAT2 + RealPLKSR live in core/extra-arches).
# --no-deps so we never clobber the base image's ROCm torch/torchvision with a
# CPU build; we add spandrel's own light runtime deps explicitly.
RUN pip install --no-deps "spandrel>=0.4.1" "spandrel_extra_arches>=0.2.0" \
    && pip install "safetensors>=0.4" "einops>=0.7" "pillow>=10" "lzallright>=0.2.6" \
    && python -c "import torch, spandrel, spandrel_extra_arches; print('torch', torch.__version__, 'hip', torch.version.hip)"

WORKDIR /work
