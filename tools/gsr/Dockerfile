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
    # gfx1100 identity (belt-and-suspenders; the base already targets it).
    HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    # Fragmentation-resistant allocator for the many differently-sized sprites.
    PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    # MIOpen: allow kernel auto-tuning + persistent cache (faster convs, WMMA).
    MIOPEN_FIND_MODE=3 \
    MIOPEN_USER_DB_PATH=/work/.miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/work/.miopen

# liblzo2 gives jngdat.py a zero-Python-dep .dat (de)compressor; lzallright is
# the portable fallback. Pillow/numpy for image work.
RUN apt-get update && apt-get install -y --no-install-recommends liblzo2-2 \
    && rm -rf /var/lib/apt/lists/*

# spandrel loads the SR model weights (DAT2 + RealPLKSR live in core/extra-arches).
# --no-deps so we never clobber the base image's ROCm torch/torchvision with a
# CPU build; we add spandrel's own light runtime deps explicitly.
RUN pip install --no-deps "spandrel>=0.4.1" "spandrel_extra_arches>=0.2.0" \
    && pip install "safetensors>=0.4" "einops>=0.7" "pillow>=10" "lzallright>=0.2.6" \
    && python -c "import torch, spandrel, spandrel_extra_arches; print('torch', torch.__version__, 'hip', torch.version.hip)"

WORKDIR /work
