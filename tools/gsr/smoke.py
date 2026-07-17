#!/usr/bin/env python3
"""Container smoke test: ROCm GPU visible, spandrel loads the model, FP16 forward
runs and returns exactly 4x. Run inside the image before a full build."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("torch", torch.__version__, "hip", torch.version.hip)
print("cuda.is_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "no ROCm device visible — check --device=/dev/kfd,/dev/dri"
print("device", torch.cuda.get_device_name(0))

from spandrel import ModelLoader, ImageModelDescriptor
name = os.environ.get("GSR_MODEL", "4x-UltraSharpV2_Lite")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", name + ".safetensors")
md = ModelLoader().load_from_file(path)
assert isinstance(md, ImageModelDescriptor)
print("loaded", name, "arch=", md.architecture.name if hasattr(md, "architecture") else "?",
      "scale=", md.scale, "in_ch=", md.input_channels)
md.to("cuda").eval()
md.model.half()

for hw in (64, 128, 256):
    x = torch.rand(1, 3, hw, hw, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); t = time.time()
    with torch.inference_mode():
        y = md(x)
    torch.cuda.synchronize()
    print(f"  in {hw}x{hw} -> out {tuple(y.shape[-2:])}  ({(time.time()-t)*1000:.0f} ms)")
    assert y.shape[-1] == hw * 4 and y.shape[-2] == hw * 4, "not 4x!"
print("SMOKE OK")
