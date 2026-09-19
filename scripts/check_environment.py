import importlib.util

import torch


print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")

for package in ("transformers", "streamlit", "captum", "umap", "nnsight", "transformer_lens"):
    print(f"{package}: {'installed' if importlib.util.find_spec(package) else 'not installed'}")

