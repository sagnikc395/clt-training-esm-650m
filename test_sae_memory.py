from sparsify.sparse_coder import SparseCoder
from sparsify.config import SparseCoderConfig

# Simulate your layer
cfg = SparseCoderConfig(
    expansion_factor=64,  # Change to 4
    n_targets=4,
    transcode=True,
)

sae = SparseCoder(d_in=1280, cfg=cfg, device="cpu")

# Count parameters
total_params = sum(p.numel() for p in sae.parameters())
memory_mb = total_params * 4 / (1024**2)  # 4 bytes per fp32

print(f"SAE per layer: {memory_mb:.1f} MB")
print(f"33 layers: {memory_mb * 33 / 1024:.1f} GB")
print(f"With 2× for optimizer state: {memory_mb * 33 * 2 / 1024:.1f} GB")
