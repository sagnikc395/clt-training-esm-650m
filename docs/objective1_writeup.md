# Objective 1 Writeup: Optimizing CLT Training on ESM-2 650M

**Date**: 2026-06-29
**Author**: Sagnik Chatterjee

---

## 1. Project Context and Motivation

### 1.1 What We Are Doing

This project trains **Cross-Layer Transcoders (CLTs)** on protein language models (pLMs), specifically ESM-2 650M, with the ultimate goal of scaling to ESM-C 3B–6B. CLTs are a mechanistic interpretability tool introduced in Anthropic's Crosscoders (2024) framework. Unlike standard Sparse Autoencoders (SAEs), which treat each layer of a transformer independently, CLTs capture cross-layer computation: the encoder at layer ℓ fires once, and the decoder produces reconstructions for all downstream target layers ℓ+1, …, ℓ+N. This is critical for pLMs because real biological computation flows across many layers—motif detectors in early layers gate domain recognition in mid-to-late layers (Nainani et al., 2025), and these dependencies are entirely invisible to per-layer SAEs.

### 1.2 Why ESM-2 650M and Why Now

ESM-2 (Lin et al., 2023) is the most widely used protein language model for structure and function prediction. Its 650M parameter variant (33 transformer layers, d_model=1280) sits at a critical scale: large enough to encode rich biological representations, but tractable enough to train auxiliary models on. The sister lab paper (Nainani et al., 2025, NeurIPS 2025 Mechanistic Interpretability Workshop) has already established that SAE latents in ESM-2 causally mediate contact prediction through a two-stage motif→domain circuit. The ProtoMech paper (Tsui et al., ICML 2026) has trained CLTs on ESM2-8M and ESM2-35M and shown they recover 82–89% of original model performance on protein family classification and function prediction. **The gap we are filling is CLT training at the 650M scale and beyond.**

The recent release of ESM-C 3B–6B by Biohub makes the scaling question urgent. If we can identify the algorithmic bottlenecks that make 650M CLT training slow, we can address them systematically before attempting the 3B–6B models, where naive approaches will be completely infeasible.

### 1.3 What the Codebase Does

This repo is a fork of EleutherAI's `sparsify` library, extended with:
- **CLT support** via `CrossLayerRunner` and multi-target `SparseCoder`
- **Tensor Parallelism** via PyTorch DTensor across both encoder and decoder weight shards
- **bfloat16 training** with autocast
- **ESM-2 hookpoint support** (CLS token masking, `input_ids` column handling)
- **KL+FVU combined loss** (arxiv:2503.17272) for more faithful end-to-end optimization
- **Fused Triton kernels** for sparse encoder backward and COO decoder
- **Sweep infrastructure** (`optimization/analyze_sweeps.py`) that fetches W&B runs and ranks hyperparameter configurations by validation FVU, replacement CE increase, and mean L0

---

## 2. Understanding the Current Bottlenecks

Before proposing what to do next, we need to understand precisely why the existing CLT training setup is slow. From reading the codebase carefully, I have identified the following bottlenecks, ordered roughly by expected impact.

### 2.1 No Activation Caching (Primary Bottleneck)

**The single largest bottleneck is that activations are recomputed from scratch on every forward pass.** The `Trainer.fit()` loop computes `self.model(x)` (the full ESM-2 650M forward pass) for every training batch. ESM-2 650M has 33 transformer layers, each with self-attention (d_model=1280, 20 heads) plus a feed-forward network. Every training step for the CLT therefore pays the cost of a full ESM-2 650M forward pass, even though the ESM-2 weights are completely frozen.

For a dataset of 1M protein sequences with context length 1022 (ESM-2's maximum), computing activations on-the-fly means the same sequence tokens are run through ESM-2 repeatedly during sweeps with different CLT hyperparameters. The README's own TODO list acknowledges this: "Support for caching activations" is listed as a missing feature. The Anthropic and EleutherAI SAE training pipelines cache activations for this reason.

For the KL+FVU loss (which we need for faithful CLT training), the problem is even worse: **two forward passes** are required per batch — one clean pass to get reference logits, then one dirty pass with CLT hooks active.

**Concrete cost**: ESM-2 650M at bfloat16 processes roughly 1–2 tokens/ms per GPU. For a batch of 32 sequences × 1022 tokens = 32,704 tokens, one forward pass takes ~16–32ms per GPU before any CLT computation. Across 100K training steps, this is 1,600–3,200 GPU-seconds wasted on frozen model recomputation.

### 2.2 Massive CLT Parameter Count

With the default configuration (`expansion_factor=128`, `d_model=1280`, `cross_layer=12`):
- **Encoder weight**: 163,840 × 1,280 = ~210M parameters (per CLT layer)
- **Decoder weight per target**: 163,840 × 1,280 = ~210M parameters × 12 targets = ~2.5B parameters
- **Total per CLT layer**: ~2.7B parameters
- **Total across all 33 layers**: ~89B parameters

This is astronomically larger than ESM-2 650M itself. Even with tensor parallelism across 4 GPUs (tp=4), each GPU holds ~22B parameters of CLT weights—far exceeding GPU VRAM. The existing training likely trains only a subset of layers at a time, or uses a much smaller expansion_factor in practice.

For comparison, ProtoMech trained on ESM2-8M (d_model=320, 6 layers) and ESM2-35M (d_model=480, 12 layers) — where the same expansion_factor yields ~4M and ~23M CLT parameters respectively. The jump to 650M (d_model=1280) is an ~11x increase in model width, causing a ~121x increase in CLT parameter count.

### 2.3 TopK Operation at Scale

The encoder maps d_model=1280 inputs to num_latents=163,840 pre-activations, then selects the top-k (e.g., k=32). For a batch of 32,704 tokens, this requires finding the top-32 elements among 163,840 values for each of 32,704 rows. While the repo uses `rtopk` (randomized TopK) and custom Triton kernels for the backward pass, the forward TopK itself is still O(B × num_latents) per step, which at B=32K tokens and num_latents=164K is a 5.4B-element operation per step.

### 2.4 Cross-Layer Gradient Accumulation and the Detach/Restore Pattern

The `CrossLayerRunner` in `runner.py` implements a sophisticated gradient detach/restore mechanism: during training with FVU loss, it detaches the latent activations from the computation graph, then manually calls `.backward(grad)` on the original activations during `runner.restore()`. This is necessary to allow per-layer backward passes without retaining the full computation graph for all 33 layers simultaneously. However, this pattern prevents use of `torch.compile` on the entire training loop and creates serialized backward passes that cannot be fused.

### 2.5 DTensor Communication Overhead

The tensor parallelism implementation uses PyTorch DTensor with Shard(0) placements and explicit `.redistribute()` calls between Replicate and Shard layouts. Each `.redistribute()` call triggers an all-gather or reduce-scatter collective, which has high latency on NVLink and catastrophic latency on PCIe. Inspecting `trainer.py:538-543`, every batch triggers redistributions for both inputs and outputs at each hooked layer.

### 2.6 Data Loading and Preprocessing

The `DataLoader` in `trainer.py:422` uses `num_workers=4` and `pin_memory=True`, which is reasonable, but for ESM-2's expected input format (protein sequences up to 1022 residues), the tokenization and chunking step (`chunk_and_tokenize`) happens before training and the result is stored as a HuggingFace Dataset. Shuffling very large datasets in HF Arrow format can be a bottleneck.

---

## 3. Proposed Next Steps

### Step 1: Systematic Profiling (1–2 weeks)

Before making any architectural changes, we need data. We should profile a short training run (1000 steps) and measure time/memory breakdown.

**What to measure**:
- Time spent in ESM-2 forward pass vs. CLT forward/backward
- Memory occupied by: (a) frozen ESM-2 weights, (b) CLT weights per layer, (c) optimizer states
- GPU utilization throughout the training loop (using `nsys profile` or `torch.profiler`)
- Peak CUDA memory usage vs. allocated
- Number of steps per second for different layer counts (train on 1 layer vs. 5 vs. 10 vs. all 33)

**Concrete tool**: Use `torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)` wrapped around the training loop body for 10 steps, export to TensorBoard Chrome trace format.

**Expected finding**: Frozen model forward pass will dominate for small CLT configs; CLT backward will dominate for large expansion factors.

### Step 2: Activation Caching Pipeline (2–3 weeks)

This is the highest-leverage single optimization. We should pre-compute and save activations from all 33 ESM-2 650M layers for the entire training dataset.

**Implementation plan**:

```python
# Step 2a: Dump activations to disk
# For each sequence in the dataset, run ESM-2 once and save:
#   - Input to each MLP block (for transcoder training)
#   - Output of each MLP block (the transcoder target)
#   - Residual stream activations at each layer boundary
# Save as memory-mapped numpy arrays (.bin files) using MemmapDataset format
# that already exists in sparsify/data.py

# File layout:
# activations/
#   layer_00_in.bin    # shape (N_tokens, 1280)
#   layer_00_out.bin   # shape (N_tokens, 1280)
#   layer_01_in.bin
#   ...
#   layer_32_out.bin
```

**Storage cost**: ESM-2 650M has 33 layers, each hookpoint produces 1280-dim float32 tensors. For 1M sequences × 512 mean tokens (many proteins are shorter than 1022):
- Per-layer per-direction: 1M × 512 × 1280 × 2 bytes (bfloat16) ≈ 1.3 TB
- For all 33 layers × 2 directions (in + out): ~86 TB

This is too large for most setups. **Practical compromise**: Cache only the hookpoints we will actually train on (e.g., every 4th layer: layers 0, 4, 8, 12, 16, 20, 24, 28, 32 = 9 layers × 2 = ~24 TB), or use a smaller dataset (100K sequences ≈ 2.4 TB for 9 layers). For bfloat16, halve these estimates.

**Alternatively**: Implement streaming activation caching using a ring buffer that pre-computes N batches ahead. This avoids storing all activations but removes ESM-2 from the gradient path by running it in a background thread.

**Benefit**: Training speed can increase 3–10x depending on how much of training time is spent in ESM-2 forward passes. Hyperparameter sweeps become ~N_sweep_runs times cheaper because activations are computed only once.

### Step 3: Reduce CLT Parameter Count via Architectural Changes (3–4 weeks)

#### Step 3a: Layer-Stride and Selective Coverage

Rather than training CLTs on all 33 layers simultaneously, use `--layer_stride 2` or `--layer_stride 4` to train on every 2nd or 4th layer. This is already supported in the codebase via `TrainConfig.layer_stride`. ProtoMech found that 6-layer and 12-layer CLTs are sufficient for high-quality circuit discovery. Start with every 4th layer (9 CLTs), validate that circuits can still be recovered, then increase coverage.

#### Step 3b: Smaller Cross-Layer Reach

The current default `--cross_layer=12` means each CLT decoder predicts 12 layers ahead. For ESM-2 650M with 33 layers, this is ~36% of the network depth per CLT. The ProtoMech paper used all-preceding-layer CLTs (full reach), but found circuits using <1% of the latent space. We should start with `--cross_layer=4` or `--cross_layer=6` and measure whether circuit discovery quality degrades. This reduces the number of decoder matrices per CLT from 12 to 4–6, cutting decoder parameter count by 2–3x.

#### Step 3c: Reduced Expansion Factor at First Pass

Start with `--expansion_factor 32` (num_latents = 40,960) rather than 128. This reduces encoder/decoder parameter count by 16x. For ESM-2 650M, this gives a CLT encoder of 40,960 × 1,280 = 52M params — manageable. Use the sweep infrastructure to find the optimal k for this size (k is typically ~num_latents/1000 to /500, so k=40–80).

The ProtoMech paper used expansion_factor approximately matching the ratio in their ESM2-35M setup. Reconstruct their setup for 650M as a baseline before going larger.

#### Step 3d: LoRA-Inspired Encoder Factorization

This is the direction Saishradha explicitly mentioned. The full encoder weight W_enc ∈ R^{num_latents × d_model} is the largest single parameter tensor. A LoRA-style factorization writes:

```
W_enc = W_enc_pretrained + A @ B
```

where A ∈ R^{num_latents × r} and B ∈ R^{r × d_model} with r << d_model.

However, in the CLT context there is no "pretrained" W_enc to start from. A more natural approach is a **product-key memory** (PKM) encoder (Lample et al., 2019):

```
W_enc_factored:
  Split latent space into two groups of size sqrt(num_latents)
  Each group has its own sub-encoder: W_enc_A ∈ R^{sqrt(L) × d_model/2}
                                       W_enc_B ∈ R^{sqrt(L) × d_model/2}
  Pre-activations: preact_i,j = W_enc_A[i, :] · x[:d_model/2] + W_enc_B[j, :] · x[d_model/2:]
  The full latent (i,j) is activated by the outer product structure
```

For num_latents=40,960 = 202 × 202 ≈ 203^2, the sub-encoders have 203 × 640 = 130K parameters each, vs. 40,960 × 1,280 = 52M parameters for the full encoder. This is a **400x** parameter reduction for the encoder at the cost of some expressivity loss.

**Alternative LoRA application — Decoder**: For CLT decoders (where weights are randomly initialized from zero and then trained), we can parameterize each decoder as:

```python
W_dec_target = W_dec_shared + alpha * (A_target @ B_target)
```

where W_dec_shared ∈ R^{num_latents × d_model} is shared across all targets, and A_target, B_target are per-target low-rank adapters. This is exactly the LoRA spirit: all targets share a base decoder, with small per-target corrections. The memory savings scale with the number of targets (cross_layer value).

#### Step 3e: Layer-Shared Encoder across Consecutive Layers

If ESM-2 650M representations are geometrically similar across consecutive layers (as suggested by the smooth gradient of features across depth in the Nainani et al. paper), we can **share the encoder weights across consecutive layer groups**. For example, group layers 0–7, 8–15, 16–23, 24–32 and use one encoder per group. This reduces encoder parameters by 4–8x at the cost of assuming smoother cross-layer feature geometry.

### Step 4: Training Loop Optimizations (1–2 weeks)

#### Step 4a: torch.compile the Full CLT

Currently, `SparseCoder.encode()` has `@torch.compile(disable=NO_COMPILE)`, but the full training loop has a custom backward (FusedEncoder) that may prevent compilation. Audit which parts of the training loop can be compiled:
- The frozen ESM-2 forward: already compiled in `trainer.py:383`
- The CLT encoder forward: already compiled
- The CLT decoder forward: needs to be included

For the CLT backward, the `FusedEncoder` custom autograd function (`fused_encoder.py:72`) prevents `torch.compile` from tracing through it. Consider whether the custom backward is actually necessary at the 650M scale or if the standard autograd graph is equally fast (PyTorch 2.x has improved sparse gradient support).

#### Step 4b: Gradient Accumulation and Mixed Precision

Currently using bfloat16 autocast for encoder/decoder (enabled in `SparseCoder.forward()` and `MidDecoder.__call__()`). Ensure the optimizer (Adam with `b1=0.0, b2=0.999`) is running in float32 for stability. Consider `--optimizer adam8` (8-bit Adam from torchao) which can reduce optimizer state memory from 16 bytes/param to 6 bytes/param.

For gradient accumulation, increase `--grad_acc_steps` to allow larger effective batch sizes without increasing per-GPU memory. For ESM-2 proteins (which are short on average), effective batch sizes of 256–512 sequences are achievable.

#### Step 4c: Activation Checkpointing for the CLT Itself

For training CLTs with large num_latents and cross_layer values, the intermediate activations (latent_acts, latent_indices, sae_out tensors) accumulate across all 33 layers during the forward pass. Implement gradient checkpointing (torch.utils.checkpoint) for the CLT forward steps so that only a fraction of these are stored. Since the CLT forward is cheap relative to ESM-2, recomputing it during backward is affordable.

#### Step 4d: Tensor Parallelism Tuning

Profile the DTensor all-gather/reduce-scatter calls using `torch.profiler`. If communication is a bottleneck, consider:
- Using NCCL's overlapping features: overlap AllReduce with the next layer's forward pass
- Using `--tp 1` (no tensor parallelism) with `--distribute_modules True` (layer-parallel) when the encoder fits on a single GPU
- Reducing `tp` and increasing gradient accumulation steps instead

### Step 5: Hyperparameter Sweep with Validated Infrastructure (1–2 weeks)

The `optimization/analyze_sweeps.py` script is already set up to fetch W&B runs and rank them. The primary ranking metric is `val_mean_fvu` (validation fraction of variance unexplained, lower is better), followed by `val_replacement_ce_increase` (how much CE loss increases when CLT replaces ESM-2, lower is better), and `val_mean_l0` (average number of active latents, lower is better for interpretability).

**Recommended sweep grid** (for ESM-2 650M at reduced scale):

| Parameter | Values to Sweep |
|-----------|----------------|
| `expansion_factor` | 16, 32, 64 |
| `k` | 16, 32, 64 |
| `lr` | 1e-4, 2e-4, 5e-4 |
| `optimizer` | adam, signum |
| `cross_layer` | 4, 8, 12 |
| `coalesce_topk` | none, concat |

Run on a subset of layers first (e.g., `--layers 8 16 24`) to identify the best hyperparameters before committing to full 33-layer training. Use Optuna TPE sampling (the sweep infrastructure already supports `optuna_` prefix runs) rather than grid search to find good configs faster.

**Key metrics to track beyond FVU**:
- `dead_pct`: Percentage of dead latents (features that never fire). High dead_pct means the CLT is using latent capacity poorly and suggests the encoder bias initialization needs improvement.
- Training throughput (tokens/sec): Directly measures optimization success.
- Memory per GPU: Must fit within available VRAM for the target cluster.
- `val_replacement_ce_increase`: This is the biological gold standard — how much worse is ESM-2 when its MLP layers are replaced by CLT reconstructions.

### Step 6: ESM-2 650M — Full Coverage with Optimized Setup (3–4 weeks)

Once Steps 1–5 are complete, run a full 33-layer CLT training on ESM-2 650M using:
- Activation caching from Step 2 (or streaming caching if storage is unavailable)
- Optimized hyperparameters from Step 5
- Reduced expansion_factor initially (32 or 64)
- Distributed training with `torchrun` across all available GPUs

**Target metrics for 650M CLT**:
- `val_mean_fvu` < 0.05 (less than 5% variance unexplained, matching ProtoMech's small-model results)
- `val_replacement_ce_increase` < 0.1 nats
- `dead_pct` < 5%
- Coverage of all 33 layers with `cross_layer` >= 4

### Step 7: Biological Validation (2–3 weeks)

Once trained CLTs are available:

#### Step 7a: Reproduce Nainani et al. Circuit on 650M CLT

Nainani et al. (2025) found a specific circuit in ESM-2 where early SAE latents (layers 4–8) detect short sequence motifs that gate mid-to-late domain recognition latents (layers 12–24) for contact prediction. The same circuit should be discoverable with higher fidelity using CLTs (which model the actual cross-layer computation rather than independent per-layer representations).

Use activation patching (as in Nainani et al.): for proteins where contact recovery jumps with additional unmasked residues (MetXA: P45131, TOP2: P06786), patch CLT latent activations between the corrupted and clean sequences and measure which latent-token pairs are causally necessary.

#### Step 7b: Reproduce ProtoMech Circuit Discovery on 650M

The ProtoMech paper (Tsui et al.) performs circuit discovery using a greedy latent selection algorithm. Apply the same algorithm to our 650M CLT:
1. Train a linear probe on the final CLT output to predict protein family (Pfam labels)
2. Use integrated gradients over CLT latent activations to attribute probe output to latents
3. Identify the minimal subset of latents that retains >=79% of probe performance while using <1% of the latent space
4. Visualize top-activating sequences for each circuit latent using Swiss-Prot annotations

Compare circuit latents to those found by ProtoMech on small models — do larger models use the same circuit structure at higher resolution?

#### Step 7c: ProteinGym Fitness Prediction

Use the trained 650M CLT to predict mutation effects on ProteinGym benchmarks (NeurIPS 2023). CLT latent activations provide a more compressed and interpretable representation than raw ESM-2 residual stream states. Evaluate whether CLT features from the circuit-relevant layers (motif detectors, domain detectors) are sufficient to achieve near-ESM-2 performance on fitness prediction — this would validate the circuit's causal role in ESM-2's function.

### Step 8: Scaling to ESM-C 3B–6B (Timeline TBD)

Once the full 650M pipeline is validated, apply the same optimizations to ESM-C 3B–6B. Key differences:
- ESM-C 3B: d_model ≈ 2048–2560, ~36 layers
- ESM-C 6B: d_model ≈ 3072–4096, ~48 layers
- Requires 8+ A100 80GB GPUs minimum for model + CLT weights
- Activation caching is even more critical at this scale (model forward pass is ~4–9x slower than 650M)
- May require `--expansion_factor 16` or `--expansion_factor 8` initially

The tensor parallelism infrastructure already in the codebase (DTensor with DP×TP device mesh) is designed to handle this — the `--tp` flag controls tensor parallelism degree. At ESM-C 6B with expansion_factor=32, num_latents ≈ 100K–130K, requiring tp=8 to fit CLT weights across 8 GPUs.

---

## 4. Evaluation Framework

### 4.1 Training Quality Metrics

| Metric | Symbol | Target | Meaning |
|--------|--------|--------|---------|
| Validation FVU | `val_mean_fvu` | < 0.05 | Fraction of variance unexplained across all layers |
| Replacement CE increase | `val_replacement_ce_increase` | < 0.1 | CE loss increase when CLT replaces ESM-2 |
| Mean L0 | `val_mean_l0` | ≈ k | Average active features per token |
| Dead latent % | `dead_pct` | < 5% | % of features never activating |
| Density mean | `mean_density_mean_ema` | 0.001–0.01 | Feature usage density |

### 4.2 Biological Validation Metrics

| Metric | Meaning |
|--------|---------|
| Protein family F1 (ProtoMech protocol) | Classification accuracy with CLT as replacement model |
| Function prediction Spearman R | Correlation on GO function labels |
| Contact prediction AUROC (Nainani protocol) | Accuracy in causal circuit analysis |
| ProteinGym Spearman R | Mutation effect prediction |
| Circuit compression ratio | % of latent space needed to retain 80% of task performance |

### 4.3 Efficiency Metrics

| Metric | Baseline | Target after optimization |
|--------|----------|--------------------------|
| Training throughput (tokens/sec) | TBD (profile first) | 3–10x improvement |
| Time to 5% FVU | TBD | < 12 GPU-hours |
| Peak GPU memory | TBD | < 40GB per GPU |
| Steps per second | TBD | 2–5x improvement |

---

## 5. Summary of Priorities

Listed in order of expected impact-to-effort ratio:

1. **Profile the current training run** to get ground-truth bottleneck data before building anything. (1–3 days)

2. **Implement activation caching** for the frozen ESM-2 650M forward pass. This is the single highest-leverage change and the existing `MemmapDataset` class already provides the infrastructure. (1–2 weeks)

3. **Reduce CLT parameter count** by starting with `expansion_factor=32`, `cross_layer=4`, and `layer_stride=2`. Run a validated baseline with these settings before increasing scale. (3–5 days)

4. **Run a hyperparameter sweep** using the existing `analyze_sweeps.py` infrastructure on the reduced-scale setup. Identify the optimal (k, lr, optimizer, coalesce_topk) combination. (1–2 weeks)

5. **Explore LoRA-inspired factorizations** for the decoder — specifically, shared decoder + per-target low-rank adapter. Benchmark whether it degrades val_mean_fvu or replacement CE compared to full-rank decoders. (2–3 weeks)

6. **Full 33-layer ESM-2 650M CLT training** once the above are in place. (2–4 weeks)

7. **Biological circuit validation** replicating and extending Nainani et al. and ProtoMech at the 650M scale. (2–3 weeks)

8. **Scale to ESM-C 3B–6B** with the full optimized pipeline. (Timeline: post-650M validation)

---

## 6. Open Questions

- **What is the right loss function for biological tasks?** The current options are FVU (pure reconstruction), KL (model faithfulness), and KL+FVU (combined). For circuit discovery in pLMs, faithfulness to ESM-2's internal computation matters more than token prediction CE. KL+FVU is likely the right choice but needs the two-pass overhead.

- **How much does cross_layer reach matter?** ProtoMech uses full-preceding-layer CLTs (every previous layer contributes). The current codebase's `cross_layer=12` is a compromise. Since ESM-2 circuits span 8–16 layers according to Nainani et al. (motif layers 4–8, domain layers 12–24), we probably need at least `cross_layer=16` for full circuit capture at the 650M scale.

- **Should we train per-MLP or per-residual-stream?** Nainani et al. and ProtoMech both use residual stream SAEs (or equivalently, treat the full layer output as the hookpoint). The current code's `--hookpoints` flag supports MLP-specific hookpoints. CLTs trained on MLP inputs/outputs (as transcoders) are more mechanistically interpretable than residual stream SAEs, but require `--transcode=True` and careful choice of hookpoints.

- **Does the `coalesce_topk` strategy matter for biological tasks?** The `concat` mode ties all decoders to the same latent dictionary, while `per-layer` uses separate dictionaries per target layer. For protein circuits where the same feature (e.g., HRD catalytic motif) needs to be readable at multiple downstream layers, `concat` may be more interpretable. This should be explored in the hyperparameter sweep.

- **How to handle ESM-2's CLS token?** ESM-2 uses a CLS token for global sequence representations, and the codebase already handles this in `trainer.py:694-698` with the `bos_mask`. However, for contact prediction (which uses position-specific outputs), the CLS token should probably be masked during CLT training with the `--filter_bos True` flag.
