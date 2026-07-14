# Cross-Layer Transcoders for Protein Language Models
### Scaling Mechanistic Interpretability to ESM-2 650M (and beyond to ESM-C)

**Presenter:** Sagnik Chatterjee — SAGE Lab
**Audience:** Xiaocong Yang (UIUC)
**Date:** 2026-07-02

> A working-session walkthrough of what I'm building, why it matters for mechanistic
> interpretability, how the code actually works, and where the hard problems are.

---

## 0. The 60-second version

I train **Cross-Layer Transcoders (CLTs)** — a sparse dictionary-learning method from
Anthropic's *Crosscoders* line of work — on the internal activations of **ESM-2 650M**, a
protein language model. The goal is to decompose the model's computation into
**sparse, human-interpretable features** and then trace **circuits**: how early-layer
features (e.g. sequence-motif detectors) causally drive late-layer features (e.g. domain
/ structure predictors).

The codebase is a fork of EleutherAI's `sparsify`. My concrete contribution right now
(**Objective 1**) is making CLT training **fast and memory-tractable at the 650M scale**,
so the same recipe can climb to **ESM-C 3B–6B**. The current training is dominated by
recomputing the frozen ESM-2 forward pass every step and by an enormous CLT parameter
count. I'm profiling, adding activation caching, and shrinking the CLT via architectural
and LoRA-style factorizations.

---

## 1. Why mechanistic interpretability for proteins?

Mechanistic interpretability (MI) asks: *what algorithm is a neural network actually
running inside its weights?* In language models this program has matured — we can find
individual circuits (e.g. the IOI circuit in GPT-2, Wang et al. 2022) and decompose
representations into monosemantic features with sparse autoencoders (SAEs).

Protein language models (pLMs) are a **high-value, under-explored MI target**:

- ESM-2 (Lin et al., 2023, Science) predicts structure/function purely from sequence,
  and is the backbone of much of computational biology today.
- But *how* it computes contacts, folds, and functional sites is opaque.
- If we can extract **causal circuits**, we get scientific hypotheses about protein
  biology, not just predictions — e.g. "this catalytic-motif feature at layer 6 gates
  this domain-recognition feature at layer 20."

Two very recent results define the frontier I'm building on:

| Paper | What they showed | Model scale |
|---|---|---|
| **Nainani et al. 2025** (NeurIPS MI Workshop, UMass) | SAE latents in ESM-2 *causally mediate* contact prediction via a two-stage **motif → domain** circuit | ESM-2, per-layer SAEs |
| **ProtoMech** — Tsui, Talreja, Saeedi, Aghazadeh (ICML 2026, `2602.12026v2`) | **CLTs** on ESM2-8M/35M recover **82–89%** of original performance and find circuits using **<1%** of the latent space | ESM2-8M, ESM2-35M |

**The gap I'm filling:** ProtoMech works at 8M–35M. Nainani uses per-layer SAEs (which
miss cross-layer computation). Nobody has trained CLTs at **650M** — where representations
are rich enough to matter biologically but the training cost explodes. That's my project.

---

## 2. Background: SAE → Transcoder → Cross-Layer Transcoder

The three are a natural progression. All three learn an overcomplete, sparse dictionary,
but they differ in *what they read and what they predict*.

```
                 reads          predicts (reconstructs)
  SAE            act at L    →   same act at L            (autoencode one site)
  Transcoder     input of L  →   OUTPUT of L              (approximate a computation, e.g. an MLP)
  CLT (mine)     input of L  →   outputs of L, L+1, ..., L+N   (cross-layer computation)
```

**Why CLT and not SAE?** Real biological computation flows *across* layers. A per-layer SAE
sees a frozen snapshot at each depth and cannot represent "feature A at layer 6 writes into
feature B at layer 20." A CLT's encoder fires **once** at layer ℓ, and its decoder emits
reconstructions into **every downstream target layer** — so the learned dictionary *is* the
cross-layer wiring. This is exactly what circuit tracing needs.

### The k-sparse mechanism (shared by all three)

```
   x  ──►  W_enc · x + b   ──►  TopK (keep k largest, zero the rest)  ──►  z (sparse)
                                                                            │
   ŷ  ◄──  W_dec · z + b_dec  ◄───────────────────────────────────────────┘
```

- **TopK activation** (Gao et al. 2024) directly enforces sparsity — exactly `k` latents
  fire per token. No L1 penalty, no tuning a sparsity coefficient. This repo uses TopK by
  default (`config.py: activation="topk"`, also `batchtopk`, `groupmax`).
- **Expansion factor** sets dictionary size: `num_latents = expansion_factor × d_model`.
  For ESM-2 650M (`d_model = 1280`), `expansion_factor=128` → **163,840 latents**.

---

## 3. ESM-2 650M as the substrate

| Property | Value |
|---|---|
| Layers | 33 transformer blocks |
| `d_model` | 1280 |
| Attention heads | 20 |
| Max context | 1022 residues |
| Special token | CLS (no BOS) — handled specially in training |

A CLT sits at each layer's MLP, reading its input and reconstructing outputs across a
window of downstream layers.

```
  ESM-2 650M (FROZEN)                          CLTs (TRAINED)
  ┌────────────────────────────┐
  │  Embedding (residues)      │
  ├────────────────────────────┤
  │  Layer 0  attn + MLP  ──────┼──► CLT_0  ─┐ encode once at L0,
  ├────────────────────────────┤            │ decode into L0..L0+N
  │  Layer 1  attn + MLP  ──────┼──► CLT_1  ─┤
  ├────────────────────────────┤            │  (cross_layer = N)
  │        ...                 │            │
  ├────────────────────────────┤            │
  │  Layer 32 attn + MLP  ──────┼──► CLT_32 ─┘
  ├────────────────────────────┤
  │  LM / contact head         │
  └────────────────────────────┘
```

---

## 4. How the code works (the parts that matter)

The fork is small and legible. Four files carry the architecture:

| File | Role |
|---|---|
| `sparsify/sparse_coder.py` | `SparseCoder` (the CLT), `MidDecoder`, `ForwardOutput`. Encode/TopK/multi-target decode. |
| `sparsify/runner.py` | `CrossLayerRunner` — orchestrates cross-layer encode/decode + gradient detach/restore. |
| `sparsify/trainer.py` | `Trainer` — hooks ESM-2, runs the loop, computes FVU / KL / KL-FVU losses, checkpoints. |
| `sparsify/config.py` | `SparseCoderConfig` + `TrainConfig` — every knob. |

### 4.1 Data flow for one training step

```
 batch of protein sequences
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  ESM-2 650M forward pass  (FROZEN, recomputed every step)     │  ← bottleneck #1
 │  forward hooks fire at each hooked MLP module                 │
 └─────────────────────────────────────────────────────────────┘
        │  (x = MLP input, y = MLP output captured by the hook)
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  CrossLayerRunner.encode(x, sparse_coder)                     │
 │     z = TopK(W_enc·x + b)     → MidDecoder holds (x, z, idx)  │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  CrossLayerRunner.decode(...)                                 │
 │   for each still-open source layer:                          │
 │      ŷ_target += W_dec[target] · z_source                    │
 │   coalesce_topk ∈ {none, concat, per-layer, group}           │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  Loss:  FVU (reconstruction) | KL (faithfulness) | KL-FVU    │
 │  detach/restore backward across layers, optimizer step       │
 └─────────────────────────────────────────────────────────────┘
```

### 4.2 The cross-layer bookkeeping (`CrossLayerRunner`)

This is the cleverest part of the repo. Because one source layer writes into *many* target
layers, the runner keeps a dict of "open" `MidDecoder` objects (`self.outputs`). When layer
ℓ's output arrives, **every** source that still reaches ℓ contributes its decode. A source is
deleted once it has emitted its last target (`will_be_last`).

```python
# runner.py — the core loop, paraphrased
for i, (hookpoint, layer_mid) in enumerate(self.outputs.items()):
    if detach_grad:
        layer_mid.detach()                    # cut graph → per-layer backward
    candidate_indices.append(layer_mid.latent_indices + i * num_latents)
    candidate_values.append(layer_mid.current_latent_acts)
    out = layer_mid(y, ...)                    # decode this source into target y
    if hookpoint != module_name:
        output += out.sae_out                 # accumulate cross-layer contributions
```

**`coalesce_topk`** controls how latents from multiple sources combine at a target:
- `none` — each source decodes independently and sums (standard CLT).
- `concat` — tie all sources into one shared dictionary, TopK over the union.
- `per-layer` / `group` — variants for per-target tied decoders.

### 4.3 The detach/restore trick (memory management)

Naively, backprop through a 33-layer CLT would require holding the whole graph for all
layers at once. Instead (`sparse_coder.py: MidDecoder.detach/restore`):

1. During forward, latent activations are **detached** (`z.detach(); z.requires_grad=True`)
   so each layer's subgraph is independent.
2. `layer_mid(y).backward()` runs per-layer.
3. `runner.restore()` re-attaches the original activations and calls
   `original_activations.backward(grad)` to push gradients back into the encoder.

This keeps memory bounded — but it's a **custom backward**, which blocks `torch.compile`
on the full loop (a real cost, see §6).

### 4.4 Loss functions (`trainer.py`)

- **`fvu`** — Fraction of Variance Unexplained; pure reconstruction. One ESM-2 pass.
- **`kl`** — KL between ESM-2's original logits and logits when the CLT replaces the MLP;
  measures *faithfulness to the model's computation*. Needs a clean + a dirty pass.
- **`kl-fvu`** — the combined loss (arXiv:2503.17272) I care about most: reconstruct well
  *and* stay faithful. `kl_coeff=0.0` auto-scales KL to match FVU magnitude.

CLS-token handling: ESM-2 has no BOS, so the trainer masks the **CLS** token
(`trainer.py:~695`, `filter_bos=True`) to avoid contaminating position-specific losses.

### 4.5 Distributed / tensor parallelism

`--tp=N` sements each CLT across N GPUs with PyTorch **DTensor** (Shard/Replicate
placements + `.redistribute()` collectives). Necessary because a single CLT layer's weights
can exceed one GPU. The redistributions are a communication cost (bottleneck #3).

---

## 5. The core problem I own: **why is 650M CLT training slow?**

Reading the code carefully, the bottlenecks in impact order:

### Bottleneck 1 — No activation caching (biggest)
`Trainer.fit()` runs the **full frozen ESM-2 650M forward pass every step**, even though
its weights never change. For KL-FVU it's **two** passes/step. The base `sparsify` README
even lists "caching activations" as an unimplemented TODO. Every hyperparameter sweep
re-runs ESM-2 over the same sequences from scratch.

### Bottleneck 2 — Enormous CLT parameter count
At `expansion_factor=128`, `d_model=1280`, `cross_layer=12`:

```
  encoder / layer         : 163,840 × 1,280      ≈ 210M params
  decoders / layer        : 210M × 12 targets    ≈ 2.5B params
  per CLT layer           : ≈ 2.7B params
  all 33 layers           : ≈ 89B params   ← dwarfs ESM-2 itself (650M)
```

That's why real runs quietly reduce scale. For context, the same recipe on ProtoMech's
ESM2-35M (`d_model=480`) is ~**121× smaller** — the jump to 650M is what breaks it.

### Bottleneck 3 — TopK at scale, DTensor comms, custom backward
- TopK over 163,840 latents × ~32K tokens/batch = billions of comparisons/step.
- Every hooked layer triggers DTensor all-gather / reduce-scatter.
- The detach/restore custom backward blocks `torch.compile` fusion.

---

## 6. What I'm doing about it (roadmap)

Ordered by impact-to-effort. Full detail in `docs/objective1_writeup.md`.

1. **Profile first** (`torch.profiler`, `nsys`) — get ground-truth split of time between
   ESM-2 forward and CLT forward/backward before changing anything.
2. **Activation caching** — precompute & memory-map ESM-2 activations
   (`MemmapDataset` infra already exists). Expected **3–10× speedup**; sweeps become ~free.
   Storage is the catch — cache a layer subset or stream a ring buffer.
3. **Shrink the CLT** — start `expansion_factor=32`, `cross_layer=4–6`, `layer_stride=2`;
   reproduce ProtoMech-style quality at reduced scale before scaling up.
4. **LoRA-style factorization** (the direction Saishradha flagged):
   - Decoder: `W_dec_target = W_dec_shared + α·(A_target · B_target)` — shared base
     dictionary + small per-target low-rank adapters. Savings scale with `cross_layer`.
   - Encoder: product-key / low-rank factorization of the 210M-param `W_enc`.
5. **Loop-level wins** — `torch.compile` coverage, 8-bit Adam, activation checkpointing on
   the CLT, TP tuning / comms overlap.
6. **Sweeps** — `optimization/analyze_sweeps.py` already ranks W&B runs by
   `val_mean_fvu`, `val_replacement_ce_increase`, `val_mean_l0`, `dead_pct`.

**Targets after optimization:** `val_mean_fvu < 0.05`, replacement CE increase `< 0.1` nats,
`dead_pct < 5%`, `< 40 GB`/GPU, 3–10× throughput.

---

## 7. Then: biology (why any of this matters)

Once 650M CLTs train cleanly:

- **Reproduce & sharpen Nainani's motif→domain circuit** with CLTs (which model the actual
  cross-layer wiring, not per-layer snapshots) via activation patching.
- **Apply ProtoMech-style circuit discovery** at 650M — greedy latent selection + integrated
  gradients — and ask: *do larger models use the same circuits at higher resolution?*
- **ProteinGym** (NeurIPS 2023 benchmark, `papers/`) — test whether circuit-relevant CLT
  features alone predict mutation effects, validating the circuit's causal role.
- **Scale to ESM-C 3B–6B** with the whole optimized pipeline — the urgent target that makes
  the efficiency work worth doing now.

---

## 8. Discussion hooks for Xiaocong

Good places to go deep or get outside perspective:

- **Is KL-FVU the right objective for biological circuits**, vs. pure faithfulness (KL)?
  Faithfulness to ESM-2's computation may matter more than reconstruction for circuit claims.
- **How much cross-layer reach is enough?** Nainani's circuit spans layers ~4→24;
  does `cross_layer` need to be ≥16 to capture it, and what does that cost?
- **Factorized decoders vs. quality** — will shared-base + low-rank adapters preserve
  circuit fidelity, or wash out target-specific features?
- **Caching vs. streaming** at ESM-C scale, where full caches are tens of TB.
- **Evaluation** — beyond FVU, what's the right *biological* gold standard for "this circuit
  is real"? (contact AUROC under patching? ProteinGym Spearman?)

---

### Appendix — key references (in `papers/`)
- **Crosscoders**, Anthropic 2024 — CLT method origin (transformer-circuits.pub).
- **ProtoMech**, Tsui et al., ICML 2026 — `2602.12026v2.pdf`.
- **Nainani et al. 2025**, NeurIPS MI Workshop — `117_Mechanistic_evidence_that_.pdf`.
- **Sparse Feature Circuits**, Marks et al., ICLR 2025 — `2403.19647v3.pdf`.
- **IOI circuit**, Wang et al. 2022 — `2211.00593v1.pdf`.
- **ESM-2 / zero-shot mutation effects**, Meier/Rives et al. — NeurIPS 2021 paper.
- **ProteinGym**, NeurIPS 2023 — benchmark for fitness prediction.
- **Scaling & evaluating SAEs**, Gao et al. 2024 — TopK recipe (base `sparsify`).
</content>
</invoke>
