# Onboarding: Cross-Layer Transcoders on ESM-2 650M

**Project:** SAGE Lab — mechanistic interpretability for protein language models
**Maintainers:** Sagnik Chatterjee; Saishradha Mohanty (PhD, Anna Green Lab, UMass Amherst)
**Last updated:** 2026-07-14

Welcome. This document is the single starting point for a new researcher joining the project. It explains *what* we are building, *why*, *how the code is laid out*, *what the current state and bottlenecks are*, and — the part you'll spend most of your time on — *how to make CLT training fast and tractable at the 650M scale*. Read sections 1–5 to get oriented, then live in sections 6–8 once you start doing the work.

---

## 1. The one-paragraph version

We train **Cross-Layer Transcoders (CLTs)** — a sparse dictionary-learning method from Anthropic's *Crosscoders* line — on the frozen internal activations of **ESM-2 650M**, a protein language model. The point is to decompose ESM-2's computation into **sparse, human-interpretable features** and then trace **circuits**: how early-layer features (e.g. sequence-motif detectors) causally drive late-layer features (e.g. domain/structure predictors). The codebase is a fork of EleutherAI's `sparsify`. The current concrete objective (**Objective 1**) is making CLT training **fast and memory-tractable at 650M**, so the same recipe can climb to **ESM-C 3B–6B**. Training is today dominated by (a) recomputing the frozen ESM-2 forward pass every step and (b) an enormous CLT parameter count.

There is also a parallel research track flagged by Saishradha (see the June 30 sync in §9): before/alongside the optimization work, **apply the existing CLTs to the two proteins from the ProtoMech mutation-effect-prediction task and check whether our 650M CLTs recover the same (or richer) circuits**. That is the cheapest way to show scientific progress and is a good first task.

---

## 2. Why mechanistic interpretability for proteins

Mechanistic interpretability (MI) asks: *what algorithm is a network actually running inside its weights?* In language models this is a maturing program — we can isolate circuits (e.g. the IOI circuit in GPT-2) and decompose representations into monosemantic features via sparse autoencoders (SAEs).

Protein language models (pLMs) are a high-value, under-explored MI target:

- **ESM-2** (Lin et al., 2023, *Science*) predicts structure and function purely from sequence and underpins much of modern computational biology.
- *How* it computes contacts, folds, and functional sites is opaque.
- Extracting **causal circuits** turns predictions into scientific hypotheses about protein biology — e.g. "this catalytic-motif feature at layer 6 gates this domain-recognition feature at layer 20."

Two recent results define the frontier we build on:

| Paper | What they showed | Model scale |
|---|---|---|
| **Nainani et al. 2025** (NeurIPS MI Workshop, UMass) — `papers/117_Mechanistic_evidence_that_.pdf` | SAE latents in ESM-2 *causally mediate* contact prediction via a two-stage **motif → domain** circuit | ESM-2, per-layer SAEs |
| **ProtoMech** — Tsui, Talreja, Saeedi, Aghazadeh (ICML 2026) — `papers/2602.12026v2.pdf` | **CLTs** on ESM2-8M/35M recover **82–89%** of original performance and find circuits using **<1%** of the latent space | ESM2-8M, ESM2-35M |

**The gap we fill:** ProtoMech works at 8M–35M. Nainani uses per-layer SAEs (which miss cross-layer computation). Nobody has trained CLTs at **650M**, where representations are rich enough to matter biologically but training cost explodes.

---

## 3. Background: SAE → Transcoder → Cross-Layer Transcoder

All three learn an overcomplete, sparse dictionary. They differ in *what they read and what they predict*:

```
                 reads          predicts (reconstructs)
  SAE            act at L    →   same act at L                 (autoencode one site)
  Transcoder     input of L  →   OUTPUT of L                   (approximate a computation, e.g. an MLP)
  CLT (ours)     input of L  →   outputs of L, L+1, ..., L+N   (cross-layer computation)
```

**Why CLT and not SAE?** Real biological computation flows *across* layers. A per-layer SAE sees a frozen snapshot at each depth and cannot represent "feature A at layer 6 writes into feature B at layer 20." A CLT's encoder fires **once** at layer ℓ, and its decoder emits reconstructions into **every downstream target layer** — so the learned dictionary *is* the cross-layer wiring. That is exactly what circuit tracing needs.

The k-sparse mechanism shared by all three:

```
   x  ──►  W_enc·x + b   ──►  TopK (keep k largest, zero rest)  ──►  z (sparse)
                                                                       │
   ŷ  ◄──  W_dec·z + b_dec  ◄──────────────────────────────────────────┘
```

- **TopK activation** (Gao et al. 2024) directly enforces sparsity — exactly `k` latents fire per token. No L1 penalty, no sparsity coefficient to tune. Default in this repo (`config.py`: `activation="topk"`; also `batchtopk`, `groupmax`).
- **Expansion factor** sets dictionary size: `num_latents = expansion_factor × d_model`. For ESM-2 650M (`d_model=1280`), `expansion_factor=128` → **163,840 latents**.

---

## 4. ESM-2 650M as the substrate

| Property | Value |
|---|---|
| Layers | 33 transformer blocks |
| `d_model` | 1280 |
| Attention heads | 20 |
| Max context | 1022 residues |
| Special token | CLS (no BOS) — masked specially in training |

A CLT sits at each layer's MLP, reading its input and reconstructing outputs across a window of downstream layers (`cross_layer = N`).

---

## 5. The codebase

Fork of EleutherAI's `sparsify`. Four files carry the architecture — read them in this order.

| File | Role |
|---|---|
| `sparsify/config.py` | `SparseCoderConfig` (aka `SaeConfig`) + `TrainConfig` — every knob. Start here. |
| `sparsify/sparse_coder.py` | `SparseCoder` (the CLT), `MidDecoder`, `ForwardOutput`. Encode / TopK / multi-target decode. |
| `sparsify/runner.py` | `CrossLayerRunner` — orchestrates cross-layer encode/decode + the gradient detach/restore trick. |
| `sparsify/trainer.py` | `Trainer` — hooks ESM-2, runs the loop, computes FVU / KL / KL-FVU losses, checkpoints, DTensor TP. |

Supporting files: `fused_encoder.py` (fused Triton kernels for the sparse encoder backward), `kernels.py`, `data.py` (`chunk_and_tokenize`, `MemmapDataset` — relevant to activation caching), `__main__.py` (CLI entry point), and the optimizers `muon.py` / `sign_sgd.py`.

**What the fork adds over base `sparsify`:** CLT support (`CrossLayerRunner`, multi-target `SparseCoder`); tensor parallelism via DTensor; bfloat16 training; input hookpoints before layernorm; KL+FVU combined loss (arXiv:2503.17272); ESM-2 support (CLS masking, `input_ids` handling).

### 5.1 One training step (data flow)

```
 batch of protein sequences
        │
        ▼
 ESM-2 650M forward pass  (FROZEN, recomputed every step)   ← bottleneck #1
 forward hooks fire at each hooked MLP module
        │  (x = MLP input, y = MLP output captured by the hook)
        ▼
 CrossLayerRunner.encode(x):  z = TopK(W_enc·x + b) → MidDecoder holds (x, z, idx)
        │
        ▼
 CrossLayerRunner.decode():   for each open source layer, ŷ_target += W_dec[target]·z_source
        │                     coalesce_topk ∈ {none, concat, per-layer, group}
        ▼
 Loss: FVU (reconstruction) | KL (faithfulness) | KL-FVU (combined)
 detach/restore backward across layers → optimizer step
```

### 5.2 Two mechanisms worth understanding before you touch the loop

- **Cross-layer bookkeeping (`CrossLayerRunner`).** One source layer writes into *many* targets. The runner keeps a dict of "open" `MidDecoder` objects; when layer ℓ's output arrives, every still-open source that reaches ℓ contributes its decode. A source is deleted once it emits its last target (`will_be_last`). `coalesce_topk` controls how latents from multiple sources combine at a target: `none` (sum independent decodes — standard CLT), `concat` (tie all sources into one shared dictionary, TopK over the union), `per-layer`/`group` (per-target tied variants).

- **Detach/restore backward (memory management).** Backprop through a 33-layer CLT would otherwise hold the whole graph at once. Instead: during forward, latent activations are detached (`z.detach(); z.requires_grad=True`) so each layer's subgraph is independent; `layer_mid(y).backward()` runs per-layer; `runner.restore()` re-attaches originals and calls `original_activations.backward(grad)` to push gradients into the encoder. This bounds memory **but is a custom backward, which blocks `torch.compile` on the full loop** (a real cost — see §7).

### 5.3 Loss functions (`trainer.py`, `--loss_fn`)

- `fvu` — Fraction of Variance Unexplained; pure reconstruction. **One** ESM-2 pass.
- `kl` — KL between ESM-2's original logits and logits when the CLT replaces the MLP; measures faithfulness to the model's computation. **Two** passes (clean + dirty).
- `kl-fvu` — combined (arXiv:2503.17272), reconstruct well *and* stay faithful; `kl_coeff=0.0` auto-scales KL to FVU magnitude. This is the objective we care about most.
- `ce` — end-to-end cross-entropy on final logits.

CLS handling: ESM-2 has no BOS, so the trainer masks the **CLS** token (`--filter_bos=True`) to avoid contaminating position-specific losses.

### 5.4 Key config knobs (from `config.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--activation` | `topk` | `topk`, `batchtopk`, `groupmax` |
| `--expansion_factor` | 32 | dictionary size = EF × d_model |
| `--k` | — | latents kept per token |
| `--transcode` | False | transcoder mode (input→output) vs SAE |
| `--skip_connection` | False | linear skip around the transcoder |
| `--cross_layer` | 0 | number of downstream targets (N) |
| `--coalesce_topk` | none | `none`/`concat`/`per-layer`/`group` |
| `--topk_coalesced` | False | apply TopK to coalesced values |
| `--loss_fn` | fvu | `ce`/`fvu`/`kl`/`kl-fvu` |
| `--kl_coeff` | 1.0 | 0.0 = auto-scale KL to FVU |
| `--optimizer` | signum | `adam`/`adam8`/`muon`/`signum` |
| `--layer_stride` | 1 | train every Nth layer (mutually exclusive with `--layers`) |
| `--tp` | — | tensor-parallel degree |

### 5.5 Ready-made scripts

Training scripts at the repo root (`train_clt.sh`, `train_clt_cl16.sh`, `train_clt_cl16_4gpu.sh`, `train_clt_allforward_8gpu.sh`, `train_clt_distributed.sh`, `train_plt_{2,4}gpu.sh`) and in `scripts/training/`. Sweep analysis lives in `optimization/analyze_sweeps.py` (fetches W&B runs, ranks by `val_mean_fvu`, then `val_replacement_ce_increase`, then `val_mean_l0`). Weight/circuit analysis helpers in `scripts/` (`weight_analysis.py`, `circuit_plots.py`).

---

## 6. Current state and the bottlenecks

Basic SAE training works on ESM-2 650M. CLT training runs but is **slow** and **memory-heavy**. In impact order:

### Bottleneck 1 — No activation caching (biggest)

`Trainer.fit()` runs the **full frozen ESM-2 650M forward pass every step**, even though its weights never change. For KL-FVU it's **two** passes/step. The base `sparsify` README itself lists "caching activations" as an unimplemented TODO. Every hyperparameter sweep re-runs ESM-2 over the same sequences from scratch.

Rough cost: ESM-2 650M at bf16 processes ~1–2 tokens/ms/GPU. A batch of 32 seqs × 1022 tokens ≈ 32.7K tokens → ~16–32 ms/GPU *before any CLT compute*. Over 100K steps that's ~1,600–3,200 GPU-seconds spent purely recomputing a frozen model.

### Bottleneck 2 — Enormous CLT parameter count

At `expansion_factor=128`, `d_model=1280`, `cross_layer=12`:

```
  encoder / layer     : 163,840 × 1,280            ≈ 210M params
  decoders / layer    : 210M × 12 targets          ≈ 2.5B params
  per CLT layer       :                            ≈ 2.7B params
  all 33 layers       :                            ≈ 89B params   ← dwarfs ESM-2 (650M) itself
```

That is why real runs quietly reduce scale. The same recipe on ProtoMech's ESM2-35M (`d_model=480`) is ~121× smaller — the jump to 650M is what breaks it.

### Bottleneck 3 — TopK at scale, DTensor comms, custom backward

- TopK over 163,840 latents × ~32K tokens/batch = billions of comparisons/step.
- Every hooked layer triggers DTensor all-gather / reduce-scatter (`.redistribute()`), high latency on PCIe.
- The detach/restore custom backward blocks `torch.compile` fusion.

Secondary: HF Arrow dataset shuffling can bottleneck data loading at very large scale.

---

## 7. How to optimize the CLT runs (the core of Objective 1)

Ordered by impact-to-effort. The exhaustive version with code sketches and storage math is in `docs/objective1_writeup.md` — this is the operational summary.

### Step 0 — Profile before changing anything (1–3 days)

Get ground-truth numbers first. Wrap ~10 steps of the loop in `torch.profiler.profile(activities=[CPU, CUDA], with_stack=True)`, export a Chrome trace, and also try `nsys profile`. Measure:

- Time in ESM-2 forward vs. CLT forward vs. CLT backward.
- Memory split: frozen ESM-2 weights / CLT weights per layer / optimizer states.
- GPU utilization and peak vs. allocated CUDA memory.
- Steps/sec as a function of layers trained (1 vs 5 vs 10 vs 33).

Expected: frozen forward dominates for small CLTs; CLT backward dominates for large expansion factors. **Don't build anything until this tells you which one you're in.**

### Step 1 — Activation caching (highest leverage; 1–2 weeks)

Precompute and memory-map ESM-2 activations so the frozen forward isn't recomputed each step. The `MemmapDataset` infrastructure already exists in `sparsify/data.py`. Dump, per sequence: MLP input (transcoder input), MLP output (transcoder target), and residual-stream activations at layer boundaries. Expected **3–10× speedup**; sweeps become nearly free because activations are computed once.

The catch is storage. Full caches are huge (all 33 layers × both directions × 1M seqs ≈ tens of TB even in bf16). Two practical routes:
- **Cache a layer subset** (e.g. every 4th layer = 9 layers) and/or a smaller dataset (100K seqs).
- **Streaming ring buffer**: pre-compute N batches ahead in a background thread so ESM-2 leaves the gradient path without materializing the full cache to disk. Prefer this at ESM-C scale.

### Step 2 — Shrink the CLT (3–5 days)

Reproduce ProtoMech-style quality at reduced scale *before* scaling up:
- `--expansion_factor 32` (num_latents 40,960; ~16× fewer params than EF=128).
- `--cross_layer 4` to `6` (fewer decoder matrices; ProtoMech found small reach sufficient for circuit discovery — but note Nainani's circuit spans layers ~4→24, so you may need ≥16 for *full* circuit capture; test empirically).
- `--layer_stride 2` (train every 2nd layer; already supported).
- Sweep `k` for the new size (typically num_latents/1000 to /500, so k≈40–80 here).

### Step 3 — LoRA-style factorization (the direction Saishradha flagged; 2–3 weeks)

The encoder `W_enc` (~210M params) and per-target decoders are where the parameters live.
- **Decoder:** `W_dec_target = W_dec_shared + α·(A_target · B_target)` — a shared base dictionary plus small per-target low-rank adapters. Memory savings scale with `cross_layer`. Most promising, cleanest to try first.
- **Encoder:** low-rank or **product-key memory (PKM)** factorization (Lample et al. 2019). Split the latent space into two √num_latents groups with sub-encoders; the full latent (i,j) is activated by an outer-product structure. Potentially ~100–400× fewer encoder params at some expressivity cost.

Benchmark each against full-rank on `val_mean_fvu` and `val_replacement_ce_increase` — the open question is whether factorization washes out target-specific features that matter for circuit fidelity.

### Step 4 — Training-loop wins (1–2 weeks)

- **`torch.compile` coverage.** Encoder/decoder forwards are already compiled; the custom `FusedEncoder` backward blocks full-loop compilation. Audit whether the custom backward is actually faster than plain autograd at 650M scale (PyTorch 2.x sparse-grad support has improved) — if not, dropping it unlocks compilation.
- **8-bit Adam** (`--optimizer adam8`) cuts optimizer state from ~16 to ~6 bytes/param. Keep the master optimizer math in fp32 for stability.
- **Activation checkpointing on the CLT forward** (`torch.utils.checkpoint`) — the CLT forward is cheap relative to ESM-2, so recomputing it in backward is a good memory trade.
- **TP tuning:** profile the DTensor collectives; overlap NCCL AllReduce with the next layer's forward; consider `--tp 1` + `--distribute_modules` (layer-parallel) when a single CLT layer fits on one GPU; or reduce `tp` and raise `--grad_acc_steps`.

### Step 5 — Sweep with the existing infra (1–2 weeks)

`optimization/analyze_sweeps.py` already ranks W&B runs. Recommended grid on a *subset* of layers (`--layers 8 16 24`) before committing to 33 layers:

| Parameter | Values |
|---|---|
| `expansion_factor` | 16, 32, 64 |
| `k` | 16, 32, 64 |
| `lr` | 1e-4, 2e-4, 5e-4 |
| `optimizer` | adam, signum |
| `cross_layer` | 4, 8, 12 |
| `coalesce_topk` | none, concat |

Track beyond FVU: `dead_pct` (< 5%), tokens/sec, memory/GPU, and `val_replacement_ce_increase` (the model-faithfulness gold standard). Prefer Optuna TPE (`optuna_` runs are supported) over grid search.

### Targets after optimization

`val_mean_fvu < 0.05` · replacement CE increase `< 0.1` nats · `dead_pct < 5%` · `< 40 GB`/GPU · **3–10× throughput**.

### On learning the low-level tooling

Per the June 30 sync (§9): learning **Triton kernels / FlashAttention-style optimization** is valuable long-term (bigger models are expensive to run), but it is *not* a prerequisite to make progress now. Saishradha's explicit advice was to **not get stuck in "panic mode" reading kernel docs** — the config-level and caching/factorization wins above deliver most of the speedup without hand-writing kernels. Treat kernel authoring as an optional later deep-dive, and lead with the applied CLT circuit work.

---

## 8. The parallel research track: apply CLTs to circuits (good first task)

This is the cheapest way to show scientific results and was Saishradha's recommended first step. Once you have any trained 650M CLT:

1. **Reproduce ProtoMech's mutation-effect-prediction setup.** ProtoMech does supervised mutation-effect prediction for **two proteins**; find them in `papers/2602.12026v2.pdf`. Run our CLTs on those two proteins and check whether they recover **the same circuits** — the first sanity check. Hypothesis: they should, and being a 650M model (vs their 35M), ours may surface **richer / more features**.
2. **Then move toward the unsupervised task.** Saishradha is writing a paper on an *unsupervised* mutation-effect-prediction task; the medium-term goal is to translate our CLTs to that setting (needs a slight change from the supervised setup).
3. **Reproduce & sharpen Nainani's motif→domain circuit** using activation patching — CLTs model the actual cross-layer wiring rather than per-layer snapshots, so the circuit should appear at higher fidelity. Suggested proteins from the writeup: MetXA (P45131), TOP2 (P06786).
4. **ProteinGym** (`papers/NeurIPS-2023-proteingym-...pdf`) — test whether circuit-relevant CLT features alone predict mutation effects, validating the circuit's causal role.

Cadence: weekly update to Saishradha, **Tuesdays** (per the June 30 sync).

---

## 9. Meeting / decision log

- **2026-06-30 sync (Saishradha), `docs/quick-note-5-02-pm.md`.** First step: apply existing CLTs to the ProtoMech two-protein mutation-effect task and check circuit recovery (supervised sanity check first, unsupervised later). Optimization/kernels are useful but not blocking — don't get stuck; prioritize showing applied results. Weekly updates on Tuesdays.
- **2026-07-02 walkthrough (Xiaocong Yang, UIUC), `docs/presentation_xiaocong.md`.** Full technical walkthrough of the method, code, and bottlenecks; also the source of the slide deck (`CLT_ESM650M_deck.{pptx,pdf}`, `slides_clt_esm650m.html`). Good discussion hooks: is KL-FVU the right objective for biological circuits vs pure KL; how much cross-layer reach is enough; factorized decoders vs circuit fidelity; caching vs streaming at ESM-C scale; the right *biological* gold standard for "this circuit is real."

---

## 10. Reference library (`papers/`)

| File | Reference |
|---|---|
| *(transformer-circuits.pub, 2024)* | **Crosscoders**, Anthropic — the CLT method origin |
| `2602.12026v2.pdf` | **ProtoMech**, Tsui et al., ICML 2026 — CLTs on ESM2-8M/35M; our direct predecessor |
| `117_Mechanistic_evidence_that_.pdf` | **Nainani et al. 2025**, NeurIPS MI Workshop (UMass) — SAE motif→domain circuit for contacts |
| `2403.19647v3.pdf` | **Sparse Feature Circuits**, Marks et al., ICLR 2025 — causal circuit discovery methods |
| `2211.00593v1.pdf` | **IOI circuit**, Wang et al. 2022 (Redwood) — foundational circuit analysis |
| `NeurIPS-2021-language-models-...pdf` | **ESM / zero-shot mutation effects**, Meier/Rives et al. |
| `NeurIPS-2023-proteingym-...pdf` | **ProteinGym**, NeurIPS 2023 — fitness-prediction benchmark |
| `SAE for MEP (2).key` | Slides — SAEs for mutation-effect prediction |
| *(arXiv:2406.04093)* | **Scaling & evaluating SAEs**, Gao et al. 2024 — the TopK recipe behind base `sparsify` |
| *(arXiv:2503.17272)* | **KL+FVU training** — the combined-loss method |

**Related docs in this repo:** `docs/objective1_writeup.md` (full optimization plan + evaluation framework), `docs/presentation_xiaocong.md` (technical walkthrough), `docs/objective1.md` (short objective statement), root `CLAUDE.md` and `README.md` (build/run/train commands).

---

## 11. Getting started checklist

```bash
# 1. Environment
uv venv --seed && uv sync
pip install -e .[dev]
pre-commit install

# 2. Sanity: run the tests
pytest tests/

# 3. Read, in order: config.py → sparse_coder.py → runner.py → trainer.py
#    plus docs/objective1_writeup.md and docs/presentation_xiaocong.md

# 4. A small CLT smoke run (reduced scale — see §7 Step 2)
uv run python -m sparsify facebook/esm2_t33_650M_UR50D <DATASET> --run_name smoke \
  --transcode=True --skip_connection=True --filter_bos=True \
  --expansion_factor 32 --k 32 --cross_layer 4 --layer_stride 2 \
  --loss_fn kl-fvu --optimizer adam --lr 1e-4

# 5. First real task: §8 — apply CLTs to the ProtoMech two-protein circuit check
```

Set `WANDB_ENTITY` for W&B logging. Checkpoints land in `checkpoints/<run_name>/`. Confirm the exact ESM-2 HF model id and dataset path with Saishradha before your first full run.
