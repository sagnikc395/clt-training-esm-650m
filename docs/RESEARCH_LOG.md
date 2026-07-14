# Research Log — CLT on ESM-2 650M

> Personal working log for the CLT / ESM-2 650M project. Track what you **read**, what you **ran**, and what you **learned**.
> Companion to the reference PDF (`docs/CLT_ESM650M_reference.pdf`) and the onboarding guide (`docs/ONBOARDING.md`).
> Keep newest entries at the **top** of each section. Copy a template block and fill it in.

**Owner:** Sagnik Chatterjee · **Started:** 2026-07-14 · **Weekly sync:** Saishradha, Tuesdays

---

## 0. Status board (edit in place — the one-glance view)

| Track | Current focus | State | Next action |
|---|---|---|---|
| Optimization | Step 0 — profiling | ⬜ not started | Run `torch.profiler` over 10 steps |
| Applied / circuits | ProtoMech 2-protein check | ⬜ not started | Identify the two proteins in `2602.12026v2.pdf` |
| Reading | Core papers | 🟡 in progress | Finish ProtoMech + Nainani |

Legend: ⬜ not started · 🟡 in progress · ✅ done · ⏸️ blocked · ❌ dropped

**Open questions I'm carrying** (move to a paper/experiment entry once resolved):
- Which two proteins does ProtoMech use for mutation-effect prediction?
- Exact ESM-2 650M HF model id + dataset path for our runs? (confirm w/ Saishradha)
- Is `kl-fvu` or pure `kl` the right objective for circuit fidelity?

---

## 1. Reading log

Track every paper / doc you go through. Rate usefulness so future-you knows what to reread.

### Template — copy this
```
### <Short title> — <authors, venue year>
- **File / link:** papers/<file>.pdf
- **Date read:** YYYY-MM-DD   ·   **Status:** skimmed / read / deep-read
- **Relevance:** ★★★★☆
- **One-line takeaway:**
- **Key points:**
  -
- **What it changes for our project:**
- **Follow-ups / things to check:**
```

### Reading queue (priority order)
1. **ProtoMech** — Tsui et al., ICML 2026 — `papers/2602.12026v2.pdf` — our direct predecessor; find the 2 proteins.
2. **Nainani et al. 2025** — `papers/117_Mechanistic_evidence_that_.pdf` — motif→domain circuit; patching protocol.
3. **Crosscoders** — Anthropic 2024 (transformer-circuits.pub) — the CLT method itself.
4. **Scaling & evaluating SAEs** — Gao et al. 2024 (arXiv:2406.04093) — TopK recipe behind `sparsify`.
5. **Sparse Feature Circuits** — Marks et al., ICLR 2025 — `papers/2403.19647v3.pdf` — circuit discovery methods.
6. **KL+FVU training** — arXiv:2503.17272 — our combined loss.
7. **IOI circuit** — Wang et al. 2022 — `papers/2211.00593v1.pdf` — foundational circuit analysis.
8. **ProteinGym** — NeurIPS 2023 — `papers/NeurIPS-2023-proteingym-...pdf` — eval benchmark.
9. **ESM / zero-shot mutation effects** — Meier/Rives — `papers/NeurIPS-2021-...pdf`.

### Entries

<!-- newest at top; copy the template above -->

_(no entries yet — start with ProtoMech)_

---

## 2. Experiment log

One block per run (or tight cluster of runs). Fill in **before** launching (hypothesis + config) and **after** (result + verdict).
Link the W&B run. A run with no recorded verdict is an incomplete experiment.

### Template — copy this
```
### EXP-000 — <short name>
- **Date:** YYYY-MM-DD   ·   **Status:** planned / running / done / failed
- **Hypothesis / question:** what am I testing and what do I expect?
- **Config:**
  - model: facebook/esm2_t33_650M_UR50D   dataset: <...>
  - expansion_factor: __   k: __   cross_layer: __   layer_stride: __
  - loss_fn: __   optimizer: __   lr: __   tp: __   batch_size: __
  - other flags: __
- **Command:**
  ```
  uv run python -m sparsify ...
  ```
- **Hardware:** __ GPUs (__), __ GB each
- **W&B:** <url>   ·   **Checkpoint:** checkpoints/<run_name>/
- **Results:**
  | metric | value |
  |---|---|
  | val_mean_fvu | |
  | val_replacement_ce_increase | |
  | val_mean_l0 | |
  | dead_pct | |
  | tokens/sec | |
  | peak mem/GPU | |
- **Verdict:** ✅ keep / 🔁 iterate / ❌ discard — because...
- **Next:** what this run implies for the next one.
```

### Targets to beat (from the reference, §13)
`val_mean_fvu < 0.05` · `val_replacement_ce_increase < 0.1` nats · `dead_pct < 5%` · `< 40 GB`/GPU · **3–10× throughput** over baseline.

### Entries

<!-- newest at top; copy the template above -->

_(no experiments yet — EXP-001 should be the Step-0 profiling run, EXP-002 the reduced-scale smoke run)_

---

## 3. Optimization progress (maps to reference §12)

Check off as you complete each step. Link the experiment / commit that did it.

- [ ] **Step 0 — Profile** the current run (`torch.profiler` + `nsys`); record forward vs backward split. → EXP-___
- [ ] **Step 1 — Activation caching** (reuse `MemmapDataset`); decide subset vs streaming ring buffer. → EXP-___
- [ ] **Step 2 — Shrink the CLT** (EF=32, cross_layer=4–6, layer_stride=2); reproduce ProtoMech-style quality. → EXP-___
- [ ] **Step 3 — LoRA-style factorization** (shared decoder + per-target adapters; PKM/low-rank encoder). → EXP-___
- [ ] **Step 4 — Loop wins** (torch.compile audit of FusedEncoder backward; adam8; activation checkpointing; TP tuning). → EXP-___
- [ ] **Step 5 — Sweep** with `optimization/analyze_sweeps.py` on a layer subset; pick best (k, lr, optimizer, coalesce_topk). → EXP-___
- [ ] **Full 33-layer run** with the winning config. → EXP-___

**Baseline numbers (fill after Step 0 — everything is measured against these):**
| metric | baseline value | date |
|---|---|---|
| tokens/sec | | |
| steps/sec | | |
| % time in ESM-2 forward | | |
| % time in CLT backward | | |
| peak mem/GPU | | |

---

## 4. Applied / circuit-discovery progress (maps to reference §14)

- [ ] Identify ProtoMech's **two proteins** for mutation-effect prediction (`2602.12026v2.pdf`).
- [ ] Run our 650M CLT on those two proteins; extract features/circuits.
- [ ] **Sanity check:** do we recover the *same* circuits ProtoMech found on 35M? More features?
- [ ] Reproduce Nainani's **motif→domain** circuit via activation patching (MetXA P45131, TOP2 P06786).
- [ ] Move toward the **unsupervised** mutation-effect task (coordinate w/ Saishradha's paper).
- [ ] ProteinGym: do circuit-relevant CLT features alone predict mutation effects?

---

## 5. Weekly updates (for the Tuesday sync)

### Template
```
### Week of YYYY-MM-DD
- **Did:**
- **Found / learned:**
- **Blocked on:**
- **Next week:**
- **Questions for Saishradha:**
```

### Entries

<!-- newest at top -->

_(first update: after profiling + reading ProtoMech)_

---

## 6. Scratch notes & gotchas

Running list of small facts, fixes, and surprises worth not re-discovering.

- ESM-2 uses a **CLS** token, not BOS → mask with `--filter_bos=True` for position-specific losses.
- `kl-fvu` needs **two** ESM-2 forward passes/step → caching matters more under this loss.
- Default `expansion_factor=128` → ~89B CLT params across 33 layers; **start small** (EF=32).
- The detach/restore custom backward in `runner.py` blocks full-loop `torch.compile`.
- `--layers` and `--layer_stride` are **mutually exclusive** (config raises if both set).
- Checkpoints → `checkpoints/<run_name>/`; set `WANDB_ENTITY` before logging.
