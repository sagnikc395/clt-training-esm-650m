# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is a fork of [sparsify](https://github.com/eleutherai/sparsify) that adds support for training cross-layer transcoders (CLTs) as described in [Crosscoders (Anthropic, 2024)](https://transformer-circuits.pub/2024/crosscoders/index.html). The codebase trains k-sparse autoencoders (SAEs) and transcoders on language model activations.

Key additions over base sparsify:
- Cross-layer transcoder (CLT) support
- Tensor parallelism with DTensor
- Enhanced sparse kernel usage
- bfloat16 training support
- Input hookpoints before layernorm
- KL+FVU training loss
- ESM2 protein language model support (basic SAE training working)

## ESM2 Support

Basic SAE training works on ESM2-650M. See main README for setup and next steps.

## Installation & Setup

```bash
# Install dependencies
uv venv --seed && uv sync

# Install for development
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install
```

If installation fails, try:
- Loosening torch version requirements in `pyproject.toml`
- Removing torch CUDA index from `[[tool.uv.index]]/[tool.uv]/[tool.uv.sources]`

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_encode.py
pytest tests/test_decode.py
```

## Training Commands

### Basic Training

Train SAEs/transcoders using the `sparsify` module:

```bash
# Basic usage
uv run python -m sparsify <MODEL_NAME> <DATASET_NAME> --run_name <NAME> \
  --filter_bos True --ctx_len 2048 --max_examples 1000000 \
  --transcode=True --skip_connection=True \
  --tp=2 --expansion_factor 128 --k 32 \
  --lr 1e-4 --optimizer adam --b1 0.0 --b2 0.999 --lr_warmup_steps 50

# Train cross-layer transcoder (CLT) with 12 layers ahead
uv run python -m sparsify <MODEL_NAME> <DATASET_NAME> \
  --cross_layer=12 \
  --train_post_encoder=True --post_encoder_scale=True

# Train per-target tied CLT
uv run python -m sparsify <MODEL_NAME> <DATASET_NAME> \
  --coalesce_topk=concat --topk_coalesced=False
```

### Distributed Training

For multi-GPU training, use `torchrun`:

```bash
# Distributed Data Parallel (DDP)
torchrun --nproc_per_node gpu -m sparsify meta-llama/Meta-Llama-3-8B \
  --batch_size 1 --layers 16 24 --k 192 --grad_acc_steps 8 --ctx_len 2048

# Module distribution across GPUs (memory efficient)
torchrun --nproc_per_node gpu -m sparsify meta-llama/Meta-Llama-3-8B \
  --distribute_modules --batch_size 1 --layer_stride 2 \
  --grad_acc_steps 8 --ctx_len 2048 --k 192 \
  --load_in_8bit --micro_acc_steps 2
```

### Using Training Scripts

Pre-configured training scripts are in `scripts/training/`:

```bash
# GPT-2 sweep with various configurations
./scripts/training/gpt2-sweep <CFG> <K> <EF> <BS> <LR>

# Example: Train CLT with k=16, expansion_factor=128, batch_size=8, lr=2e-4
./scripts/training/gpt2-sweep cross 16 128 8 2e-4

# Available configurations: none, no-affine, btopk, tied, source-tied, cross, etc.
```

## Code Architecture

### Core Components

**sparsify/sparse_coder.py** - `SparseCoder` and `Sae` classes
- Main autoencoder/transcoder implementation
- Forward pass returns `ForwardOutput` with sae_out, latent activations/indices, and FVU
- Multi-target support for cross-layer transcoders
- `MidDecoder` class for intermediate decoding in cross-layer setups

**sparsify/trainer.py** - `Trainer` and `SaeTrainer` classes
- Handles training loops, optimization, and checkpointing
- Automatically configures hookpoints from layer specifications
- Supports DDP and tensor parallelism via DeviceMesh
- End-to-end training with CE/KL loss or reconstruction with FVU loss

**sparsify/runner.py** - `CrossLayerRunner` class
- Orchestrates multi-layer encoding/decoding for CLTs
- Manages gradients across layers with detach/restore mechanisms
- Implements TopK coalescing strategies (concat, per-layer, group)
- Maintains state for cross-layer feature passing

**sparsify/config.py** - Configuration dataclasses
- `SparseCoderConfig` (aliased as `SaeConfig`): Model architecture settings
- `TrainConfig`: Training hyperparameters and optimization settings
- `TranscoderConfig`: Shorthand for transcoder-specific configs

**sparsify/fused_encoder.py** - Optimized encoder kernels
- Fused CUDA kernels for efficient TopK/BatchTopK operations
- Dead latent penalty computation
- `EncoderOutput` dataclass for encoder forward pass results

**sparsify/__main__.py** - CLI entry point
- Parses `RunConfig` combining model, dataset, and training settings
- Handles distributed initialization with `torchrun`
- Loads model/dataset artifacts and initializes Trainer

### Key Abstractions

**Hookpoints**: String patterns matching module names where SAEs/transcoders are inserted
- Use Unix glob patterns: `h.*.attn`, `h.[012].mlp.act`
- Automatically resolved from `--layers` or `--hookpoints` arguments
- Naturally sorted to ensure consistent ordering

**Cross-layer transcoders**:
- Set `--cross_layer=N` to predict N layers ahead
- `CrossLayerRunner` coordinates encoding at layer L and decoding at layer L+N
- Supports multiple coalescing strategies via `--coalesce_topk`

**Tensor Parallelism**:
- Configure with `--tp=N` for N-way tensor parallelism
- Uses DTensor for sharding across devices
- Encoder/decoder weights sharded appropriately based on `tp_output`

### Loss Functions

- `fvu`: Fraction of variance unexplained (reconstruction loss)
- `ce`: Cross-entropy loss on final model logits
- `kl`: KL divergence between SAE-modified and original logits
- `kl-fvu`: Combined KL and FVU loss (for KL+FVU training)

### Optimizers

- `adam`: Standard Adam with warmup/linear decay
- `adam8`: 8-bit Adam (memory efficient)
- `muon`: Muon optimizer
- `signum`: Sign-based SGD (default)

## Code Style & Linting

```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Auto-format with black
black sparsify/

# Lint with ruff
ruff check sparsify/ --fix
```

Note: `scripts/` directory is excluded from black and ruff checks.

## Checkpointing

Checkpoints are saved to `checkpoints/<run_name>/` with structure:
- `<hookpoint_name>/`: Per-hookpoint SAE weights and config
- Training state includes optimizer state, step count, and dataset position

Resume training with `--resume` flag or finetune from existing checkpoint with `--finetune <path>`.

## Weights & Biases

Logging controlled by:
- `--log_to_wandb`: Enable/disable W&B logging (default: True)
- `--run_name`: W&B run name (also determines checkpoint path)
- `--wandb_log_frequency`: Steps between logs (default: 1)

Set `WANDB_ENTITY` environment variable for W&B organization.

## Important Notes

- Dataset must have `input_ids` column or will be tokenized automatically
- Context length (`--ctx_len`) must match model's expected sequence length
- For KL loss, use `--filter_bos=True` to exclude BOS tokens
- Requires recent wandb version (>=0.19.11) for proper logging
- Pre-commit hooks enforce trailing whitespace, EOF fixes, and file size limits

