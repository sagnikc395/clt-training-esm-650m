#!/bin/bash
#SBATCH --job-name=esm2_plt_1m
#SBATCH --output=clt-training/logs_plt/plt_train_%j.out
#SBATCH --error=clt-training/logs_plt/plt_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --partition=superpod-a100
#SBATCH --time=72:00:00

# =============================================
# Per-Layer Transcoder (PLT) Training
# =============================================
# cross_layer=0 (default) means each SAE predicts only its own
# layer's output — no forward cross-layer connections.
#
# With expansion_factor=10 on ESM2-650M (d=1280):
#   - 33 SAEs, each with 1 encoder + 1 decoder
#   - SAE params: ~1B total (vs ~7.2B for cross_layer=16)
#
# No TP needed — SAE easily fits on a single GPU.
# 2 GPUs used for data parallelism (dp=2) to double throughput.
#
# With 2 GPUs (dp=2, tp=1):
#   - SAE params: ~2 GB per GPU (replicated)
#   - adam8 states: ~1 GB per GPU
#   - ESM2 frozen: ~2.6 GB
#   - Batch + backward: ~38.6 GB attn matrices (28 seqs/GPU)
#   - Total: ~76.6 GB per GPU (tight but safe on 80 GB A100)
#
# batch_size=56 chosen after OOM at 96:
#   - batch_size=96 (48/GPU) OOM'd needing 59.88 GiB alloc (attn matrices)
#   - batch_size=48 (24/GPU) ran OK at ~71 GiB total
#   - batch_size=56 (28/GPU) → ~38.6 GiB attn + ~38 GiB fixed = ~76.6 GiB
#   - ~2.7 GiB headroom on 79.25 GiB A100
# =============================================

NGPUS=2

# Create logs directory if it doesn't exist
mkdir -p clt-training/logs_plt

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Num GPUs: $NGPUS"
echo "Start time: $(date)"
echo "=========================================="

# Check GPUs
nvidia-smi --list-gpus

# Activate virtual environment
source .venv/bin/activate

# Set CPU threads per process to avoid oversubscription
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NGPUS))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set master port for distributed training (avoid collisions)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Run distributed training
SPARSIFY_NO_COMPILE=1 \
SPARSIFY_DISABLE_TRITON=1 \
SPARSIFY_USE_XFORMERS=0 \
TORCHDYNAMO_DISABLE=1 \
TORCHINDUCTOR_DISABLE=1 \
uv run torchrun --nproc_per_node=$NGPUS --master_port=$MASTER_PORT \
  -m sparsify facebook/esm2_t33_650M_UR50D \
  clt-training/protein_test_dataset_1M \
  --loss_fn fvu \
  --run_name esm2_plt_64_k64 \
  --transcode True \
  --expansion_factor 10 \
  --dtype bfloat16 \
  --k 64 \
  --batch_size 70 \
  --lr 1e-4 \
  --optimizer adam8 \
  --save_every 500 \
  --grad_acc_steps 4

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

