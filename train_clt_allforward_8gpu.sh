#!/bin/bash
#SBATCH --job-name=esm2_clt_allfw_8gpu
#SBATCH --output=clt-training-optimized/logs_clt/clt_allfw_%j.out
#SBATCH --error=clt-training-optimized/logs_clt/clt_allfw_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:8
#SBATCH --partition=superpod-a100
#SBATCH --reservation=sai
#SBATCH --time=72:00:00

# =============================================
# CLT Training with ALL FORWARD connections (cross_layer=33)
# Using 8 A100-80GB GPUs
# =============================================
#
# cross_layer=33 = every layer decodes to ALL downstream layers.
#   Layer 0 → 33 targets, Layer 1 → 32 targets, ..., Layer 32 → 1 target
#   Total decoder target-layer pairs: 33*34/2 = 561
#
# Estimated SAE parameters: ~9.7B (1.35x the 7.2B of cross_layer=16)
#   - In bfloat16: ~19.5 GB total SAE weights
#   - Encoder params (33 SAEs): ~541M
#   - Decoder params (561 target-layer pairs): ~9.2B
#
# Memory budget per GPU with TP=4, DP=2 (8 GPUs total):
#   - SAE params:  ~19.5 GB / 4 TP = ~4.9 GB per GPU
#   - Adam8 states: ~9.7B / 4 TP ≈ ~2.4 GB per GPU
#   - ESM2 frozen: ~2.6 GB (replicated)
#   - Batch + backward: ~10-15 GB (also split by TP)
#   - Total: ~20-25 GB per GPU → ample headroom on 80 GB A100
#
# Compared to 4-GPU cross_layer=16 baseline (31 s/it):
#   - 2x GPUs → higher TP reduces per-GPU SAE memory
#   - TP=4 vs TP=2 → more communication but smaller per-rank tensors
#   - batch_size=8 (reduced from 28 to avoid OOM in sparse backward)
#   - grad_acc_steps=14 → effective batch = 8 * 2 DP * 14 acc = 224
# =============================================

NGPUS=8

# Create logs directory
mkdir -p clt-training-optimized/logs_clt

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Num GPUs: $NGPUS"
echo "Mode: CLT all-forward (cross_layer=33)"
echo "Start time: $(date)"
echo "=========================================="

# Check GPUs
nvidia-smi --list-gpus

# Activate virtual environment
source clt-training/.venv/bin/activate

# Set CPU threads per process to avoid oversubscription
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run distributed training with torchrun (handles DDP + TP)
SPARSIFY_NO_COMPILE=1 \
SPARSIFY_DISABLE_TRITON=1 \
SPARSIFY_USE_XFORMERS=0 \
TORCHDYNAMO_DISABLE=1 \
TORCHINDUCTOR_DISABLE=1 \
uv run torchrun --nproc_per_node=$NGPUS \
  -m sparsify facebook/esm2_t33_650M_UR50D \
  clt-training/protein_test_dataset_1M \
  --loss_fn fvu \
  --run_name esm2_clt_allfw_b8_s1m_k64_cl33_dp2tp4 \
  --cross_layer 33 \
  --transcode True \
  --expansion_factor 10 \
  --dtype bfloat16 \
  --k 64 \
  --batch_size 8 \
  --lr 1e-4 \
  --optimizer adam8 \
  --save_every 500 \
  --grad_acc_steps 14 \
  --wandb_log_frequency 5 \
  --tp 4

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

