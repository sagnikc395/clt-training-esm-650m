#!/bin/bash
#SBATCH --job-name=esm2_plt_4gpu
#SBATCH --output=clt-training/logs_plt/plt_train_4gpu_%j.out
#SBATCH --error=clt-training/logs_plt/plt_train_4gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:4
#SBATCH --partition=superpod-a100
#SBATCH --time=72:00:00

# =============================================
# Per-Layer Transcoder (PLT) Training on 4 GPUs
# =============================================
# cross_layer=0 (default) — each SAE predicts its own layer only.
# No TP needed; all 4 GPUs used for data parallelism (dp=4).
#
# With 4 GPUs (dp=4, tp=1):
#   - SAE params: ~2 GB per GPU (replicated, ~1B total params)
#   - adam8 states: ~1 GB per GPU
#   - ESM2 frozen: ~2.6 GB
#   - Fixed total: ~5.6 GB per GPU
#
# Batch size chosen conservatively from 2-GPU history:
#   - batch_size=48 (24/GPU) on 2 GPUs → ~71 GiB (confirmed safe)
#   - batch_size=96 (48/GPU) on 2 GPUs → OOM
#   - batch_size=96 here = 24/GPU on 4 GPUs → same per-GPU load as safe run
# =============================================

NGPUS=4

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
source clt-training/.venv/bin/activate

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
  --run_name esm2_plt_b70_s1m_k64_dp4 \
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

