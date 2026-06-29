#!/bin/bash
#SBATCH --job-name=esm2_clt_1m_cl16
#SBATCH --output=clt-training/logs_1m_all_layer/clt_train_cl16_%j.out
#SBATCH --error=clt-training/logs_1m_all_layer/clt_train_cl16_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --gres=gpu:4
#SBATCH --partition=superpod-a100
#SBATCH --time=72:00:00

# =============================================
# CLT Training with cross_layer=16
# =============================================
# cross_layer=16 significantly increases SAE size:
#   cross_layer=4:  ~20 GB SAE params (9 SAEs, each decodes to 4 layers)
#   cross_layer=16: ~80 GB SAE params (3 SAEs at stride=4, each decodes to 16 layers)
#
# SAE params ~80 GB = can't fit replicated on one 80 GB A100.
# Must use TP to shard SAE weights across GPUs.
#
# With 4 GPUs (DP=2, tp=2):
#   - SAE params: ~40 GB per GPU (sharded across TP dim)
#   - adam8 states: ~10 GB per GPU
#   - ESM2 frozen: ~2.6 GB
#   - Batch + backward: ~10-15 GB (also split by TP)
#   - Total: ~62-68 GB per GPU
# =============================================

NGPUS=4

# Create logs directory if it doesn't exist
mkdir -p clt-training/logs_1m_all_layer

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

# Run distributed training
SPARSIFY_NO_COMPILE=1 \
SPARSIFY_DISABLE_TRITON=1 \
SPARSIFY_USE_XFORMERS=0 \
TORCHDYNAMO_DISABLE=1 \
TORCHINDUCTOR_DISABLE=1 \
uv run torchrun --nproc_per_node=$NGPUS \
  -m sparsify facebook/esm2_t33_650M_UR50D \
  clt-training/protein_test_dataset_1M \
  --loss_fn fvu \
  --run_name esm2_clt_b8_s1m_k64_cl16_dp2tp2 \
  --cross_layer 16 \
  --transcode True \
  --expansion_factor 10 \
  --dtype bfloat16 \
  --k 64 \
  --batch_size 24 \
  --lr 1e-4 \
  --optimizer adam8 \
  --save_every 500 \
  --tp 2

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

