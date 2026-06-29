#!/bin/bash
#SBATCH --job-name=esm2_clt_1m_dist
#SBATCH --output=clt-training/logs_1m_all_layer/clt_train_dist_%j.out
#SBATCH --error=clt-training/logs_1m_all_layer/clt_train_dist_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --gres=gpu:2
#SBATCH --partition=superpod-a100
#SBATCH --time=48:00:00

# =============================================
# Distributed CLT Training (Multi-GPU, 1 Node)
# =============================================
# Request 4 GPUs on a SINGLE node for TP=4.
# This avoids multi-node communication overhead.
#
# GPU Configs (adjust --gres and NGPUS):
#   2 GPUs: --gres=gpu:2, NGPUS=2, --tp 2  (~39 GB/GPU, EF=16 cross=4)
#   4 GPUs: --gres=gpu:4, NGPUS=4, --tp 4  (~20 GB/GPU, EF=16 cross=4)
#   4 GPUs: --gres=gpu:4, NGPUS=4, --tp 4  (~41 GB/GPU, EF=16 cross=32)
# =============================================
# Use tp > 1 only when SAE params alone exceed 80 GB otherwise it adds additional memory overhead.

NGPUS=2

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
source .venv/bin/activate

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
  --run_name esm2_test_clt_b60_s1m_k64_dp2 \
  --cross_layer 4 \
  --transcode True \
  --expansion_factor 16 \
  --k 64 \
  --batch_size 36 \
  --lr 1e-4 \
  --optimizer adam8 \
  --save_every 500 \
  --tp 1

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

