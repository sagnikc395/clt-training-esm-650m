#!/bin/bash
#SBATCH --job-name=esm2_clt_1m
#SBATCH --output=clt-training/logs_1m_all_layer/clt_train_%j.out
#SBATCH --error=clt-training/logs_1m_all_layer/clt_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --partition=superpod-a100
#SBATCH --time=48:00:00

# Create logs directory if it doesn't exist
mkdir -p clt-training/logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Check GPU
nvidia-smi --list-gpus

# Activate virtual environment
source .venv/bin/activate

# Run training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPARSIFY_NO_COMPILE=1 \
SPARSIFY_DISABLE_TRITON=1 \
SPARSIFY_USE_XFORMERS=0 \
TORCHDYNAMO_DISABLE=1 \
TORCHINDUCTOR_DISABLE=1 \
uv run python -m sparsify facebook/esm2_t33_650M_UR50D \
  clt-training/protein_test_dataset_1M \
  --loss_fn fvu \
  --run_name esm2_test_clt_b128_s1m_k64 \
  --cross_layer 4 \
  --transcode True \
  --expansion_factor 16 \
  --k 64 \
  --batch_size 60 \
  --lr 1e-4 \
  --save_every 1000

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="

