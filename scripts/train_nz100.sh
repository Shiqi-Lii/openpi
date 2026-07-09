#!/usr/bin/env bash
set -euo pipefail
# Run with: bash scripts/train_nz100.sh

# Modify these values for your training run.
DATA_REPO_ID="data/data_lerobot_v21_h264"
EXP_NAME="nz100_v1"
GPU_ID="0"
NUM_TRAIN_STEPS="30000"
BATCH_SIZE="32"
NUM_WORKERS="4"
LOG_INTERVAL="100"
SAVE_INTERVAL="1000"
KEEP_PERIOD="5000"
FSDP_DEVICES="1"


CONDA_ROOT="/mnt/16T/App_dir/conda_dir/miniconda3"
CONDA_ENV="${CONDA_ROOT}/envs/openpi_lsq"
WORK_DIR="/mnt/16T/lisq5005_dir"
CACHE_DIR="${WORK_DIR}/cache"
ASSETS_DIR="${WORK_DIR}/openpi_assets"
CHECKPOINT_DIR="${WORK_DIR}/openpi_checkpoints"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


mkdir -p "${CACHE_DIR}" "${ASSETS_DIR}" "${CHECKPOINT_DIR}"

export UV_CACHE_DIR="${CACHE_DIR}/uv"
export HF_HOME="${CACHE_DIR}/huggingface"
export XDG_CACHE_HOME="${CACHE_DIR}"
export WANDB_DIR="${CACHE_DIR}/wandb"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"

python scripts/generate_lerobot_episodes_stats.py "${DATA_REPO_ID}"

python scripts/compute_norm_stats.py \
    --config-name pi05_nz100 \
    --repo-id "${DATA_REPO_ID}" \
    --assets-base-dir "${ASSETS_DIR}"

python scripts/train.py pi05_nz100 \
    --data.repo-id "${DATA_REPO_ID}" \
    --assets-base-dir "${ASSETS_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_DIR}" \
    --exp-name "${EXP_NAME}" \
    --num-train-steps "${NUM_TRAIN_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --log-interval "${LOG_INTERVAL}" \
    --save-interval "${SAVE_INTERVAL}" \
    --keep-period "${KEEP_PERIOD}" \
    --wandb-enabled \
    --fsdp-devices "${FSDP_DEVICES}"
