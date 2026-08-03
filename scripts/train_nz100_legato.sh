#!/usr/bin/env bash
set -euo pipefail
# Run with: bash scripts/train_nz100_legato.sh

# Modify these values for your Legato training run.
DATA_REPO_ID="/mnt/16T/lisq5005_dir/openpi/data/data_open_close_package"
EXP_NAME="nz100_legato_v1_open_close_package"
GPU_ID="0"
NUM_TRAIN_STEPS="30000"
BATCH_SIZE="32"
NUM_WORKERS="8"
LOG_INTERVAL="100"
SAVE_INTERVAL="1000"
KEEP_PERIOD="5000"
FSDP_DEVICES="1"
CONFIG_NAME="pi05_nz100_legato"

# Legato parameters. Keep LEGATO_TRAIN_NUM_STEPS aligned with inference num_steps.
LEGATO_OMEGA_DIM="31"
LEGATO_LOSS_ACTION_DIM="16"
LEGATO_TRAIN_NUM_STEPS="10"
LEGATO_FULL_GUIDANCE_MIN="0"
LEGATO_FULL_GUIDANCE_MAX="4"
LEGATO_RAMP_MIN="0"
LEGATO_RAMP_MAX="0"

CONDA_ROOT="/mnt/16T/App_dir/conda_dir/miniconda3"
CONDA_ENV="${CONDA_ROOT}/envs/openpi_lsq"
WORK_DIR="/mnt/16T/lisq5005_dir"
CACHE_DIR="${WORK_DIR}/.cache"
ASSETS_DIR="${WORK_DIR}/openpi_assets"
CHECKPOINT_DIR="${WORK_DIR}/openpi_checkpoints"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${CACHE_DIR}" "${ASSETS_DIR}" "${CHECKPOINT_DIR}"

export OPENPI_DATA_HOME="${CACHE_DIR}/openpi"
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
    --config-name "${CONFIG_NAME}" \
    --repo-id "${DATA_REPO_ID}" \
    --assets-base-dir "${ASSETS_DIR}" \
    --num-workers "${NUM_WORKERS}" \
    --skip-videos

python scripts/train.py "${CONFIG_NAME}" \
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
    --fsdp-devices "${FSDP_DEVICES}" \
    --model.legato-enabled \
    --model.legato-omega-dim "${LEGATO_OMEGA_DIM}" \
    --model.legato-loss-action-dim "${LEGATO_LOSS_ACTION_DIM}" \
    --model.legato-train-num-steps "${LEGATO_TRAIN_NUM_STEPS}" \
    --model.legato-randomize-schedule \
    --model.legato-full-guidance-min "${LEGATO_FULL_GUIDANCE_MIN}" \
    --model.legato-full-guidance-max "${LEGATO_FULL_GUIDANCE_MAX}" \
    --model.legato-ramp-min "${LEGATO_RAMP_MIN}" \
    --model.legato-ramp-max "${LEGATO_RAMP_MAX}"
