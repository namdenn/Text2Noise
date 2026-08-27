#!/bin/bash
set -euo pipefail

CONDA_ENV_PATH="${CONDA_ENV_PATH:-diffuse}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$SCRIPT_DIR}"
METADATA_DIR="${METADATA_DIR:-$PROJ_ROOT/sgmse/metadata_combination_encoded_audioldm_cpt}"
TRAIN_JSONL="$METADATA_DIR/train.jsonl"

RUN_ID="${RUN_ID:-v4_flow_matching_12_08_2026}"
CKPT_DIR="$PROJ_ROOT/logs/$RUN_ID"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICES="${DEVICES:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"

echo "--- Activating Environment ---"
if [ -n "${CONDA_INIT:-}" ]; then
    source "$CONDA_INIT"
fi
conda activate "$CONDA_ENV_PATH"

export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FORCE_CUDA=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export PL_TORCH_DISTRIBUTED_BACKEND="nccl"

export WANDB_MODE="${WANDB_MODE:-online}"


if [ ! -f "$TRAIN_JSONL" ]; then
    echo "--- Encoded metadata not found. Starting offline encoding... ---"
    python "$PROJ_ROOT/sgmse/data_module.py"
else
    echo "--- Encoded metadata found at $TRAIN_JSONL. Skipping encoding step. ---"
fi

RESUME_PATH="None"
if [ -f "$CKPT_DIR/last.ckpt" ]; then
    RESUME_PATH="$CKPT_DIR/last.ckpt"
    echo "--- Found existing checkpoint: $RESUME_PATH ---"
fi

if [ "$RESUME_PATH" == "None" ]; then
    echo "--- No checkpoint found. Starting fresh training. ---"
fi

echo "--- Launching Training Loop ---"
cd "$PROJ_ROOT"

python train_fm.py \
    --train_jsonl "$METADATA_DIR/train.jsonl" \
    --val_jsonl "$METADATA_DIR/val.jsonl" \
    --test_jsonl "$METADATA_DIR/test.jsonl" \
    --backbone ncsnpp \
    --sde ot_flow \
    --sigma-min 1e-4 \
    --conditioning_dim 512 \
    --conditioning_fusion film \
    --loss_reduction mean \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --accelerator gpu \
    --devices "$DEVICES" \
    --max_epochs "$MAX_EPOCHS" \
    --num_sanity_val_steps 2 \
    --log_every_n_steps 10 \
    --wandb_project "se-smd" \
    --run_id "$RUN_ID" \
    --resume_from_checkpoint "$RESUME_PATH"

echo "--- Process Complete ---"
