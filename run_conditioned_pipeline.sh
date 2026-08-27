#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-diffuse}"
PROJ_ROOT="${PROJ_ROOT:-$SCRIPT_DIR}"
METADATA_DIR="${METADATA_DIR:-$PROJ_ROOT/generated/metadata/encoded}"
TRAIN_JSONL="$METADATA_DIR/train.jsonl"

RUN_ID="${RUN_ID:-v1}"
CKPT_DIR="$PROJ_ROOT/logs/$RUN_ID"

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

export WANDB_MODE=online


if [ ! -f "$TRAIN_JSONL" ]; then
    echo "--- Encoded metadata not found. Starting offline encoding... ---"
    python "$PROJ_ROOT/sgmse/data_module.py"
else
    echo "--- Encoded metadata found at $TRAIN_JSONL. Skipping encoding step. ---"
fi

RESUME_PATH="None"
if [ -d "$CKPT_DIR" ]; then
    LAST_CKPT=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | head -n 1)
    if [ -n "$LAST_CKPT" ]; then
        RESUME_PATH="$LAST_CKPT"
        echo "--- Found existing checkpoint: $RESUME_PATH ---"
        echo "--- Resuming from Epoch $(echo $LAST_CKPT | grep -o 'epoch=[0-9]*') ---"
    fi
fi

if [ "$RESUME_PATH" == "None" ]; then
    echo "--- No checkpoint found. Starting fresh training. ---"
fi

echo "--- Launching Training Loop ---"
cd "$PROJ_ROOT"

python train.py \
    --train_jsonl "$METADATA_DIR/train.jsonl" \
    --val_jsonl "$METADATA_DIR/val.jsonl" \
    --test_jsonl "$METADATA_DIR/test.jsonl" \
    --backbone ncsnpp \
    --sde ouve \
    --conditioning_dim 512 \
    --conditioning_fusion film \
    --batch_size 8 \
    --num_workers 4 \
    --accelerator gpu \
    --devices 2 \
    --max_epochs 200 \
    --wandb_project "se-smd" \
    --run_id "$RUN_ID" \
    --resume_from_checkpoint "$RESUME_PATH"

echo "--- Process Complete ---"
