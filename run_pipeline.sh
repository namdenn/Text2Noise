#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-diffuse}"
PROJ_ROOT="${PROJ_ROOT:-$SCRIPT_DIR}"
METADATA_DIR="$PROJ_ROOT/sgmse/metadata_combination_encoded"
TRAIN_JSONL="$METADATA_DIR/train.jsonl"

RUN_ID="combined_dataset_v1"

echo "--- Activating Environment ---"
if [ -n "${CONDA_INIT:-}" ]; then
    source "$CONDA_INIT"
fi
conda activate "$CONDA_ENV_PATH"

export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export PL_TORCH_DISTRIBUTED_BACKEND="nccl"

if [ ! -f "$TRAIN_JSONL" ]; then
    echo "--- Encoded metadata not found. Starting offline encoding... ---"
    python "$PROJ_ROOT/sgmse/data_module.py"
else
    echo "--- Encoded metadata found at $TRAIN_JSONL. Skipping encoding step. ---"
fi

echo "--- Starting Training ---"
cd "$PROJ_ROOT"

python train.py \
    --train_jsonl "$METADATA_DIR/train.jsonl" \
    --val_jsonl "$METADATA_DIR/val.jsonl" \
    --test_jsonl "$METADATA_DIR/test.jsonl" \
    --backbone ncsnpp \
    --sde ouve \
    --conditioning_dim 512 \
    --batch_size 8 \
    --num_workers 4 \
    --accelerator gpu \
    --devices 2 \
    --max_epochs 200 \
    --run_id "$RUN_ID" \
    --wandb_project "se-smd"

echo "--- Process Complete ---"
