#!/bin/bash

# =========================
# Paths and run parameters
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-diffuse}"
PROJ_ROOT="${PROJ_ROOT:-$SCRIPT_DIR}"

RAW_METADATA_DIR="$PROJ_ROOT/sgmse/metadata_combination_v5"
METADATA_DIR="$PROJ_ROOT/sgmse/metadata_combination_v5_clap_audio"
TRAIN_JSONL="$METADATA_DIR/train.jsonl"

CLAP_MODEL="laion/clap-htsat-unfused"
CLAP_BATCH_SIZE=8

RUN_ID="v5_clap_audio_12_08_2026"
CKPT_DIR="$PROJ_ROOT/logs/$RUN_ID"

BATCH_SIZE=8
NUM_WORKERS=4
DEVICES=2
MAX_EPOCHS=200

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

# Cache CLAP audio embeddings once. The generated JSONL files have the same
# format consumed by the original v2/v3 training pipeline.
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "--- CLAP metadata not found. Encoding noise audio files... ---"
    cd "$PROJ_ROOT"
    python -m sgmse.data_module_v5 \
        --input-dir "$RAW_METADATA_DIR" \
        --output-dir "$METADATA_DIR" \
        --model-name-or-path "$CLAP_MODEL" \
        --batch-size "$CLAP_BATCH_SIZE"
else
    echo "--- CLAP metadata found at $TRAIN_JSONL. Skipping encoding step. ---"
fi

# Resume the newest checkpoint when one exists, following the original
# run_conditioned_pipeline.sh behavior.
RESUME_PATH="None"
if [ -d "$CKPT_DIR" ]; then
    LAST_CKPT=$(ls -t "$CKPT_DIR"/*.ckpt 2>/dev/null | head -n 1)
    if [ -n "$LAST_CKPT" ]; then
        RESUME_PATH="$LAST_CKPT"
        echo "--- Found existing checkpoint: $RESUME_PATH ---"
    fi
fi

if [ "$RESUME_PATH" == "None" ]; then
    echo "--- No checkpoint found. Starting fresh training. ---"
fi

echo "--- Launching v5 Training Loop ---"
cd "$PROJ_ROOT"

python train.py \
    --train_jsonl "$METADATA_DIR/train.jsonl" \
    --val_jsonl "$METADATA_DIR/val.jsonl" \
    --test_jsonl "$METADATA_DIR/test.jsonl" \
    --backbone ncsnpp \
    --sde ouve \
    --conditioning_dim 512 \
    --conditioning_fusion film \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --accelerator gpu \
    --devices "$DEVICES" \
    --max_epochs "$MAX_EPOCHS" \
    --wandb_project "se-smd" \
    --run_id "$RUN_ID" \
    --resume_from_checkpoint "$RESUME_PATH"

echo "--- Process Complete ---"
