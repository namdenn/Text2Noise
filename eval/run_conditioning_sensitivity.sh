#!/bin/bash

set -e

if [ -n "${CONDA_INIT:-}" ]; then
    source "$CONDA_INIT"
fi
conda activate "${CONDA_ENV:-diffuse}"

export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export FORCE_CUDA=1
export NCCL_P2P_DISABLE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJ_ROOT"

# if command -v nvidia-smi &> /dev/null; then
#     export TORCH_CUDA_ARCH_LIST="Auto"
# else
#     export TORCH_CUDA_ARCH_LIST="8.0" 
# fi

CKPT_DIFFUSION="${CKPT_DIFFUSION:-$PROJ_ROOT/logs/v2_01_08_2026/last.ckpt}"
CKPT_TEXT="${CKPT_TEXT:-$PROJ_ROOT/checkpoints/text_encoder_only.pt}"
TEXT_MODEL_NAME_OR_PATH="${TEXT_MODEL_NAME_OR_PATH:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval/result/conditioning_sensitivity}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
DEVICE="${DEVICE:-auto}"

if [ ! -f "$CKPT_DIFFUSION" ]; then
    echo "Missing diffusion checkpoint: $CKPT_DIFFUSION" >&2
    exit 1
fi

if [ ! -f "$CKPT_TEXT" ]; then
    echo "Missing text checkpoint: $CKPT_TEXT" >&2
    exit 1
fi

ARGS=(
    eval/conditioning_sensitivity.py
    --ckpt_diffusion "$CKPT_DIFFUSION" \
    --ckpt_text "$CKPT_TEXT" \
    --text_model_name_or_path "$TEXT_MODEL_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --labels babble car cafe street lr white \
    --num_seeds 4 \
    --seed_start 0 \
    --duration_sec 4.0 \
    --num_E 30 \
    --device "$DEVICE" \
    --include_null \
    --include_random
)

if [ "$LOCAL_FILES_ONLY" != "0" ]; then
    ARGS+=(--local_files_only)
fi

python "${ARGS[@]}"
