#!/bin/bash
set -euo pipefail

if [ -n "${CONDA_INIT:-}" ]; then
    source "$CONDA_INIT"
fi
conda activate "${CONDA_ENV:-diffuse}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJ_ROOT"

SEGMENT="${1:--1}"
TOTAL_SEGMENTS="${2:-1}"

DATASET="${DATASET:-WSJ0}"
DATA_DIR="${DATA_DIR:-$PROJ_ROOT/eval/ntcd_timit.json}"
NOISY_ROOT="${NOISY_ROOT:?Set NOISY_ROOT to the noisy evaluation corpus directory}"
CLEAN_ROOT="${CLEAN_ROOT:?Set CLEAN_ROOT to the clean evaluation corpus directory}"


SPEECH_CKPT="${SPEECH_CKPT:?Set SPEECH_CKPT to the speech-prior checkpoint}"
V2_NOISE_CKPT="${V2_NOISE_CKPT:-$PROJ_ROOT/logs/v2_01_08_2026/last.ckpt}"
V2_METADATA="${V2_METADATA:-$PROJ_ROOT/sgmse/metadata_combination_encoded_audioldm_cpt/test.jsonl}"

SAVE_ROOT="${SAVE_ROOT:-$PROJ_ROOT/eval/result}"
ALGO_TYPE="separate_paradiffuseen"
TAG="v2_01_08_2026"
NUM_E=30
NBATCH=4
STARTSTEP=0
LMBD=5.75

EXTRA_ARGS=()
if [[ "${COMPUTE_METRICS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--compute_metrics --dnn_mos)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi
if [[ "${VERBOSE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--verbose)
fi
python eval/evaluation.py \
    --dataset "$DATASET" \
    --segment "$SEGMENT" \
    --num_segments "$TOTAL_SEGMENTS" \
    --ckpt_path "$SPEECH_CKPT" \
    --ckpt_noise_path "$V2_NOISE_CKPT" \
    --metadata-jsonl "$V2_METADATA" \
    --algo_type "$ALGO_TYPE" \
    --tag "$TAG" \
    --data_dir "$DATA_DIR" \
    --clean_root "$CLEAN_ROOT" \
    --noisy_root "$NOISY_ROOT" \
    --save_root "$SAVE_ROOT" \
    --num_E "$NUM_E" \
    --nbatch "$NBATCH" \
    --startstep "$STARTSTEP" \
    --lambda "$LMBD" \
    "${EXTRA_ARGS[@]}"
