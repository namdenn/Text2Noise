#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-diffuse}"

if [ -n "${CONDA_INIT:-}" ]; then
    source "$CONDA_INIT"
fi
conda activate "$CONDA_ENV"
cd "$PROJ_ROOT"

TOTAL_SEGMENTS="${TOTAL_SEGMENTS:-10}"

for ((segment_id = 0; segment_id < TOTAL_SEGMENTS; segment_id++)); do
    echo "Running v3 evaluation segment ${segment_id}/${TOTAL_SEGMENTS}"
    bash eval/SE_eval_v3.sh "$segment_id" "$TOTAL_SEGMENTS"
done

echo "All ${TOTAL_SEGMENTS} v3 evaluation segments completed."
