#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$SCRIPT_DIR}"
CONDA_ENV="${CONDA_ENV:-diffuse}"

if [ -n "${CONDA_INIT:-}" ]; then
    source "$CONDA_INIT"
fi
conda activate "$CONDA_ENV"

cd "$PROJ_ROOT"

bash run_conditioned_pipeline.sh
