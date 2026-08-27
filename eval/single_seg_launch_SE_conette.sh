#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 SEGMENT_ID TOTAL_SEGMENTS" >&2
    exit 2
fi

if ! command -v oarsub >/dev/null 2>&1; then
    echo "Error: oarsub is not available on host $(hostname)." >&2
    echo "Submit this script from a configured cluster frontend." >&2
    exit 127
fi

SEGMENT_ID="$1"
TOTAL_SEGMENTS="$2"
if ! [[ "$SEGMENT_ID" =~ ^[0-9]+$ && "$TOTAL_SEGMENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: segment arguments must be non-negative/positive integers." >&2
    exit 2
fi
if (( SEGMENT_ID >= TOTAL_SEGMENTS )); then
    echo "Error: SEGMENT_ID=$SEGMENT_ID must be smaller than TOTAL_SEGMENTS=$TOTAL_SEGMENTS." >&2
    exit 2
fi

GPUS="${GPUS:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
CLUSTER_FILTER="${CLUSTER_FILTER:-gpu='YES'}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

OUT_DIR="${OUT_DIR:-OUT/conette}"
mkdir -p "$OUT_DIR"

JOB_COMMAND="bash '$PROJ_ROOT/eval/SE_eval_conette.sh' '$SEGMENT_ID' '$TOTAL_SEGMENTS'"
echo "Submitting CoNeTTE segment ${SEGMENT_ID}/${TOTAL_SEGMENTS}"
echo "Job command: $JOB_COMMAND"

oarsub \
    -vv \
    -p "$CLUSTER_FILTER" \
    -q production \
    -l "nodes=1/gpu=$GPUS,walltime=$WALLTIME" \
    -O "$OUT_DIR/oar_conette.%jobid%.output" \
    -E "$OUT_DIR/oar_conette.%jobid%.error" \
    "$JOB_COMMAND"
