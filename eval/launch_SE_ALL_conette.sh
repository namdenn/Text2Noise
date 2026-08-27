#!/bin/bash
set -euo pipefail

if ! command -v oarsub >/dev/null 2>&1; then
    echo "Error: oarsub is not available on host $(hostname)." >&2
    echo "Run this launcher from the Nancy frontend (fnancy)." >&2
    echo "Inside an OAR job, run eval/SE_eval_v3.sh directly." >&2
    exit 127
fi

TOTAL_SEGMENTS="${TOTAL_SEGMENTS:-10}"
if ! [[ "$TOTAL_SEGMENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: TOTAL_SEGMENTS must be a positive integer, got '$TOTAL_SEGMENTS'." >&2
    exit 2
fi

for ((segment_id = 0; segment_id < TOTAL_SEGMENTS; segment_id++)); do
    bash ./eval/single_seg_launch_SE_v3.sh "$segment_id" "$TOTAL_SEGMENTS"
done
