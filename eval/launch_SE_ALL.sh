#!/bin/bash
set -euo pipefail

if ! command -v oarsub >/dev/null 2>&1; then
  echo "Error: oarsub is not available on host $(hostname)." >&2
  echo "Run this launcher from a configured cluster frontend." >&2
  echo "Inside an OAR job, run: bash eval/SE_eval.sh [SEGMENT TOTAL_SEGMENTS]" >&2
  exit 127
fi

total_seg="${TOTAL_SEGMENTS:-10}"
if ! [[ "$total_seg" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: TOTAL_SEGMENTS must be a positive integer, got '$total_seg'." >&2
  exit 2
fi

itotal_seg=$((total_seg-1))


# Run speech enhancement on each segment of test files

for i in $(seq 0 "$itotal_seg"); do
  bash ./eval/single_seg_launch_SE.sh "$i" "$total_seg"
done
