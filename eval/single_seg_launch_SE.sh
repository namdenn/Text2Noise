#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 SEGMENT_ID TOTAL_SEGMENTS" >&2
    exit 2
fi

if ! command -v oarsub >/dev/null 2>&1; then
    echo "Error: oarsub is not available on host $(hostname)." >&2
    echo "Submit this script from the Nancy frontend." >&2
    exit 127
fi

seg_id="$1"
total_seg="$2"
if ! [[ "$seg_id" =~ ^[0-9]+$ && "$total_seg" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: segment arguments must be non-negative/positive integers." >&2
    exit 2
fi
if (( seg_id >= total_seg )); then
    echo "Error: SEGMENT_ID=$seg_id must be smaller than TOTAL_SEGMENTS=$total_seg." >&2
    exit 2
fi

gpus="${GPUS:-1}"
walltime="${WALLTIME:-24:00:00}"
cluster_filter="${CLUSTER_FILTER:-cluster in ('gruss')}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJ_ROOT:-$(cd -- "$script_dir/.." && pwd)}"

out_dir="${OUT_DIR:-OUT}"
mkdir -p "$out_dir"

job_command="bash '$project_root/eval/SE_eval.sh' '$seg_id' '$total_seg'"
echo "Submitting v2 segment $seg_id/$total_seg"
echo "Job command: $job_command"
oarsub \
    -vv \
    -p "$cluster_filter" \
    -q production \
    -l "nodes=1/gpu=${gpus},walltime=${walltime}" \
    -O "$out_dir/oar_job.%jobid%.output" \
    -E "$out_dir/oar_job.%jobid%.error" \
    "$job_command"
