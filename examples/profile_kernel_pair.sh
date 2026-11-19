#!/usr/bin/env bash
# Run beside the lab binaries. Each report contains one matched launch.
set -euo pipefail
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
if [[ $# != 2 ]]; then
    echo "usage: bash profile_kernel_pair.sh [--dry-run] gemm|transpose|reduction|graph-short|graph-skew|softmax|stride OUTPUT_DIR" >&2
    exit 2
fi
mode=$1
out=$2
binary=./kernel_lab
workload=$mode
skip=2
filters=()
case $mode in
    gemm) kernels=(gemm_direct gemm_tile gemm_two_columns) ;;
    transpose) kernels=(transpose_direct 'transpose_tile<32>' 'transpose_tile<33>') ;;
    reduction) kernels=(reduce_tree reduce_warp_tail reduce_warp) ;;
    graph-short|graph-skew) kernels=(csr_thread csr_four_lanes csr_warp) ;;
    softmax) kernels=(softmax_serial softmax_eight_lanes softmax_warp32) ;;
    stride)
        kernels=(stride_add stride_add stride_add)
        strides=(4 2 1)
        binary=./stride_lab
        skip=0
        filters=(--nvtx --nvtx-include steady/)
        ;;
    *) echo "unknown workload: $mode" >&2; exit 2 ;;
esac
sections=()
for section in SpeedOfLight LaunchStats Occupancy SchedulerStats WarpStateStats ComputeWorkloadAnalysis MemoryWorkloadAnalysis InstructionStats SourceCounters; do
    sections+=(--section "$section")
done
if ! $dry_run; then mkdir -p -- "$out"; fi
for i in "${!kernels[@]}"; do
    kernel=${kernels[i]}
    label=${kernel//[<>]/}
    if [[ $mode == stride ]]; then workload=${strides[i]}; label="stride$workload"; fi
    report="$out/$mode-$label"
    cmd=(ncu --kernel-name-base demangled -k "regex:$kernel[(]" "${filters[@]}"
         --launch-skip "$skip" --launch-count 1 --import-source yes "${sections[@]}"
         -o "$report" "$binary" "$workload")
    if $dry_run; then printf '%q ' "${cmd[@]}"; printf '\n'; else "${cmd[@]}"; fi
done
