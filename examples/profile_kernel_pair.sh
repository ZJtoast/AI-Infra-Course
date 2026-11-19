#!/usr/bin/env bash
# Run beside the lab binaries. Each report contains one matched launch.
set -euo pipefail
dry_run=false
if [[ ${1:-} == --dry-run ]]; then dry_run=true; shift; fi
if [[ $# != 2 ]]; then
    echo "usage: bash profile_kernel_pair.sh [--dry-run] gemm|transpose|reduction|graph-short|graph-skew|softmax|stride|fma|hist-hot|hist-spread OUTPUT_DIR" >&2
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
    fma) kernels=('fma_chains<1>' 'fma_chains<2>' 'fma_chains<4>') ;;
    hist-hot|hist-spread) kernels=('segment_histogram<0>' 'segment_histogram<1>' 'segment_histogram<2>') ;;
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
# Resolve names from the device image. Different NCU versions print template
# arguments differently in demangled mode (e.g. <4> versus <(int)4>).
image=$(cuobjdump --dump-resource-usage "$binary")
declare -A symbols
while IFS= read -r symbol; do
    [[ -n $symbol ]] || continue
    signature=$(c++filt "$symbol")
    name=${signature%%(*}
    name=${name#void }
    symbols["$name"]=$symbol
done < <(printf '%s\n' "$image" | sed -n 's/^[[:space:]]*Function \([^:]*\):$/\1/p')
if ! $dry_run; then mkdir -p -- "$out"; fi
for i in "${!kernels[@]}"; do
    kernel=${kernels[i]}
    label=${kernel//[<>]/}
    if [[ $mode == stride ]]; then workload=${strides[i]}; label="stride$workload"; fi
    report="$out/$mode-$label"
    symbol=${symbols[$kernel]:-}
    if [[ -z $symbol ]]; then echo "device function not found: $kernel in $binary" >&2; exit 1; fi
    cmd=(ncu --kernel-name-base mangled -k "regex:^${symbol}$" "${filters[@]}"
         --launch-skip "$skip" --launch-count 1 --import-source yes "${sections[@]}"
         -o "$report" "$binary" "$workload")
    if $dry_run; then printf '%q ' "${cmd[@]}"; printf '\n'; else
        if [[ -e "$report.ncu-rep" ]]; then echo "report already exists: $report.ncu-rep" >&2; exit 1; fi
        "${cmd[@]}"
        [[ -s "$report.ncu-rep" ]] || { echo "no report for $kernel" >&2; exit 1; }
    fi
done
