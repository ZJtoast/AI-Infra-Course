#!/usr/bin/env bash
# Run beside the lab binaries. Each report contains one matched launch.
set -euo pipefail
dry_run=false
use_sudo=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) dry_run=true; shift ;;
        --sudo) use_sudo=true; shift ;;
        *) break ;;
    esac
done
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: bash profile_kernel_pair.sh [--dry-run] [--sudo] gemm|transpose|reduction|graph-short|graph-skew|softmax|stride|fma|hist-hot|hist-spread|tensor-full|tensor-thin|tensor-edge|tensor-control|tensor-grid|registers OUTPUT_DIR [CUDA_DEVICE]" >&2
    exit 2
fi
mode=$1
out=$2
device_args=()
if [[ $# == 3 ]]; then
    if [[ ! $3 =~ ^[0-9]+$ ]]; then
        echo "CUDA_DEVICE must be a nonnegative device index" >&2
        exit 2
    fi
    device_args=("$3")
fi
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
    tensor-full|tensor-thin|tensor-edge|tensor-control) kernels=(half_scalar_gemm 'warp_tile_gemm<1>' 'warp_tile_gemm<2>') ;;
    tensor-grid) kernels=('warp_tile_gemm<1>' 'warp_tile_gemm<1>' 'warp_tile_gemm<1>'); warp_counts=(1 2 4) ;;
    registers) kernels=('channels_limited<64>' 'channels_limited<8>' channels_free) ;;
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
local_metrics=""
if [[ $mode == registers ]]; then
    local_metrics=",l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_local_op_st.sum"
fi
profiler=(ncu)
if $use_sudo; then
    # Give sudo the resolved tool path so a profiling-only sudoers entry works.
    ncu_path=$(command -v ncu)
    profiler=(sudo -- "$ncu_path")
fi
for i in "${!kernels[@]}"; do
    kernel=${kernels[i]}
    label=${kernel//[<>]/}
    if [[ $mode == stride ]]; then workload=${strides[i]}; label="stride$workload"; fi
    if [[ $mode == tensor-grid ]]; then
        # All three partitions use the same binary symbol. In compare() they
        # validate three times, then each variant warms up and launches 20 times.
        skip=$((4 + 21*i))
        label="warps${warp_counts[i]}"
    fi
    report="$out/$mode-$label"
    symbol=${symbols[$kernel]:-}
    if [[ -z $symbol ]]; then echo "device function not found: $kernel in $binary" >&2; exit 1; fi
    cmd=("${profiler[@]}" --kernel-name-base mangled -k "regex:^${symbol}$" "${filters[@]}"
         --launch-skip "$skip" --launch-count 1 --import-source yes "${sections[@]}"
         --metrics dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warps_eligible.avg.per_cycle_active,smsp__issue_active.avg.pct_of_peak_sustained_active"${local_metrics}"
         -o "$report" "$binary" "$workload" "${device_args[@]}")
    if $dry_run; then printf '%q ' "${cmd[@]}"; printf '\n'; else
        if [[ -e "$report.ncu-rep" ]]; then echo "report already exists: $report.ncu-rep" >&2; exit 1; fi
        "${cmd[@]}"
        [[ -s "$report.ncu-rep" ]] || { echo "no report for $kernel" >&2; exit 1; }
    fi
done
