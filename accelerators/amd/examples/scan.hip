#include <hip/hip_runtime.h>

// 每块 B 个线程；out 存每项的包含式前缀和，totals 收集每块末值。
template<int B>
__global__ void scan_tiles(const int* in, int* out, int* totals, int n) {
    __shared__ int v[B];
    int t = threadIdx.x, i = blockIdx.x * B + t;
    v[t] = i < n ? in[i] : 0;
    __syncthreads();
    for (int d = 1; d < B; d *= 2) {
        int left = t >= d ? v[t-d] : 0; // 先读上一轮，再允许任何线程写
        __syncthreads();
        v[t] += left;
        __syncthreads();
    }
    if (i < n) out[i] = v[t];
    if (t == B-1) totals[blockIdx.x] = v[t];
}
template<int B>
__global__ void add_bases(int* out, const int* scanned_totals, int n) {
    int i = blockIdx.x * B + threadIdx.x;
    if (i < n && blockIdx.x > 0) out[i] += scanned_totals[blockIdx.x-1];
}
