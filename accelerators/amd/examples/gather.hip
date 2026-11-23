#include <hip/hip_runtime.h>

// 第 i 个输出行由 ids[i] 指定。输入、输出均为行主序 float 数组。
__global__ void gather_rows(const float* in, const int* ids, float* out,
                            int count, int features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * features) return;
    int out_row = index / features;
    int channel = index % features;
    out[index] = in[ids[out_row] * features + channel];
}
