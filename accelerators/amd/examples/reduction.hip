#include <hip/hip_runtime.h>
template<int B>
__global__ void block_sum(const float* x, float* sums, int n) {
    __shared__ float v[B];                          // 每个块独有的 B 个槽
    unsigned lane = threadIdx.x;
    unsigned pos = blockIdx.x * B + lane;
    v[lane] = pos < n ? x[pos] : 0.0f;             // 末块用加法单位元补齐
    __syncthreads();                               // 所有槽写好，再开始配对
    for (unsigned step = B / 2; step > 0; step /= 2) {
        if (lane < step) v[lane] += v[lane + step];
        __syncthreads();                           // 整个块进入下一轮
    }
    if (lane == 0) sums[blockIdx.x] = v[0];        // 每块只交付一个小计
}
