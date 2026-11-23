#include <hip/hip_runtime.h>

// grid.x 等于行数，block.x 等于 B（2 的幂），每行 D 个特征。
template<int B>
__global__ void rms_rows(const float* x, const float* weight, float* y, int D, float eps) {
    __shared__ float part[B];
    int t = threadIdx.x, base = blockIdx.x * D;
    float sum = 0;
    for (int j=t; j<D; j+=B) { float v=x[base+j]; sum += v*v; }
    part[t]=sum;
    __syncthreads();
    for (int step=B/2; step; step/=2) {
        if (t<step) part[t] += part[t+step];
        __syncthreads();
    }
    float scale = rsqrtf(part[0]/D + eps);
    for (int j=t; j<D; j+=B) y[base+j] = x[base+j] * scale * weight[j];
}
