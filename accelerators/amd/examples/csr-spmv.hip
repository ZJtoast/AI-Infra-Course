#include <hip/hip_runtime.h>
__global__ void csr_rows(const int* offsets, const int* cols, const float* weights,
                         const float* x, float* y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;                       // 一个有效线程负责一个顶点
    float acc = 0;
    for (int e = offsets[row]; e < offsets[row+1]; ++e) {
        acc += weights[e] * x[cols[e]];             // 边表连续，特征按列号间接读
    }
    y[row] = acc;                                  // 空行也有定义良好的输出
}
