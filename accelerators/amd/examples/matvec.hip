#include <hip/hip_runtime.h>

// 一个线程负责一行，A 按行连续存放，共 rows*cols 个 float。
__global__ void matvec(const float* A, const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col)
        sum += A[row * cols + col] * x[col];
    y[row] = sum;
}
