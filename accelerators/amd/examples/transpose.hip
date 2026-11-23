#include <hip/hip_runtime.h>

// block=(16,16)，grid=(ceil(cols/16),ceil(rows/16))，输入、输出各存一张矩阵。
__global__ void tiled_transpose(const float* x, float* y, int rows, int cols) {
    __shared__ float tile[16][17];
    int u = threadIdx.x, v = threadIdx.y;
    int r = blockIdx.y * 16 + v, c = blockIdx.x * 16 + u;
    if (r < rows && c < cols) tile[v][u] = x[r * cols + c];
    __syncthreads();                         // 等整块输入就绪，尾块也全部到达
    int out_r = blockIdx.x * 16 + v, out_c = blockIdx.y * 16 + u;
    if (out_r < cols && out_c < rows)
        y[out_r * rows + out_c] = tile[u][v]; // 写入转置后的行，邻居线程写相邻列
}
