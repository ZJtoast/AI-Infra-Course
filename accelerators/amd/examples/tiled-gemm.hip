#include <hip/hip_runtime.h>
template<int T>
__global__ void tiled(const float* a, const float* b, float* c, int M, int N, int K) {
    __shared__ float at[T][T], bt[T][T];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.y * T + y, col = blockIdx.x * T + x;
    float acc = 0;
    for (int k = 0; k < K; k += T) {
        at[y][x] = row < M && k+x < K ? a[row*K + k+x] : 0; // 搬 A 的一行
        bt[y][x] = k+y < K && col < N ? b[(k+y)*N + col] : 0; // 搬 B 的一行
        __syncthreads();                                     // 两片都已到齐
        for (int j = 0; j < T; ++j) acc += at[y][j] * bt[j][x];
        __syncthreads();                                     // 槽位可供下一片覆盖
    }
    if (row < M && col < N) c[row*N + col] = acc;
}
