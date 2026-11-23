#include <hip/hip_runtime.h>

// 单图 N=1；数组顺序为 CHW、O×C×K×K、O×OH×OW。grid.z = 输出通道数。
__global__ void conv_chw(const float* in, const float* weight, float* out,
                         int C, int H, int W, int K, int stride, int pad, int OH, int OW) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y, oc = blockIdx.z;
    if (ox >= OW || oy >= OH) return;
    float acc = 0;
    for (int c = 0; c < C; ++c)
        for (int ky = 0; ky < K; ++ky)
            for (int kx = 0; kx < K; ++kx) {
                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                    acc += in[(c*H+iy)*W+ix] * weight[((oc*C+c)*K+ky)*K+kx];
            }
    out[(oc*OH+oy)*OW+ox] = acc;
}
