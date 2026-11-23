#include <hip/hip_runtime.h>

// 同一 grid 和 block；每个 wave 独立装载操作数，供 LDS 版本对照。
__global__ void mfma_direct(const float* a, const float* b, float* c, int M, int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__)
    using Acc4 = float __attribute__((ext_vector_type(4)));
    int t=threadIdx.x, x=t%16, y=(t%64)/16;
    int r0=16*blockIdx.y, col=32*blockIdx.x+16*(t/64)+x;
    Acc4 acc={0,0,0,0};
    for(int k0=0;k0<K;k0+=4) {
        int k=k0+y;
        float av=(r0+x<M && k<K) ? a[(r0+x)*K+k] : 0;
        float bv=(k<K && col<N) ? b[k*N+col] : 0;
        acc=__builtin_amdgcn_mfma_f32_16x16x4f32(av,bv,acc,0,0,0);
    }
    for(int i=0;i<4;++i) {
        int row=r0+4*y+i;
        if(row<M && col<N) c[row*N+col]=acc[i];
    }
#endif
}

// grid=(ceil(N/32),ceil(M/16)), block=128, row-major A[M,K], B[K,N], C[M,N].
__global__ void mfma_block(const float* a, const float* b, float* c, int M, int N, int K) {
#if defined(__HIP_DEVICE_COMPILE__)
    using Acc4 = float __attribute__((ext_vector_type(4)));
    __shared__ float tile_a[16][4];
    __shared__ float tile_b[4][32];
    const int t = threadIdx.x, wave = t/64, lane = t%64;
    const int x = lane%16, y = lane/16;
    const int r0 = 16*blockIdx.y, c0 = 32*blockIdx.x;
    Acc4 acc = {0,0,0,0};
    for (int k0=0; k0<K; k0+=4) {
        if (t<64) {                                  // 两个 wave 共用同一片 A
            int r=r0+t/4, k=k0+t%4;
            tile_a[t/4][t%4] = r<M && k<K ? a[r*K+k] : 0;
        }
        int k=k0+t/32, col=c0+t%32;                  // B 的一行拆为 32 个连续字
        tile_b[t/32][t%32] = k<K && col<N ? b[k*N+col] : 0;
        __syncthreads();
        float av=tile_a[x][y], bv=tile_b[y][16*wave+x];
        acc=__builtin_amdgcn_mfma_f32_16x16x4f32(av,bv,acc,0,0,0);
        __syncthreads();                             // 等两个 wave 都读完，再覆盖 LDS
    }
    for (int reg=0; reg<4; ++reg) {
        int row=r0+4*y+reg, col=c0+16*wave+x;
        if (row<M && col<N) c[row*N+col]=acc[reg];
    }
#endif
}
