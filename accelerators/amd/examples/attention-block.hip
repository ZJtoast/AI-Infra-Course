#include <hip/hip_runtime.h>

// block=64, grid=ceil(rows/4), D=F=4; q position is past+r, keys start at 0.
__global__ void attention_block4(const float* q, const float* k, const float* v,
                                 float* out, int rows, int keys, int past) {
    __shared__ float qs[4][4], ks[4][4], vs[4][4], w[4][4];
    __shared__ float m[4], den[4], keep[4];
    const int t=threadIdx.x, r=t/4, f=t%4, r0=blockIdx.x*4;
    float num=0;
    if(t<16) qs[r][f]=r0+r<rows ? q[(r0+r)*4+f] : 0;
    if(t<4) { m[t]=-__builtin_huge_valf(); den[t]=0; }
    __syncthreads();
    for(int j0=0;j0<keys && j0<=past+r0+3;j0+=4) {
        if(t<16) {
            ks[r][f]=j0+r<keys ? k[(j0+r)*4+f] : 0;
            vs[r][f]=j0+r<keys ? v[(j0+r)*4+f] : 0;
        }
        __syncthreads();
        if(t<16) {
            float dot=0;
            for(int d=0;d<4;++d) dot+=qs[r][d]*ks[f][d];
            w[r][f]=(r0+r<rows && j0+f<keys && j0+f<=past+r0+r)
                      ? dot*0.5f : -__builtin_huge_valf();
        }
        __syncthreads();
        if(t<4) {
            float next=m[t];
            for(int j=0;j<4;++j) next=fmaxf(next,w[t][j]);
            bool live=next != -__builtin_huge_valf();
            keep[t]=live ? expf(m[t]-next) : 0;
            den[t]*=keep[t];
            for(int j=0;j<4;++j) {
                w[t][j]=live ? expf(w[t][j]-next) : 0;
                den[t]+=w[t][j];
            }
            m[t]=next;
        }
        __syncthreads();
        if(t<16) {
            num*=keep[r];
            for(int j=0;j<4;++j) num+=w[r][j]*vs[j][f];
        }
        __syncthreads();                          // 下一块即将覆盖 K、V 和权重
    }
    if(t<16 && r0+r<rows) out[(r0+r)*4+f] = num/den[r];
}
