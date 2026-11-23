#include <hip/hip_runtime.h>

__global__ void residual_add(const float* x,const float* residual,float* u,int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) u[i]=x[i]+residual[i];
}
// 1 <= cols <= 64；每行独占一个完整 wave，u 还会提供给后一层。
__global__ void residual_rms64(const float* x,const float* residual,const float* w,
                               float* u,float* y,int cols,float eps) {
    int lane=threadIdx.x,i=blockIdx.x*cols+lane;
    float v=lane<cols ? x[i]+residual[i] : 0;
    float sum=v*v;
    for(int off=32;off;off/=2) sum+=__shfl_down(sum,off,64);
    float scale=rsqrtf(__shfl(sum,0,64)/cols+eps);
    if(lane<cols) { u[i]=v; y[i]=v*w[lane]*scale; }
}
// 有限分数，1 <= cols <= 64；第四项等填充槽对分母贡献 0。
__global__ void softmax_wave64(const float* x,float* y,int cols) {
    int lane=threadIdx.x,i=blockIdx.x*cols+lane;
    float v=lane<cols ? x[i] : -__builtin_huge_valf();
    float m=v;
    for(int off=32;off;off/=2) m=fmaxf(m,__shfl_down(m,off,64));
    float w=expf(v-__shfl(m,0,64)),sum=w;
    for(int off=32;off;off/=2) sum+=__shfl_down(sum,off,64);
    sum=__shfl(sum,0,64);
    if(lane<cols) y[i]=w/sum;
}
