#include <hip/hip_runtime.h>

// 行主序 float 输入与 int8 输出；每行按连续 64 个特征分组。
__device__ inline signed char q8_code(float v, float scale) {
    float mag=floorf(fabsf(v/scale)+0.5f);
    float rounded=v<0 ? -mag : mag;
    return static_cast<signed char>(fmaxf(-127,fminf(127,rounded)));
}

// 基线第一遍：一个线程扫完整组，得到后续要用的步长。
__global__ void scales_serial(const float* x,float* scales,int cols) {
    int group=blockIdx.x, row=blockIdx.y;
    if(threadIdx.x!=0) return;
    float maxabs=0;
    for(int j=64*group;j<cols && j<64*(group+1);++j)
        maxabs=fmaxf(maxabs,fabsf(x[row*cols+j]));
    scales[row*gridDim.x+group]=maxabs>0 ? maxabs/127.0f : 1;
}
__global__ void quant_pass(const float* x,const float* scales,signed char* codes,int cols) {
    int row=blockIdx.y, j=64*blockIdx.x+threadIdx.x;
    if(j<cols) codes[row*cols+j]=q8_code(x[row*cols+j],scales[row*gridDim.x+blockIdx.x]);
}

// gfx90a/gfx942，launch block=64，一个完整 wave 处理一组。
__global__ void quant_wave64(const float* x,signed char* codes,float* scales,int cols) {
    int lane=threadIdx.x, row=blockIdx.y, g=blockIdx.x, j=64*g+lane;
    float v=j<cols ? x[row*cols+j] : 0;
    float a=fabsf(v);
    for(int d=32;d>0;d/=2) a=fmaxf(a,__shfl_down(a,d,64));
    a=__shfl(a,0,64);                              // 组内每个有效值都用相同步长
    float s=a>0 ? a/127.0f : 1;
    if(lane==0) scales[row*gridDim.x+g]=s;
    if(j<cols) codes[row*cols+j]=q8_code(v,s);
}
__global__ void dequant_group(const signed char* codes,const float* scales,float* y,int cols) {
    int row=blockIdx.y, g=blockIdx.x, j=64*g+threadIdx.x;
    if(j<cols) y[row*cols+j]=float(codes[row*cols+j])*scales[row*gridDim.x+g];
}
