#include <hip/hip_runtime.h>

// 启动 block.x 必须是硬件 warpSize 的整数倍。最后不足一组的输入以零参加。
__global__ void wave_sum(const float* in, float* partial, int n) {
    int local=threadIdx.x, i=blockIdx.x*blockDim.x+local;
    int lane=local%warpSize, wave=local/warpSize;
    float x=i<n ? in[i] : 0.0f;
    for (int d=warpSize/2; d>0; d/=2) x+=__shfl_down(x,d,warpSize);
    if(lane==0) partial[blockIdx.x*(blockDim.x/warpSize)+wave]=x;
}
