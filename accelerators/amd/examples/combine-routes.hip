#include <hip/hip_runtime.h>

// val:[routes,F] 按专家打包；dst_id 和 gate 各长 routes。发起前将 out 清零。
__global__ void combine_routes(const float* val,const int* dst_id,const float* gate,
                               float* out,int routes,int F) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=routes*F) return;
    int route=i/F, feature=i%F;
    float contribution=gate[route]*val[i];
    atomicAdd(out+dst_id[route]*F+feature,contribution);
}
