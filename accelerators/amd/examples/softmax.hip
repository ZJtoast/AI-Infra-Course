#include <hip/hip_runtime.h>

// 每块一行，B 为 2 的幂，行中有 n 个有限分数；y 与输入形状相同。
template<int B>
__global__ void softmax_rows(const float* x, float* y, int n) {
    __shared__ float v[B];
    int t=threadIdx.x, base=blockIdx.x*n;
    float m=-__builtin_huge_valf();
    for (int j=t;j<n;j+=B) m=fmaxf(m,x[base+j]);
    v[t]=m; __syncthreads();
    for (int s=B/2;s;s/=2) {
        if (t<s) v[t]=fmaxf(v[t],v[t+s]);
        __syncthreads();
    }
    m=v[0]; __syncthreads(); // 先让大家读到行最大值，再复用同一块 LDS
    float z=0;
    for (int j=t;j<n;j+=B) z+=expf(x[base+j]-m);
    v[t]=z; __syncthreads();
    for (int s=B/2;s;s/=2) {
        if (t<s) v[t]+=v[t+s];
        __syncthreads();
    }
    for (int j=t;j<n;j+=B) y[base+j]=expf(x[base+j]-m)/v[0];
}
