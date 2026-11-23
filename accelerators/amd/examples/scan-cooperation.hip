#include <hip/hip_runtime.h>

// grid=ceil(n/64), block=64。两种内核都输出块内包含式前缀及块总数。
__global__ void scan_tree64(const int* x,int* out,int* totals,int n) {
    __shared__ int tree[64];
    int t=threadIdx.x, i=blockIdx.x*64+t, own=i<n ? x[i] : 0;
    tree[t]=own;
    __syncthreads();
    for(int step=1;step<64;step*=2) {
        int right=(t+1)*2*step-1;
        if(right<64) tree[right]+=tree[right-step];
        __syncthreads();
    }
    if(t==0) { totals[blockIdx.x]=tree[63]; tree[63]=0; }
    __syncthreads();
    for(int step=32;step>0;step/=2) {
        int right=(t+1)*2*step-1;
        if(right<64) {
            int left=tree[right-step];
            tree[right-step]=tree[right];             // 左子树沿用父进位
            tree[right]+=left;                        // 右子树再加左子树总数
        }
        __syncthreads();
    }
    if(i<n) out[i]=tree[t]+own;
}
__global__ void scan_shuffle64(const int* x,int* out,int* totals,int n) {
    int lane=threadIdx.x, i=blockIdx.x*64+lane;
    int sum=i<n ? x[i] : 0;
    for(int d=1;d<64;d*=2) {
        int left=__shfl_up(sum,d,64);
        if(lane>=d) sum+=left;
    }
    if(i<n) out[i]=sum;
    if(lane==63) totals[blockIdx.x]=sum;
}
