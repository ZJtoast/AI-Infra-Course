#include <hip/hip_runtime.h>

// expert[r] 已在 [0,E) 中。两个算法均按专家分组，组内保持原 route 顺序。
__global__ void pack_reference(const int* expert,int* packed,int n) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=n) return;
    int target=expert[r], rank=0;
    for(int i=0;i<n;++i)
        rank+=(expert[i]<target || (expert[i]==target && i<r));
    packed[rank]=r;
}
__global__ void count_route_tiles(const int* expert,int* counts,int n,int E) {
    int lane=threadIdx.x, r=blockIdx.x*64+lane;
    int key=r<n ? expert[r] : -1;
    for(int e=0;e<E;++e) {
        unsigned long long mask=__ballot(key==e);
        if(lane==0) counts[blockIdx.x*E+e]=__popcll(mask);
    }
}
// 网格 (1)，block (1)；小专家表的串行前缀，便于直接对照 starts 与 bases。
__global__ void route_bases(const int* counts,int* bases,int* starts,int tiles,int E) {
    int pos=0;
    for(int e=0;e<E;++e) {
        starts[e]=pos;
        for(int tile=0;tile<tiles;++tile) {
            int i=tile*E+e;
            bases[i]=pos;
            pos+=counts[i];
        }
    }
    starts[E]=pos;
}
__global__ void scatter_routes(const int* expert,const int* bases,int* packed,int n,int E) {
    int lane=threadIdx.x, r=blockIdx.x*64+lane;
    int key=r<n ? expert[r] : -1;
    unsigned long long lower=(1ULL<<lane)-1ULL;
    for(int e=0;e<E;++e) {
        unsigned long long peers=__ballot(key==e);
        if(key==e) {
            int within=__popcll(peers & lower);
            packed[bases[blockIdx.x*E+e]+within]=r;
        }
    }
}

__global__ void inverse_slots(const int* packed,int* slot,int routes) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p<routes) slot[packed[p]]=p;
}
// 原路由编号为 token*K + 分支号；专家结果与 gate 分别按打包、原路由次序存放。
__global__ void combine_by_token(const float* values,const float* gate,const int* slot,
                                 float* y,int tokens,int K,int F) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=tokens*F) return;
    int token=i/F, f=i%F;
    float sum=0;
    for(int branch=0;branch<K;++branch) {
        int r=token*K+branch;
        sum+=gate[r]*values[slot[r]*F+f];
    }
    y[i]=sum;
}
