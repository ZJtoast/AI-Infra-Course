#include <hip/hip_runtime.h>

// Q:[N,D]，K:[S,D]，V:[S,F]；查询第 r 行的逻辑位置等于 past+r。
__global__ void causal_attention(const float* q,const float* k,const float* v,float* out,
                                 int N,int S,int D,int F,int past) {
    int f=blockIdx.x*blockDim.x+threadIdx.x, r=blockIdx.y;
    if (r>=N || f>=F) return;
    float m=-__builtin_huge_valf(), den=0, num=0;
    for(int j=0;j<S && j<=past+r;++j) {
        float s=0;
        for(int d=0;d<D;++d) s += q[r*D+d]*k[j*D+d];
        s *= rsqrtf(float(D));
        float next=fmaxf(m,s), keep=expf(m-next), w=expf(s-next);
        num = num*keep + w*v[j*F+f];
        den = den*keep + w;
        m = next;
    }
    out[r*F+f]=num/den;
}
