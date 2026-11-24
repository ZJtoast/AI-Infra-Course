// hipcc -O2 -std=c++17 row_variance.hip -o row_variance
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Moments {int n; float mean, m2;};

__device__ Moments merge(Moments a, Moments b) {
    if (!a.n) return b;
    if (!b.n) return a;
    const int n=a.n+b.n;
    const float delta=b.mean-a.mean;
    return {n,a.mean+delta*(float(b.n)/n),
            a.m2+b.m2+delta*delta*(float(a.n)*b.n/n)};
}

__global__ void row_variance(const float* x,float* out,int width) {
    __shared__ Moments tree[128];
    int r=blockIdx.x, lane=threadIdx.x;
    Moments local{0,0,0};
    for(int c=lane;c<width;c+=128) local=merge(local,{1,x[r*width+c],0});
    tree[lane]=local;
    __syncthreads();
    for(int step=64;step;step/=2){
        if(lane<step) tree[lane]=merge(tree[lane],tree[lane+step]);
        __syncthreads();
    }
    if(!lane) {
        out[2*r]=tree[0].mean;
        out[2*r+1]=tree[0].n ? tree[0].m2/tree[0].n : 0.f;
    }
}

void check(hipError_t e) {
    if(e!=hipSuccess) { std::fprintf(stderr,"%s\n",hipGetErrorString(e)); std::exit(1); }
}

int main(){
    for(int width: {0,1,4,129,259}) {
        std::vector<float> x(std::max(1,2*width)), result(4);
        for(int c=0;c<width;++c) {
            x[c]=10000.0f+2*(c%11);
            x[width+c]=-7.0f;
        }
        float *dx,*dy;
        check(hipMalloc(reinterpret_cast<void**>(&dx),x.size()*sizeof(float)));
        check(hipMalloc(reinterpret_cast<void**>(&dy),4*sizeof(float)));
        check(hipMemcpy(dx,x.data(),x.size()*sizeof(float),hipMemcpyHostToDevice));
        hipLaunchKernelGGL(row_variance,2,128,0,0,dx,dy,width);
        check(hipGetLastError());check(hipDeviceSynchronize());
        check(hipMemcpy(result.data(),dy,4*sizeof(float),hipMemcpyDeviceToHost));
        for(int r=0;r<2;++r) {
            double mean=0,variance=0;
            for(int c=0;c<width;++c) mean+=x[r*width+c];
            if(width) mean/=width;
            for(int c=0;c<width;++c) {double d=x[r*width+c]-mean;variance+=d*d;}
            if(width) variance/=width;
            if(!(std::abs(result[r*2]-mean)<.002 && std::abs(result[r*2+1]-variance)<.01)) return 2;
        }
        check(hipFree(dx));check(hipFree(dy));
        std::printf("two rows of width %d passed\n",width);
    }
}
