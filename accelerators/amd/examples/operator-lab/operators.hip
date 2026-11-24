#include <hip/hip_runtime.h>
#include <cmath>

// Fixed teaching tile: a row is 128 FP32 features; there are eight rows.
__global__ void rope(const float* x, const float* co, const float* si, float* y) {
    const int r=blockIdx.x, c=threadIdx.x, i=r*256+c, angle=r*128+c;
    const float a=x[i], b=x[i+128];
    y[i]=a*co[angle]-b*si[angle];
    y[i+128]=b*co[angle]+a*si[angle];
}

// All 128 threads enter every barrier, including threads whose value is already in the sum.
__device__ float block_sum(float value, float* shared) {
    int c=threadIdx.x;
    shared[c]=value;
    __syncthreads();
    for (int step=64; step; step/=2) {
        if (c<step) shared[c]+=shared[c+step];
        __syncthreads();
    }
    const float sum=shared[0];
    __syncthreads();  // Release the slot only after all lanes have read the result.
    return sum;
}

__global__ void layernorm(const float* x,const float* gamma,const float* bias,float* y) {
    __shared__ float shared[128];
    const int c=threadIdx.x, i=blockIdx.x*128+c;
    const float delta=x[i]-block_sum(x[i],shared)/128.0f;
    const float variance=block_sum(delta*delta,shared)/128.0f;
    y[i]=delta*rsqrtf(variance+1e-5f)*gamma[c]+bias[c];
}

__global__ void smooth_l1_grad(const float* x,const float* target,const float* grad,float* y) {
    const int i=blockIdx.x*128+threadIdx.x;
    const float slope=fminf(1.0f,fmaxf(-1.0f,(x[i]-target[i])*0.5f));
    y[i]=grad[i]*slope;
}

__global__ void space_to_depth(const float* image,float* y) {
    const int pixel=blockIdx.x, c=threadIdx.x;
    const int r=pixel/4, col=pixel%4;
    const int patch=(r/2)*2+col/2, offset=(r%2)*2+col%2;
    y[(patch*4+offset)*128+c]=image[pixel*128+c];
}

__global__ void bag_max(const float* features,const int* ids,float* y) {
    const int r=blockIdx.x, c=threadIdx.x;
    float best=-INFINITY;
    bool seen=false;
    for (int j=0; j<3; ++j) {
        const int n=ids[r*3+j];
        if (n>=0) { best=fmaxf(best,features[n*128+c]); seen=true; }
    }
    y[r*128+c]=seen ? best : 0.0f;
}

__global__ void moe_combine(const float* slots,const int* indices,const float* weights,float* y) {
    const int r=blockIdx.x, c=threadIdx.x, p=r*2;
    const float a=slots[indices[p]*128+c], b=slots[indices[p+1]*128+c];
    y[r*128+c]=weights[p]*a+weights[p+1]*b;
}

__global__ void attention_dv(const float* prob,const float* dy,float* dv) {
    const int key=blockIdx.x, c=threadIdx.x;
    float acc=0;
    for (int q=key; q<8; ++q) acc+=prob[q*8+key]*dy[q*128+c];
    dv[key*128+c]=acc;
}

__global__ void lora(const float* x,const float* down,const float* up,const float* base,float* y) {
    __shared__ float shared[128];
    const int c=threadIdx.x, i=blockIdx.x*128+c;
    const float a=block_sum(x[i]*down[c],shared);
    const float b=block_sum(x[i]*down[128+c],shared);
    y[i]=base[i]+0.5f*(a*up[c]+b*up[128+c]);
}

// Host adapter for check_device.py. Inputs and output are existing device allocations.
extern "C" int run_operator(int op,void** args,void* result) {
    auto f=[&](int i){return static_cast<const float*>(args[i]);};
    auto idx=[&](int i){return static_cast<const int*>(args[i]);};
    auto y=static_cast<float*>(result);
    switch(op) {
        case 0: hipLaunchKernelGGL(rope,8,128,0,0,f(0),f(1),f(2),y); break;
        case 1: hipLaunchKernelGGL(layernorm,8,128,0,0,f(0),f(1),f(2),y); break;
        case 2: hipLaunchKernelGGL(smooth_l1_grad,8,128,0,0,f(0),f(1),f(2),y); break;
        case 3: hipLaunchKernelGGL(space_to_depth,16,128,0,0,f(0),y); break;
        case 4: hipLaunchKernelGGL(bag_max,8,128,0,0,f(0),idx(1),y); break;
        case 5: hipLaunchKernelGGL(moe_combine,8,128,0,0,f(0),idx(1),f(2),y); break;
        case 6: hipLaunchKernelGGL(attention_dv,8,128,0,0,f(0),f(1),y); break;
        case 7: hipLaunchKernelGGL(lora,8,128,0,0,f(0),f(1),f(2),f(3),y); break;
        default: return -1;
    }
    hipError_t status=hipGetLastError();
    return status==hipSuccess ? int(hipDeviceSynchronize()) : int(status);
}
