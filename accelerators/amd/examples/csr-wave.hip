#include <hip/hip_runtime.h>

// one 64-lane wave per row; features[num_cols,4], out[rows,4].
__global__ void csr_four(const int* ptr,const int* col,const float* w,
                         const float* features,float* out) {
    int row=blockIdx.x,lane=threadIdx.x,channel=lane%4,group=lane/4;
    int first=ptr[row],last=ptr[row+1];
    float sum=0;
    for(int base=first;base<last;base+=16) {
        int edge=base+group;
        int c=channel==0 && edge<last ? col[edge] : 0;
        float weight=channel==0 && edge<last ? w[edge] : 0;
        c=__shfl(c,lane-channel,64);
        weight=__shfl(weight,lane-channel,64);
        if(edge<last) sum+=weight*features[4*c+channel];
    }
    for(int off=32;off>=4;off/=2) sum+=__shfl_down(sum,off,64);
    if(lane<4) out[4*row+channel]=sum;
}
