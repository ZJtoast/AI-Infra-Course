#include <hip/hip_runtime.h>

// grid.x=rows, block.x=64, one CDNA wave processes one complete output row.
__global__ void matvec_wave64(const float* a,const float* x,float* y,int cols) {
    int row=blockIdx.x,lane=threadIdx.x;
    float partial=0;
    for(int j=lane;j<cols;j+=64) partial+=a[row*cols+j]*x[j];
    for(int d=32;d>0;d/=2) partial+=__shfl_down(partial,d,64);
    if(lane==0) y[row]=partial;
}
