#include <hip/hip_runtime.h>

// block=(32,8); grid=(ceil(cols/32),ceil(rows/32)); one block moves a 32x32 tile.
__global__ void transpose_naive(const float* x,float* y,int rows,int cols) {
    int c=32*blockIdx.x+threadIdx.x;
    for(int j=0;j<32;j+=8) {
        int r=32*blockIdx.y+threadIdx.y+j;
        if(r<rows && c<cols) y[c*rows+r]=x[r*cols+c];
    }
}
template<int Pitch>
__global__ void transpose32(const float* x,float* y,int rows,int cols) {
    __shared__ float tile[32][Pitch];                 // Pitch=32 或 33，以 FP32 为单位
    int u=threadIdx.x, v=threadIdx.y;
    int in_col=32*blockIdx.x+u, out_col=32*blockIdx.y+u;
    for(int j=0;j<32;j+=8) {
        int in_row=32*blockIdx.y+v+j;
        if(in_row<rows && in_col<cols) tile[v+j][u]=x[in_row*cols+in_col];
    }
    __syncthreads();
    for(int j=0;j<32;j+=8) {
        int out_row=32*blockIdx.x+v+j;
        if(out_row<cols && out_col<rows) y[out_row*rows+out_col]=tile[u][v+j];
    }
}
