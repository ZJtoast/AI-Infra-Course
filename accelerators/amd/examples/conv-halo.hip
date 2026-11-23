#include <hip/hip_runtime.h>

// 单通道 2×2 互相关，stride=1、无填充；block=(8,8)，输出=(H-1,W-1)。
__global__ void conv_halo2(const float* x,const float* weight,float* out,int H,int W) {
    __shared__ float image[9][9];
    int u=threadIdx.x,v=threadIdx.y, x0=8*blockIdx.x,y0=8*blockIdx.y;
    for(int i=v*8+u;i<81;i+=64) {                    // 右边与下边的 halo 也搬进来
        int r=y0+i/9,c=x0+i%9;
        image[i/9][i%9]=r<H && c<W ? x[r*W+c] : 0;
    }
    __syncthreads();
    int r=y0+v,c=x0+u;
    if(r<H-1 && c<W-1)
        out[r*(W-1)+c]=weight[0]*image[v][u]+weight[1]*image[v][u+1]
                     +weight[2]*image[v+1][u]+weight[3]*image[v+1][u+1];
}
// 单通道最大池化，2×2 窗口，步幅为 2，尾部不足一整个窗口时舍去。
__global__ void maxpool_stride2(const float* x,float* out,int H,int W) {
    int c=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
    if(r>=H/2 || c>=W/2) return;
    int p=2*r*W+2*c;
    float top=fmaxf(x[p],x[p+1]),bottom=fmaxf(x[p+W],x[p+W+1]);
    out[r*(W/2)+c]=fmaxf(top,bottom);
}
