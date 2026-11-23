#include <hip/hip_runtime.h>

// grid=(max_length,requests), block.x>=1；每块沿特征轴处理一个逻辑位置。
__global__ void gather_kv_batch(const float* pages,const int* table,const int* length,
                                float* out,int page_size,int table_width,int max_length,int features) {
    int pos=blockIdx.x, request=blockIdx.y;
    bool live=pos<length[request];
    int row=0;
    if(live) {
        int physical=table[request*table_width+pos/page_size];
        row=physical*page_size+(pos%page_size);
    }
    for(int f=threadIdx.x;f<features;f+=blockDim.x) {
        int dest=(request*max_length+pos)*features+f;
        out[dest]=live ? pages[row*features+f] : 0;
    }
}
