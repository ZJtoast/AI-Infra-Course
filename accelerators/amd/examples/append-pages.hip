#include <hip/hip_runtime.h>

// 本例处理一个请求。src:[new_tokens,F]，cache:[physical_pages,page_size,F]。
__global__ void append_pages(const float* src,const int* table,float* cache,
                             int new_tokens,int F,int page_size,int prior) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=new_tokens*F) return;
    int token=i/F, feature=i%F, position=prior+token;
    int page=table[position/page_size], slot=position%page_size;
    cache[(page*page_size+slot)*F+feature]=src[i];
}
