#include "operator-check.hpp"
#include "mfma-block.hip"
#include "attention-block.hip"
#include "causal-attention.hip"
#include "quant-group.hip"
#include "route-stable.hip"
#include "scan.hip"
#include "scan-cooperation.hip"
#include "transpose-layout.hip"
#include <cstring>

// 在默认流上排入同一实现多次，先预热，再用同一流上的 GPU 事件计时。
template<class Run>
void time_case(const char* name,Run run,size_t useful_bytes=0) {
    for(int i=0;i<3;++i) run();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t start,stop;
    HIP_CHECK(hipEventCreate(&start));HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start,0));
    for(int i=0;i<20;++i) run();
    HIP_CHECK(hipEventRecord(stop,0));HIP_CHECK(hipEventSynchronize(stop));
    float ms=0;HIP_CHECK(hipEventElapsedTime(&ms,start,stop));ms/=20;
    std::printf("%-22s %.6f ms",name,ms);
    if(useful_bytes) std::printf("  %.3f useful GB/s", double(useful_bytes)/(ms*1e6));
    std::puts("");HIP_CHECK(hipEventDestroy(start));HIP_CHECK(hipEventDestroy(stop));
}
std::vector<float> sample(size_t n) {
    std::vector<float> v(n);for(size_t i=0;i<n;++i)v[i]=float(int(i%19)-9)/16;return v;
}
int main(int argc,char** argv) {
    if(argc!=2){std::fprintf(stderr,"benchmark gemm|attention|quant|route|scan|transpose\n");return 1;}
    const char* task=argv[1];
    if(!std::strcmp(task,"gemm")){
        const int M=256,N=256,K=256;Buffer<float> a(sample(M*K)),b(sample(K*N)),c(M*N);
        dim3 g(N/32,M/16);
        time_case("direct MFMA",[&]{LAUNCH(mfma_direct,g,128,a.ptr,b.ptr,c.ptr,M,N,K);});
        time_case("LDS MFMA",[&]{LAUNCH(mfma_block,g,128,a.ptr,b.ptr,c.ptr,M,N,K);});
    }else if(!std::strcmp(task,"attention")){
        const int R=32,S=512;Buffer<float> q(sample(R*4)),k(sample(S*4)),v(sample(S*4)),out(R*4);
        time_case("per-feature SDPA",[&]{LAUNCH(causal_attention,dim3(1,R),64,q.ptr,k.ptr,v.ptr,out.ptr,R,S,4,4,S-R);});
        time_case("4-query SDPA",[&]{LAUNCH(attention_block4,(R+3)/4,64,q.ptr,k.ptr,v.ptr,out.ptr,R,S,S-R);});
    }else if(!std::strcmp(task,"quant")){
        const int rows=4096,cols=256;dim3 g(cols/64,rows);
        Buffer<float> x(sample(rows*cols)),s(rows*(cols/64));Buffer<signed char> q(rows*cols);
        time_case("two-pass INT8",[&]{
            LAUNCH(scales_serial,g,1,x.ptr,s.ptr,cols);
            LAUNCH(quant_pass,g,64,x.ptr,s.ptr,q.ptr,cols);
        });
        time_case("fused INT8",[&]{LAUNCH(quant_wave64,g,64,x.ptr,q.ptr,s.ptr,cols);});
    }else if(!std::strcmp(task,"route")){
        const int n=2048,E=8,tiles=n/64;std::vector<int> tags(n);
        for(int i=0;i<n;++i)tags[i]=(i*7+i/3)%E;
        Buffer<int> e(tags),packed(n),counts(tiles*E),bases(tiles*E),starts(E+1);
        time_case("all-pairs rank",[&]{LAUNCH(pack_reference,tiles,64,e.ptr,packed.ptr,n);});
        time_case("count/prefix/scatter",[&]{
            LAUNCH(count_route_tiles,tiles,64,e.ptr,counts.ptr,n,E);
            LAUNCH(route_bases,1,1,counts.ptr,bases.ptr,starts.ptr,tiles,E);
            LAUNCH(scatter_routes,tiles,64,e.ptr,bases.ptr,packed.ptr,n,E);
        });
    }else if(!std::strcmp(task,"scan")){
        const int n=65536,tiles=n/64;
        Buffer<int> x(std::vector<int>(n,1)),y(n),totals(tiles);
        time_case("local LDS scan",[&]{LAUNCH(scan_tiles<64>,tiles,64,x.ptr,y.ptr,totals.ptr,n);});
        time_case("local tree scan",[&]{LAUNCH(scan_tree64,tiles,64,x.ptr,y.ptr,totals.ptr,n);});
        time_case("local wave scan",[&]{LAUNCH(scan_shuffle64,tiles,64,x.ptr,y.ptr,totals.ptr,n);});
    }else if(!std::strcmp(task,"transpose")){
        const int rows=1024,cols=1024;dim3 g(cols/32,rows/32),b(32,8);
        Buffer<float> x(sample(rows*cols)),y(rows*cols);size_t bytes=2ULL*rows*cols*sizeof(float);
        time_case("stride stores",[&]{LAUNCH(transpose_naive,g,b,x.ptr,y.ptr,rows,cols);},bytes);
        time_case("LDS pitch 32",[&]{LAUNCH(transpose32<32>,g,b,x.ptr,y.ptr,rows,cols);},bytes);
        time_case("LDS pitch 33",[&]{LAUNCH(transpose32<33>,g,b,x.ptr,y.ptr,rows,cols);},bytes);
    }else{return 1;}
}
