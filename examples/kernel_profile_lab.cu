// CUDA profiling lab: run one workload per process, e.g. ./kernel_lab transpose
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
template<typename T> struct Buffer {
    T* p;
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n*sizeof(T))); }
    Buffer(const Buffer&) = delete;
    ~Buffer() { cudaFree(p); }
    void upload(const std::vector<T>& h) { check(cudaMemcpy(p,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice)); }
};
struct Variant { const char* name; std::function<void()> launch; };

static void compare(const std::vector<Variant>& variants, float* out, const std::vector<float>& ref) {
    cudaEvent_t begin,end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    std::vector<float> got(ref.size());
    for (const auto& v : variants) {
        check(cudaMemset(out, 0xff, got.size()*sizeof(float)));  // NaN 标出漏写的输出
        v.launch(); check(cudaGetLastError());
        check(cudaMemcpy(got.data(),out,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for (size_t i=0; i<got.size(); ++i)
            if (!std::isfinite(got[i]) || std::abs(got[i]-ref[i]) > 2e-6f+2e-5f*std::abs(ref[i])) {
                std::fprintf(stderr,"%s [%zu] got %g expected %g\n",v.name,i,got[i],ref[i]); std::exit(2);
            }
    }
    constexpr int launches=20;
    for (unsigned round=0; round<5; ++round)
        for (unsigned j=0; j<variants.size(); ++j) {
            const auto& v=variants[(round+j)%variants.size()]; // 轮换先后顺序
            v.launch(); check(cudaEventRecord(begin));
            for (int i=0; i<launches; ++i) v.launch();
            check(cudaGetLastError()); check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
            float ms; check(cudaEventElapsedTime(&ms,begin,end));
            std::printf("%s,%u,%.4f\n",v.name,round+1,ms*1000/launches);
        }
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
}
__global__ void transpose_direct(const float* a,float* b,int n) {
    int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y*32+threadIdx.y;
    for (int j=0;j<32;j+=8) b[x*n+y+j]=a[(y+j)*n+x];
}
template<int pitch> __global__ void transpose_tile(const float* a,float* b,int n) {
    __shared__ float tile[32][pitch];
    int x=threadIdx.x, y=threadIdx.y;
    for(int j=0;j<32;j+=8) tile[y+j][x]=a[(blockIdx.y*32+y+j)*n+blockIdx.x*32+x];
    __syncthreads();
    for(int j=0;j<32;j+=8) b[(blockIdx.x*32+y+j)*n+blockIdx.y*32+x]=tile[x][y+j];
}
__global__ void gemm_direct(const float* a,const float* b,float* c,int n) {
    int i=blockIdx.y*16+threadIdx.y,j=blockIdx.x*16+threadIdx.x;
    float s=0;
    for(int k=0;k<n;++k) s=fmaf(a[i*n+k],b[k*n+j],s);
    c[i*n+j]=s;
}
__global__ void gemm_tile(const float* a,const float* b,float* c,int n) {
    __shared__ float left[16][16],right[16][16];
    int y=threadIdx.y,x=threadIdx.x,i=blockIdx.y*16+y,j=blockIdx.x*16+x;
    float s=0;
    for(int base=0;base<n;base+=16) {
        left[y][x]=a[i*n+base+x]; right[y][x]=b[(base+y)*n+j];
        __syncthreads();
        #pragma unroll
        for(int k=0;k<16;++k) s=fmaf(left[y][k],right[k][x],s);
        __syncthreads();
    }
    c[i*n+j]=s;
}
__device__ float warp_sum(float v) {
    for(int d=16; d; d/=2) v+=__shfl_down_sync(0xffffffff,v,d);
    return v;
}
__global__ void reduce_tree(const float* x,float* out) {
    __shared__ float part[256];
    int t=threadIdx.x; part[t]=x[blockIdx.x*256+t]; __syncthreads();
    for(int d=128;d;d/=2) {
        if(t<d) part[t]+=part[t+d];
        __syncthreads();
    }
    if(!t) out[blockIdx.x]=part[0];
}
__global__ void reduce_warp(const float* x,float* out) {
    __shared__ float sums[8];
    int t=threadIdx.x,lane=t%32,warp=t/32;
    float s=warp_sum(x[blockIdx.x*256+t]);
    if(!lane) sums[warp]=s;
    __syncthreads();
    if(!warp) {
        s=warp_sum(lane<8?sums[lane]:0.0f);
        if(!lane) out[blockIdx.x]=s;
    }
}
__global__ void csr_thread(const int* offset,const int* col,const float* x,float* y,int rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows) return;
    float s=0;
    for(int e=offset[i];e<offset[i+1];++e) s+=x[col[e]];
    y[i]=s;
}
__global__ void csr_warp(const int* offset,const int* col,const float* x,float* y,int rows) {
    int t=blockIdx.x*blockDim.x+threadIdx.x, i=t/32,lane=t%32;
    if(i>=rows) return;
    float s=0;
    for(int e=offset[i]+lane;e<offset[i+1];e+=32) s+=x[col[e]];
    s=warp_sum(s); if(!lane) y[i]=s;
}
__global__ void softmax_serial(const float* x,float* y,int rows,int cols) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows) return;
    x+=i*cols; y+=i*cols; float m=-INFINITY,den=0;
    for(int j=0;j<cols;++j) m=fmaxf(m,x[j]);
    for(int j=0;j<cols;++j) den+=expf(x[j]-m);
    for(int j=0;j<cols;++j) y[j]=expf(x[j]-m)/den;
}
__global__ void softmax_warp32(const float* x,float* y,int rows) {
    int t=blockIdx.x*blockDim.x+threadIdx.x,i=t/32,lane=t%32; if(i>=rows) return;
    float value=x[t],m=value;
    for(int d=16;d;d/=2) m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d));
    float ex=expf(value-m),den=warp_sum(ex);
    y[t]=ex/__shfl_sync(0xffffffff,den,0);
}
static std::vector<float> values(int n,int period) {
    std::vector<float> v(n);
    for(int i=0;i<n;++i) v[i]=float(i%period-period/2);
    return v;
}
int main(int argc,char** argv) {
    std::string mode=argc==2?argv[1]:"";
    if(mode=="transpose" || mode=="gemm") {
        int n=mode=="transpose"?2048:512;
        auto a=values(n*n,13); std::vector<float> ref(n*n);
        Buffer<float> A(n*n),Y(n*n); A.upload(a);
        if(mode=="transpose") {
            for(int i=0;i<n;++i) for(int j=0;j<n;++j) ref[j*n+i]=a[i*n+j];
            dim3 g(n/32,n/32),b(32,8);
            compare({{"direct",[&]{transpose_direct<<<g,b>>>(A.p,Y.p,n);}},
                     {"tile32",[&]{transpose_tile<32><<<g,b>>>(A.p,Y.p,n);}},
                     {"tile33",[&]{transpose_tile<33><<<g,b>>>(A.p,Y.p,n);}}},Y.p,ref);
        } else {
            auto b=values(n*n,7); Buffer<float> B(n*n); B.upload(b);
            for(int i=0;i<n;++i) for(int k=0;k<n;++k) for(int j=0;j<n;++j)
                ref[i*n+j]+=a[i*n+k]*b[k*n+j];
            dim3 grid(n/16,n/16),block(16,16);
            compare({{"direct",[&]{gemm_direct<<<grid,block>>>(A.p,B.p,Y.p,n);}},
                     {"tile",[&]{gemm_tile<<<grid,block>>>(A.p,B.p,Y.p,n);}}},Y.p,ref);
        }
    } else if(mode=="reduction") {
        int blocks=4096; auto x=values(blocks*256,23); std::vector<float> ref(blocks);
        for(int i=0;i<blocks*256;++i) ref[i/256]+=x[i];
        Buffer<float> X(x.size()),Y(blocks); X.upload(x);
        compare({{"tree",[&]{reduce_tree<<<blocks,256>>>(X.p,Y.p);}},
                 {"warp",[&]{reduce_warp<<<blocks,256>>>(X.p,Y.p);}}},Y.p,ref);
    } else if(mode=="graph-short" || mode=="graph-skew") {
        int rows=32768; std::vector<int> off(rows+1),col;
        auto x=values(rows,13); std::vector<float> ref(rows);
        for(int i=0;i<rows;++i) {
            int degree=(mode=="graph-skew" && i%32==0)?256:4;
            for(int j=0;j<degree;++j) {
                int v=(i*17+j*101)%rows; col.push_back(v); ref[i]+=x[v];
            }
            off[i+1]=int(col.size());
        }
        Buffer<int> O(off.size()),C(col.size()); Buffer<float> X(rows),Y(rows);
        O.upload(off); C.upload(col); X.upload(x);
        compare({{"thread",[&]{csr_thread<<<rows/128,128>>>(O.p,C.p,X.p,Y.p,rows);}},
                 {"warp",[&]{csr_warp<<<rows/4,128>>>(O.p,C.p,X.p,Y.p,rows);}}},Y.p,ref);
    } else if(mode=="softmax") {
        int rows=32768,cols=32; auto x=values(rows*cols,19); std::vector<float> ref(x.size());
        for(int i=0;i<rows;++i) {
            auto first=x.begin()+i*cols; float m=*std::max_element(first,first+cols);
            double den=0;
            for(int j=0;j<cols;++j) den+=std::exp(double(first[j]-m));
            for(int j=0;j<cols;++j) ref[i*cols+j]=float(std::exp(double(first[j]-m))/den);
        }
        Buffer<float> X(x.size()),Y(x.size()); X.upload(x);
        compare({{"serial",[&]{softmax_serial<<<rows/128,128>>>(X.p,Y.p,rows,cols);}},
                 {"warp32",[&]{softmax_warp32<<<rows/4,128>>>(X.p,Y.p,rows);}}},Y.p,ref);
    } else {
        std::fprintf(stderr,"usage: kernel_lab gemm|transpose|reduction|graph-short|graph-skew|softmax\n");
        return 1;
    }
}
