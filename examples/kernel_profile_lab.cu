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
// Each thread uses the same A element for two independent output columns.
__global__ void gemm_two_columns(const float* a, const float* b, float* c, int n) {
    __shared__ float left[16][16], right[16][32];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.y * 16 + y, col = blockIdx.x * 32 + x;
    float sum0 = 0, sum1 = 0;
    for (int base = 0; base < n; base += 16) {
        left[y][x] = a[row*n + base+x];
        right[y][x] = b[(base+y)*n + col];
        right[y][x+16] = b[(base+y)*n + col+16];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float av = left[y][k];
            sum0 = fmaf(av, right[k][x], sum0);
            sum1 = fmaf(av, right[k][x+16], sum1);
        }
        __syncthreads();
    }
    c[row*n + col] = sum0;
    c[row*n + col+16] = sum1;
}

__global__ void reduce_warp_tail(const float* x, float* out) {
    __shared__ float part[256];
    int t = threadIdx.x;
    part[t] = x[blockIdx.x * 256 + t];
    __syncthreads();
    for (int distance = 128; distance >= 32; distance /= 2) {
        if (t < distance) part[t] += part[t+distance];
        __syncthreads();
    }
    if (t < 32) {
        float sum = warp_sum(part[t]);
        if (!t) out[blockIdx.x] = sum;
    }
}

__global__ void csr_four_lanes(const int* offset, const int* col, const float* x,
                               float* y, int rows) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread / 4, lane = thread % 4;
    float sum = 0;
    if (row < rows) {
        for (int edge = offset[row] + lane; edge < offset[row+1]; edge += 4)
            sum += x[col[edge]];
    }
    // All lanes reach both shuffles. 'width=4' partitions each warp into eight groups.
    for (int delta = 2; delta > 0; delta /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, delta, 4);
    if (row < rows && lane == 0) y[row] = sum;
}

__global__ void softmax_eight_lanes(const float* x, float* y, int rows) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int row = thread / 8, lane = thread % 8;
    // Fixed lab launch has exactly rows*8 threads and rows is divisible by 32.
    int index = row*32 + lane;
    float values[4], maximum = -INFINITY;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        values[j] = x[index + j*8];
        maximum = fmaxf(maximum, values[j]);
    }
    for (int delta = 4; delta > 0; delta /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta, 8));
    float sum = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        values[j] = expf(values[j] - maximum);
        sum += values[j];
    }
    for (int delta = 4; delta > 0; delta /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, delta, 8);
    #pragma unroll
    for (int j = 0; j < 4; ++j) y[index+j*8] = values[j] / sum;
}

// Each thread evolves four independent FP32 values through the same recurrence.
// The group size changes how many independent accumulators are live together.
template<int group>
__global__ void fma_chains(const float* x, float* y, int count, int iterations) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= count) return;
    #pragma unroll 1
    for (int first = 0; first < 4; first += group) {
        float state[group];
        #pragma unroll
        for (int j = 0; j < group; ++j) state[j] = x[(first+j)*count + thread];
        #pragma unroll 1
        for (int i = 0; i < iterations; ++i) {
            #pragma unroll
            for (int j = 0; j < group; ++j) state[j] = fmaf(state[j], 1.000012f, 0.000317f);
        }
        #pragma unroll
        for (int j = 0; j < group; ++j) y[(first+j)*count + thread] = state[j];
    }
}

// One independent 32-bin histogram per block. The final consumer wants a
// histogram per segment, so every block owns its slice in every variant.
template<int variant>
__global__ void segment_histogram(const int* keys, float* counts) {
    constexpr int bins = 32, items = 1024;
    int lane = threadIdx.x % 32;
    float* output = counts + blockIdx.x*bins;
    __shared__ float local[bins];
    float* bins_out = variant == 0 ? output : local;
    if (threadIdx.x < bins) bins_out[threadIdx.x] = 0.0f;
    __syncthreads();
    for (int i = threadIdx.x; i < items; i += blockDim.x) {
        int key = keys[blockIdx.x*items+i];
        if (variant == 2) {
            unsigned peers = __match_any_sync(0xffffffff, key);
            if (lane == __ffs(peers)-1) atomicAdd(bins_out+key, float(__popc(peers)));
        } else {
            atomicAdd(bins_out+key, 1.0f);
        }
    }
    if (variant != 0) {
        __syncthreads();
        if (threadIdx.x < bins) output[threadIdx.x] = local[threadIdx.x];
    }
}

#include <cuda_fp16.h>
#include <mma.h>
#ifdef PROFILE_CUBLAS
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

static void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,"cuBLAS status %d\n",int(status));
        std::exit(1);
    }
}
#endif

// All operands use row-major storage. The host pads each leading dimension to
// a multiple of 16 half values; cudaMalloc supplies the base alignment.
__global__ void half_scalar_gemm(const half* a, const half* b, float* c,
                                 int rows, int cols, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;
    float sum = 0.0f;
    for (int i = 0; i < k; ++i)
        sum = fmaf(__half2float(a[row*k+i]), __half2float(b[i*cols+col]), sum);
    c[row*cols+col] = sum;
}

template<int tiles>
__global__ void warp_tile_gemm(const half* a, const half* b, float* c,
                               int rows, int cols, int k) {
    using namespace nvcuda;
    int row = (blockIdx.y * (blockDim.x/32) + threadIdx.x / 32)*16;
    int col = blockIdx.x * tiles * 16;
    if (row >= rows) return;  // 同一 warp 的全部线程共同退出。
    wmma::fragment<wmma::accumulator,16,16,16,float> accum[tiles];
    #pragma unroll
    for (int j = 0; j < tiles; ++j) wmma::fill_fragment(accum[j],0.0f);
    for (int inner = 0; inner < k; inner += 16) {
        wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> left;
        wmma::load_matrix_sync(left, a+row*k+inner, k);
        #pragma unroll
        for (int j = 0; j < tiles; ++j) {
            if (col + 16*j < cols) {
                wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> right;
                wmma::load_matrix_sync(right, b+inner*cols+col+16*j, cols);
                wmma::mma_sync(accum[j],left,right,accum[j]);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < tiles; ++j) {
        if (col + 16*j < cols)
            wmma::store_matrix_sync(c+row*cols+col+16*j,accum[j],cols,wmma::mem_row_major);
    }
}

static void run_tensor_gemm(const std::string& mode) {
    int m = 512, n = 512, k = 256;
    if (mode == "tensor-thin" || mode == "tensor-grid") m = 17;
    if (mode == "tensor-edge") { m = 17; n = 257; k = 17; }
    if (mode == "tensor-control") { m = 32; n = 272; k = 32; }
    auto pad = [](int x) { return (x+15)/16*16; };
    int pm = pad(m), pn = pad(n), pk = pad(k);
    std::vector<half> a(pm*pk,__float2half(0)), b(pk*pn,__float2half(0));
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < k; ++col)
            a[row*pk+col] = __int2half_rn((3*row+col)%7-3);
    for (int row = 0; row < k; ++row)
        for (int col = 0; col < n; ++col)
            b[row*pn+col] = __int2half_rn((row+5*col)%11-5);
    std::vector<float> reference(pm*pn);
    for (int row = 0; row < m; ++row)
        for (int inner = 0; inner < k; ++inner)
            for (int col = 0; col < n; ++col)
                reference[row*pn+col] += __half2float(a[row*pk+inner])*__half2float(b[inner*pn+col]);
    Buffer<half> A(a.size()), B(b.size());
    Buffer<float> C(reference.size());
    A.upload(a); B.upload(b);
    if (mode == "tensor-grid") {
        std::vector<Variant> partitions;
        for (int warps : {1, 2, 4}) {
            const char* label = warps == 1 ? "one_warp" : (warps == 2 ? "two_warps" : "four_warps");
            partitions.push_back({label,[&,warps]{
                dim3 grid(pn/16,(pm+16*warps-1)/(16*warps));
                warp_tile_gemm<1><<<grid,32*warps>>>(A.p,B.p,C.p,pm,pn,pk);
            }});
        }
        compare(partitions,C.p,reference);
        return;
    }
    std::vector<Variant> variants = {
        {"scalar",[&]{half_scalar_gemm<<<dim3(pn/16,pm/16),dim3(16,16)>>>(A.p,B.p,C.p,pm,pn,pk);}},
        {"warp1",[&]{warp_tile_gemm<1><<<dim3(pn/16,(pm+63)/64),128>>>(A.p,B.p,C.p,pm,pn,pk);}},
        {"warp2",[&]{warp_tile_gemm<2><<<dim3((pn+31)/32,(pm+63)/64),128>>>(A.p,B.p,C.p,pm,pn,pk);}}
    };
#ifdef PROFILE_CUBLAS
    cublasHandle_t handle;
    blas_check(cublasCreate(&handle));
    float alpha=1.0f, beta=0.0f;
    variants.push_back({"blas",[&]{
        nvtxRangePushA("tensor_blas");
        // Row-major C=A*B is column-major C^T=B^T*A^T in the same memory.
        blas_check(cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,pn,pm,pk,
                               &alpha,B.p,CUDA_R_16F,pn,A.p,CUDA_R_16F,pk,
                               &beta,C.p,CUDA_R_32F,pn,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        nvtxRangePop();
    }});
#endif
    compare(variants,C.p,reference);
#ifdef PROFILE_CUBLAS
    blas_check(cublasDestroy(handle));
#endif
}

// One thread owns the same position in each of 64 channel planes.
// Keep either all its channel values, or only one tile, live across the recurrence.
template<int tile>
__device__ __forceinline__ void advance_channels(const float* input, float* output,
                                                 int positions, int iterations) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= positions) return;
    #pragma unroll 1
    for (int first = 0; first < 64; first += tile) {
        float state[tile];
        #pragma unroll
        for (int j = 0; j < tile; ++j)
            state[j] = input[(first + j) * positions + pos];
        #pragma unroll 1
        for (int k = 0; k < iterations; ++k) {
            #pragma unroll
            for (int j = 0; j < tile; ++j)
                state[j] = fmaf(state[j], 1.000012f, 0.000317f);
        }
        #pragma unroll
        for (int j = 0; j < tile; ++j)
            output[(first + j) * positions + pos] = state[j];
    }
}

template<int tile>
__global__ void __maxnreg__(32) channels_limited(const float* input, float* output,
                                                  int positions, int iterations) {
    advance_channels<tile>(input, output, positions, iterations);
}

__global__ void channels_free(const float* input, float* output, int positions, int iterations) {
    advance_channels<64>(input, output, positions, iterations);
}

static void run_register_study() {
    constexpr int positions = 16384, iterations = 32, threads = 128;
    auto input = values(64 * positions, 29);
    auto reference = input;
    for (float& value : reference)
        for (int k = 0; k < iterations; ++k)
            value = std::fma(value, 1.000012f, 0.000317f);
    Buffer<float> x(input.size()), y(input.size());
    x.upload(input);
    const int blocks = positions / threads;
    compare({
        {"limited64", [&]{ channels_limited<64><<<blocks, threads>>>(x.p, y.p, positions, iterations); }},
        {"limited8",  [&]{ channels_limited<8><<<blocks, threads>>>(x.p, y.p, positions, iterations); }},
        {"free64",    [&]{ channels_free<<<blocks, threads>>>(x.p, y.p, positions, iterations); }}
    }, y.p, reference);
}

int main(int argc,char** argv) {
    std::string mode = (argc == 2 || argc == 3) ? argv[1] : "";
    if (argc == 3) {
        char* end = nullptr;
        long device = std::strtol(argv[2], &end, 10);
        int count = 0;
        check(cudaGetDeviceCount(&count));
        if (end == argv[2] || *end || device < 0 || device >= count) {
            std::fprintf(stderr, "unknown CUDA device: %s\n", argv[2]);
            return 1;
        }
        check(cudaSetDevice(static_cast<int>(device)));
    }
    if (mode == "tensor-full" || mode == "tensor-thin" ||
        mode == "tensor-edge" || mode == "tensor-control" || mode == "tensor-grid") {
        run_tensor_gemm(mode);
    } else if (mode == "registers") {
        run_register_study();
    } else if (mode == "fma") {
        constexpr int threads = 32768, iterations = 256;
        auto x = values(threads*4, 29);
        auto ref = x;
        for (float& v : ref)
            for (int i = 0; i < iterations; ++i) v = std::fma(v, 1.000012f, 0.000317f);
        Buffer<float> X(x.size()), Y(ref.size());
        X.upload(x);
        compare({{"one", [&]{ fma_chains<1><<<threads/128,128>>>(X.p,Y.p,threads,iterations); }},
                 {"two", [&]{ fma_chains<2><<<threads/128,128>>>(X.p,Y.p,threads,iterations); }},
                 {"four",[&]{ fma_chains<4><<<threads/128,128>>>(X.p,Y.p,threads,iterations); }}}, Y.p,ref);
    } else if (mode == "hist-hot" || mode == "hist-spread") {
        constexpr int blocks = 1024, items = 1024, bins = 32;
        std::vector<int> keys(blocks*items);
        std::vector<float> ref(blocks*bins);
        for (int i = 0; i < int(keys.size()); ++i) {
            keys[i] = mode == "hist-hot" ? (i/32)%bins : i%bins;
            ++ref[(i/items)*bins+keys[i]];
        }
        Buffer<int> K(keys.size());
        Buffer<float> Y(ref.size());
        K.upload(keys);
        compare({{"global", [&]{ segment_histogram<0><<<blocks,128>>>(K.p,Y.p); }},
                 {"shared", [&]{ segment_histogram<1><<<blocks,128>>>(K.p,Y.p); }},
                 {"grouped",[&]{ segment_histogram<2><<<blocks,128>>>(K.p,Y.p); }}},Y.p,ref);
    } else if(mode=="transpose" || mode=="gemm") {
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
                     {"tile",[&]{gemm_tile<<<grid,block>>>(A.p,B.p,Y.p,n);}},
                     {"reg2",[&]{gemm_two_columns<<<dim3(n/32,n/16),block>>>(A.p,B.p,Y.p,n);}}},Y.p,ref);
        }
    } else if(mode=="reduction") {
        int blocks=4096; auto x=values(blocks*256,23); std::vector<float> ref(blocks);
        for(int i=0;i<blocks*256;++i) ref[i/256]+=x[i];
        Buffer<float> X(x.size()),Y(blocks); X.upload(x);
        compare({{"tree",[&]{reduce_tree<<<blocks,256>>>(X.p,Y.p);}},
                 {"tail",[&]{reduce_warp_tail<<<blocks,256>>>(X.p,Y.p);}},
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
                 {"four",[&]{csr_four_lanes<<<rows/32,128>>>(O.p,C.p,X.p,Y.p,rows);}},
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
                 {"eight",[&]{softmax_eight_lanes<<<rows/16,128>>>(X.p,Y.p,rows);}},
                 {"warp32",[&]{softmax_warp32<<<rows/4,128>>>(X.p,Y.p,rows);}}},Y.p,ref);
    } else {
        std::fprintf(stderr,"usage: kernel_lab gemm|transpose|reduction|graph-short|graph-skew|softmax|fma|hist-hot|hist-spread|tensor-full|tensor-thin|tensor-edge|tensor-control|tensor-grid|registers [device]\n");
        return 1;
    }
}
