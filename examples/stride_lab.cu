#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void check(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(call) check((call), #call)

__global__ void initialize(float* x, float* y, float* z, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        x[i] = float(i % 7);
        y[i] = float(i % 11);
        z[i] = -1.0f;
    }
}

__global__ void stride_add(const float* x, const float* y, float* z,
                           int outputs, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < outputs) {
        int j = i * stride;
        z[j] = x[j] + y[j];
    }
}

int main(int argc, char** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "1") && std::strcmp(argv[1], "2") && std::strcmp(argv[1], "4"))) {
        std::fprintf(stderr, "usage: stride_lab 1|2|4\n");
        return EXIT_FAILURE;
    }
    constexpr int n = 1 << 18;
    constexpr int threads = 256;
    constexpr int steps = 40;
    int stride = std::atoi(argv[1]);
    int count = n * stride;
    size_t bytes = size_t(count) * sizeof(float);
    float *x, *y, *z;
    CHECK(cudaMalloc(&x, bytes));
    CHECK(cudaMalloc(&y, bytes));
    CHECK(cudaMalloc(&z, bytes));
    initialize<<<(count + threads - 1) / threads, threads>>>(x, y, z, count);
    CHECK(cudaGetLastError());
    for (int t = 0; t < 4; ++t)
        stride_add<<<n / threads, threads>>>(x, y, z, n, stride);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, finish;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&finish));
    CHECK(cudaProfilerStart());
    nvtxRangePushA("steady");
    CHECK(cudaEventRecord(start));
    for (int t = 0; t < steps; ++t)
        stride_add<<<n / threads, threads>>>(x, y, z, n, stride);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(finish));
    CHECK(cudaEventSynchronize(finish));
    nvtxRangePop();
    CHECK(cudaProfilerStop());
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, finish));
    std::vector<float> result(count);
    CHECK(cudaMemcpy(result.data(), z, bytes, cudaMemcpyDeviceToHost));
    for (int j = 0; j < count; ++j) {
        float reference = j % stride == 0 ? float(j % 7 + j % 11) : -1.0f;
        if (result[j] != reference) {
            std::fprintf(stderr, "wrong result at %d\n", j);
            return EXIT_FAILURE;
        }
    }
    std::printf("stride=%d, launches=%d, mean GPU interval=%.6f ms, verified\n",
                stride, steps, ms / steps);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(finish));
    CHECK(cudaFree(z));
    CHECK(cudaFree(y));
    CHECK(cudaFree(x));
    return EXIT_SUCCESS;
}
