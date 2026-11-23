#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check_status(hipError_t status, const char* call, int line) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "line %d: %s: %s\n", line, call, hipGetErrorString(status));
    std::exit(EXIT_FAILURE);
}
#define HIP_CHECK(call) check_status((call), #call, __LINE__)
#define LAUNCH(kernel, grid, block, ...) do { \
    kernel<<<grid, block>>>(__VA_ARGS__); HIP_CHECK(hipGetLastError()); \
} while (0)

template<class T> struct Buffer {
    T* ptr;
    size_t count;
    explicit Buffer(const std::vector<T>& values) : count(values.size()) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
        HIP_CHECK(hipMemcpy(ptr, values.data(), count * sizeof(T), hipMemcpyHostToDevice));
    }
    Buffer(std::initializer_list<T> values) : Buffer(std::vector<T>(values)) {}
    explicit Buffer(size_t n) : Buffer(std::vector<T>(n)) {}
    ~Buffer() { hipFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void expect(const char* label, const std::vector<T>& expected) const {
        if (expected.size() != count) std::abort();
        std::vector<T> result(count);
        HIP_CHECK(hipMemcpy(result.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            double a = result[i], b = expected[i];
            if (!(std::abs(a-b) <= 1e-5 * (1 + std::abs(b)))) {
                std::fprintf(stderr, "%s[%zu]: got %.8g, expected %.8g\n", label, i, a, b);
                std::exit(EXIT_FAILURE);
            }
        }
        std::printf("PASS %s\n", label);
    }
};
