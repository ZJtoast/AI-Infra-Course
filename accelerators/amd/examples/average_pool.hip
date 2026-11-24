// One FP32 image, size preserved by a one-pixel border; 3x3 average, stride 1.
#include "operator-check.hpp"

__global__ void average_pool3(const float* input, float* output,
                              int height, int width, bool include_padding) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= height || col >= width) return;
    float sum = 0;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int y = row + dy;
            const int x = col + dx;
            if (0 <= y && y < height && 0 <= x && x < width) {
                sum += input[y * width + x];
                ++count;
            }
        }
    }
    output[row * width + col] = sum / (include_padding ? 9 : count);
}

int main() {
    const std::vector<float> pixels{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> sums{12, 21, 16, 27, 45, 33, 24, 39, 28};
    const std::vector<int> counts{4, 6, 4, 6, 9, 6, 4, 6, 4};
    Buffer<float> x(pixels), y(9);
    for (bool include_padding : {false, true}) {
        std::vector<float> expected(9);
        for (int i = 0; i < 9; ++i)
            expected[i] = sums[i] / (include_padding ? 9 : counts[i]);
        LAUNCH(average_pool3, dim3(1), dim3(8, 8), x.ptr, y.ptr, 3, 3, include_padding);
        HIP_CHECK(hipDeviceSynchronize());
        y.expect("3x3 average pool", expected);
    }
}
