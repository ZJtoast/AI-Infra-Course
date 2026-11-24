// ROCm HIP. One wave per CSR row, two FP32 features per node, finite edge logits.
#include "operator-check.hpp"
#include <algorithm>

__global__ void csr_attention(const int* ptr, const int* col, const float* score,
                              const float* value, float* out) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int first = ptr[row], end = ptr[row + 1];
    if (first == end) {
        if (lane == 0) { out[2*row] = 0; out[2*row+1] = 0; }
        return;
    }
    float peak = -INFINITY;
    for (int e = first + lane; e < end; e += warpSize)
        peak = fmaxf(peak, score[e]);
    for (int delta = warpSize / 2; delta; delta /= 2) {
        const float other = __shfl_down(peak, delta, warpSize);
        if (lane < delta) peak = fmaxf(peak, other);
    }
    peak = __shfl(peak, 0, warpSize);

    float denom = 0.0f, x = 0.0f, y = 0.0f;
    for (int e = first + lane; e < end; e += warpSize) {
        const float weight = expf(score[e] - peak);
        const int node = col[e];
        denom += weight;
        x += weight * value[2*node];
        y += weight * value[2*node+1];
    }
    for (int delta = warpSize / 2; delta; delta /= 2) {
        const float d = __shfl_down(denom, delta, warpSize);
        const float a = __shfl_down(x, delta, warpSize);
        const float b = __shfl_down(y, delta, warpSize);
        if (lane < delta) { denom += d; x += a; y += b; }
    }
    if (lane == 0) { out[2*row] = x / denom; out[2*row+1] = y / denom; }
}

int main() {
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    // Row 4 has more edges than lanes. Lane 0 must process at least two edges.
    const int long_row = props.warpSize + 3;
    std::vector<int> offsets{0, 3, 3, 4, 6, 6 + long_row};
    std::vector<int> cols{2, 0, 2, 1, 0, 1};
    std::vector<float> logits{0, std::log(2.0f), std::log(4.0f), -1000, 1000, 1000};
    std::vector<float> values{7,0, 1,5, 0,7};
    for (int i = 0; i < long_row; ++i) {
        cols.push_back(i % 3);
        logits.push_back(float(i % 5) - 1000.0f);
    }

    std::vector<float> expected{2,5, 0,0, 1,5, 4,2.5f};
    double total = 0, a = 0, b = 0;
    const double peak = *std::max_element(logits.begin()+6, logits.end());
    for (int e = 6; e < offsets.back(); ++e) {
        const double w = std::exp(logits[e] - peak);
        total += w; a += w * values[2*cols[e]]; b += w * values[2*cols[e]+1];
    }
    expected.push_back(float(a / total)); expected.push_back(float(b / total));
    Buffer<int> ptr(offsets), col(cols);
    Buffer<float> score(logits), value(values), out(expected.size());
    LAUNCH(csr_attention, dim3(5), dim3(props.warpSize), ptr.ptr, col.ptr,
           score.ptr, value.ptr, out.ptr);
    HIP_CHECK(hipDeviceSynchronize());
    out.expect("CSR edge softmax and weighted gather", expected);
}
