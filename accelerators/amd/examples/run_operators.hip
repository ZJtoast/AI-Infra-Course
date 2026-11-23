#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include "transpose.hip"
#include "matvec.hip"
#include "tiled-gemm.hip"
#include "reduction.hip"
#include "scan.hip"
#include "conv-chw.hip"
#include "csr-spmv.hip"
#include "gather.hip"
#include "rmsnorm.hip"
#include "softmax.hip"
#include "causal-attention.hip"
#include "append-pages.hip"
#include "combine-routes.hip"
#include "wave-sum.hip"

#include "operator-check.hpp"

void transpose_case(int rows, int cols) {
    std::vector<float> input(rows * cols), expected(rows * cols);
    std::iota(input.begin(), input.end(), 1.0f);
    for (int r=0; r<rows; ++r)
        for (int c=0; c<cols; ++c)
            expected[c*rows+r] = input[r*cols+c];
    Buffer<float> x(input), y(expected.size());
    LAUNCH(tiled_transpose, dim3((cols+15)/16,(rows+15)/16), dim3(16,16),
           x.ptr, y.ptr, rows, cols);
    y.expect("transpose", expected);
}

int main() {
    hipDeviceProp_t device{};
    HIP_CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("%s, wave size %d\n", device.name, device.warpSize);
    transpose_case(2,3);
    transpose_case(17,19);
    {
        Buffer<float> a{1,2,3,4,5,6}, x{2,0,-1}, y(2);
        LAUNCH(matvec, 1, 4, a.ptr,x.ptr,y.ptr,2,3);
        y.expect("matvec", {-1,2});
        Buffer<float> b{7,8,9,10,11,12}, c(4);
        LAUNCH(tiled<2>, dim3(1,1), dim3(2,2), a.ptr,b.ptr,c.ptr,2,2,3);
        c.expect("tiled GEMM", {58,64,139,154});
    }
    {
        Buffer<float> x{1,2,3,4,5,6}, part(2), sum(1);
        LAUNCH(block_sum<4>, 2, 4, x.ptr,part.ptr,6);
        part.expect("partial reduction", {10,11});
        LAUNCH(block_sum<4>, 1, 4, part.ptr,sum.ptr,2);
        sum.expect("reduction", {21});
        Buffer<int> v{2,1,0,3,4,1}, prefix(6), totals(2), scanned(2), final_total(1);
        LAUNCH(scan_tiles<4>, 2, 4, v.ptr,prefix.ptr,totals.ptr,6);
        prefix.expect("within-tile scan", {2,3,3,6,4,5});
        totals.expect("tile totals", {6,5});
        LAUNCH(scan_tiles<4>, 1, 4, totals.ptr,scanned.ptr,final_total.ptr,2);
        scanned.expect("scan of tile totals", {6,11});
        LAUNCH(add_bases<4>, 2, 4, prefix.ptr,scanned.ptr,6);
        prefix.expect("device-wide scan", {2,3,3,6,10,11});
    }
    {
        Buffer<float> x{1,2,3,4,5,6,7,8,9}, w{-1,0,0,1}, y(4);
        LAUNCH(conv_chw, dim3(1,1,1), dim3(2,2), x.ptr,w.ptr,y.ptr,1,3,3,2,2,1,2,2);
        y.expect("strided padded convolution", {1,3,7,4});
        LAUNCH(conv_chw, dim3(1,1,1), dim3(2,2), x.ptr,w.ptr,y.ptr,1,3,3,2,1,0,2,2);
        y.expect("valid convolution", {4,4,4,4});
    }
    {
        Buffer<int> offsets{0,2,2,5}, columns{0,2,0,1,2}, ids{2,0,2};
        Buffer<float> weights{2,1,1,3,2}, v{4,5,6}, y(3);
        LAUNCH(csr_rows, 1, 4, offsets.ptr,columns.ptr,weights.ptr,v.ptr,y.ptr,3);
        y.expect("CSR including an empty row", {14,0,31});
        Buffer<float> features{4,1,5,-1,6,2}, gathered(6);
        LAUNCH(gather_rows, 2, 4, features.ptr,ids.ptr,gathered.ptr,3,2);
        gathered.expect("gather with repeated index", {6,2,4,1,6,2});
    }
    {
        Buffer<float> x{0,4,0,0}, w{1,3,1,1}, out(4);
        LAUNCH(rms_rows<4>, 1, 4, x.ptr,w.ptr,out.ptr,4,0.0f);
        out.expect("RMSNorm", {0,6,0,0});
        Buffer<float> zeros(4);
        LAUNCH(rms_rows<4>, 1, 4, zeros.ptr,w.ptr,out.ptr,4,1e-5f);
        out.expect("all-zero RMSNorm row", {0,0,0,0});
        Buffer<float> logits{0,1,2,0,0,0}, p(6);
        LAUNCH(softmax_rows<4>, 2, 4, logits.ptr,p.ptr,3);
        p.expect("row softmax", {.09003057f,.24472847f,.66524096f,1.f/3,1.f/3,1.f/3});
        Buffer<float> q{1,1}, k{0,std::log(2.f),std::log(4.f),std::log(8.f)};
        Buffer<float> v{2,0,5,1,8,-1,1,3}, o(4);
        LAUNCH(causal_attention, dim3(1,2), 4, q.ptr,k.ptr,v.ptr,o.ptr,2,4,1,2,2);
        o.expect("causal attention", {44.f/7,-2.f/7,52.f/15,22.f/15});
    }
    {
        Buffer<float> src{30,31,40,41}, cache(32);
        Buffer<int> page_table{4,1,7};
        LAUNCH(append_pages, 1, 4, src.ptr,page_table.ptr,cache.ptr,2,2,2,3);
        std::vector<float> expected(32);
        expected[6]=30; expected[7]=31; expected[28]=40; expected[29]=41;
        cache.expect("paged KV append across two physical pages", expected);
        Buffer<float> packed{4,6,6,7}, gates{.75f,.25f,.25f,.75f}, y(2);
        Buffer<int> token{0,1,0,1};
        LAUNCH(combine_routes, 1, 4, packed.ptr,token.ptr,gates.ptr,y.ptr,4,1);
        y.expect("MoE combine", {4.5f,6.75f});
    }
    {
        const int width = device.warpSize, n = 64, groups = (n+width-1)/width;
        std::vector<float> input(n), expected(groups);
        std::iota(input.begin(),input.end(),1.0f);
        for (int i=0; i<n; ++i) expected[i/width] += input[i];
        Buffer<float> values(input), partial(groups);
        LAUNCH(wave_sum, groups, width, values.ptr,partial.ptr,n);
        partial.expect("wave reductions", expected);
    }
}
