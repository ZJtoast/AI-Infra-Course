#include "operator-check.hpp"
#include "mfma-block.hip"
#include "causal-attention.hip"
#include "attention-block.hip"
#include "quant-group.hip"
#include "route-stable.hip"
#include "scan.hip"
#include "scan-cooperation.hip"
#include "transpose-layout.hip"
#include "matvec.hip"
#include "matvec-wave.hip"
#include "conv-chw.hip"
#include "conv-halo.hip"
#include "csr-wave.hip"
#include "rmsnorm.hip"
#include "softmax.hip"
#include "row-fusions.hip"
#include "kv-gather.hip"
#include <algorithm>
#include <numeric>

// ROCm: hipcc -std=c++17 -O2 --offload-arch=gfx942 advanced.hip -o advanced
void gemm_case(bool single_entry) {
    constexpr int M=17,N=35,K=5;
    std::vector<float> a(M*K), b(K*N), ref(M*N);
    for(int r=0;r<M;++r) for(int d=0;d<K;++d)
        a[r*K+d]=single_entry ? float(r==16 && d==4) : float(r+d+1);
    for(int d=0;d<K;++d) for(int c=0;c<N;++c)
        b[d*N+c]=single_entry ? float(d==4 && c==34) : float((d+1)*(1+c%3));
    for(int r=0;r<M;++r) for(int c=0;c<N;++c)
        for(int d=0;d<K;++d) ref[r*N+c]+=a[r*K+d]*b[d*N+c];
    Buffer<float> aa(a), bb(b), cc(M*N);
    dim3 grid((N+31)/32,(M+15)/16);
    LAUNCH(mfma_direct,grid,128,aa.ptr,bb.ptr,cc.ptr,M,N,K);
    cc.expect("direct MFMA",ref);
    LAUNCH(mfma_block,grid,128,aa.ptr,bb.ptr,cc.ptr,M,N,K);
    cc.expect("MFMA with LDS",ref);
}

void attention_case() {
    constexpr int R=3,S=5,D=4,F=4;
    std::vector<float> q(R*D),k(S*D),v{2,0,1,-1,8,1,0,2,-1,2,3,0,3,-1,2,4,4,0,0,1};
    std::vector<float> w{1,3,2,6,12},ref(R*F);
    for(int r=0;r<R;++r) q[r*D]=1;
    for(int j=0;j<S;++j) k[j*D]=2*std::log(w[j]);
    for(int r=0;r<R;++r) {
        float den=std::accumulate(w.begin(),w.begin()+3+r,0.0f);
        for(int f=0;f<F;++f)
            for(int j=0;j<3+r;++j) ref[r*F+f]+=w[j]*v[j*F+f]/den;
    }
    Buffer<float> qq(q),kk(k),vv(v),y(R*F);
    LAUNCH(causal_attention,dim3(1,R),F,qq.ptr,kk.ptr,vv.ptr,y.ptr,R,S,D,F,2);
    y.expect("per-feature SDPA",ref);
    LAUNCH(attention_block4,1,64,qq.ptr,kk.ptr,vv.ptr,y.ptr,R,S,2);
    y.expect("query-block SDPA",ref);
}

void quant_case() {
    std::vector<float> x(2*66), restored(x.size());
    x[0]=254;x[1]=-63;x[2]=1;x[3]=-1;x[4]=-81;x[64]=63.5f;x[65]=-5.25f;
    std::vector<float> scales{2,.5f,1,1};std::vector<signed char> codes(x.size());
    for(int i=0;i<int(x.size());++i) {
        float s=scales[(i/66)*2+(i%66)/64];
        codes[i]=static_cast<signed char>(std::round(x[i]/s));
        restored[i]=s*codes[i];
    }
    Buffer<float> in(x),s(4),out(x.size());Buffer<signed char> q(x.size());
    LAUNCH(scales_serial,dim3(2,2),1,in.ptr,s.ptr,66);
    LAUNCH(quant_pass,dim3(2,2),64,in.ptr,s.ptr,q.ptr,66);
    s.expect("serial scales",scales);q.expect("two-pass INT8",codes);
    LAUNCH(quant_wave64,dim3(2,2),64,in.ptr,q.ptr,s.ptr,66);
    s.expect("wave scales",scales);q.expect("fused INT8",codes);
    LAUNCH(dequant_group,dim3(2,2),64,q.ptr,s.ptr,out.ptr,66);
    out.expect("dequantized",restored);
}

void routes_case(const std::vector<int>& expert) {
    constexpr int experts=3;
    int n=static_cast<int>(expert.size()),tiles=(n+63)/64;
    std::vector<int> order(n),starts(experts+1);
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return expert[a]<expert[b];});
    for(int e:expert) ++starts[e+1];
    std::partial_sum(starts.begin(),starts.end(),starts.begin());
    Buffer<int> ids(expert), packed(n), counts(tiles*experts), base(tiles*experts), ptr(experts+1);
    LAUNCH(pack_reference,tiles,64,ids.ptr,packed.ptr,n);
    packed.expect("naive stable packing",order);
    LAUNCH(count_route_tiles,tiles,64,ids.ptr,counts.ptr,n,experts);
    LAUNCH(route_bases,1,1,counts.ptr,base.ptr,ptr.ptr,tiles,experts);
    ptr.expect("expert boundaries",starts);
    LAUNCH(scatter_routes,tiles,64,ids.ptr,base.ptr,packed.ptr,n,experts);
    packed.expect("ballot stable packing",order);
}

void scans_case() {
    std::vector<int> x(64,1);x.insert(x.end(),{2,1,0,3,4,1});
    std::vector<int> ref(x.size());std::partial_sum(x.begin(),x.end(),ref.begin());
    Buffer<int> input(x),y(x.size()),tot(2),prefix(2),all(1);
    for(int version=0;version<3;++version) {
        if(version==0) LAUNCH(scan_tiles<64>,2,64,input.ptr,y.ptr,tot.ptr,70);
        if(version==1) LAUNCH(scan_tree64,2,64,input.ptr,y.ptr,tot.ptr,70);
        if(version==2) LAUNCH(scan_shuffle64,2,64,input.ptr,y.ptr,tot.ptr,70);
        tot.expect("tile sums",{64,11});
        LAUNCH(scan_shuffle64,1,64,tot.ptr,prefix.ptr,all.ptr,2);
        LAUNCH(add_bases<64>,2,64,y.ptr,prefix.ptr,70);
        y.expect("inclusive scan",ref);
    }
}

void transpose_case32() {
    constexpr int R=35,C=37;
    std::vector<float> x(R*C),ref(R*C);
    std::iota(x.begin(),x.end(),1.0f);
    for(int r=0;r<R;++r) for(int c=0;c<C;++c) ref[c*R+r]=x[r*C+c];
    Buffer<float> input(x),out(R*C);dim3 block(32,8),grid((C+31)/32,(R+31)/32);
    LAUNCH(transpose_naive,grid,block,input.ptr,out.ptr,R,C);
    out.expect("stride stores",ref);
    LAUNCH(transpose32<32>,grid,block,input.ptr,out.ptr,R,C);
    out.expect("LDS pitch32",ref);
    LAUNCH(transpose32<33>,grid,block,input.ptr,out.ptr,R,C);
    out.expect("LDS pitch33",ref);
}

void matvec_and_conv() {
    constexpr int R=2,C=130;
    std::vector<float> x(C),a(R*C),ref(R);
    for(int c=0;c<C;++c) x[c]=float(c%5-2);
    for(int r=0;r<R;++r) for(int c=0;c<C;++c) {
        a[r*C+c]=float((r+c)%7-3);
        ref[r]+=a[r*C+c]*x[c];
    }
    Buffer<float> xa(x),aa(a),out(R);
    LAUNCH(matvec,1,64,aa.ptr,xa.ptr,out.ptr,R,C);out.expect("one thread per row",ref);
    LAUNCH(matvec_wave64,R,64,aa.ptr,xa.ptr,out.ptr,C);out.expect("one wave per row",ref);
    constexpr int H=10,W=11;
    std::vector<float> img(H*W),cv((H-1)*(W-1)),pool(((H-1)/2)*((W-1)/2));
    for(int i=0;i<H*W;++i) img[i]=float((i*11)%19-9);
    for(int r=0;r<H-1;++r) for(int c=0;c<W-1;++c)
        cv[r*(W-1)+c]=img[(r+1)*W+c+1]-img[r*W+c];
    int pitch=W-1;
    for(int r=0;r<(H-1)/2;++r) for(int c=0;c<(W-1)/2;++c) {
        int p=2*r*pitch+2*c;
        pool[r*((W-1)/2)+c]=std::max({cv[p],cv[p+1],cv[p+pitch],cv[p+pitch+1]});
    }
    Buffer<float> input(img),w{-1,0,0,1},y(cv.size()),z(pool.size());
    dim3 block(8,8),grid((W-2)/8+1,(H-2)/8+1);
    LAUNCH(conv_chw,grid,block,input.ptr,w.ptr,y.ptr,1,H,W,2,1,0,H-1,W-1);
    y.expect("direct convolution",cv);
    LAUNCH(conv_halo2,grid,block,input.ptr,w.ptr,y.ptr,H,W);
    y.expect("halo convolution",cv);
    LAUNCH(maxpool_stride2,grid,block,y.ptr,z.ptr,H-1,W-1);z.expect("max pooling",pool);
    Buffer<float> neg{-8,-7,-5,-9},yneg(1);
    LAUNCH(maxpool_stride2,1,dim3(2,2),neg.ptr,yneg.ptr,2,2);yneg.expect("negative window",{-5});
}

void sparse_and_fused_rows() {
    Buffer<int> p{0,2,2,5},c{0,2,0,1,2};
    Buffer<float> w{2,1,1,3,2},x{4,1,0,2,5,-1,2,0,6,2,1,-2},out(12);
    LAUNCH(csr_four,3,64,p.ptr,c.ptr,w.ptr,x.ptr,out.ptr);
    out.expect("four-channel CSR",{14,4,1,2,0,0,0,0,31,2,8,-2});
    Buffer<float> main{2,1,0,-1},res{-2,3,0,1},gain{1,3,1,1},u(4),y(4);
    LAUNCH(residual_add,1,64,main.ptr,res.ptr,u.ptr,4);
    LAUNCH(rms_rows<4>,1,4,u.ptr,gain.ptr,y.ptr,4,0.0f);
    y.expect("separate residual and RMSNorm",{0,6,0,0});
    LAUNCH(residual_rms64,1,64,main.ptr,res.ptr,gain.ptr,u.ptr,y.ptr,4,0.0f);
    u.expect("fused residual",{0,4,0,0});y.expect("fused norm",{0,6,0,0});
    Buffer<float> logits{0,1,2,1000,1001,1002},prob(6);
    std::vector<float> ref{.09003057f,.24472847f,.66524096f,.09003057f,.24472847f,.66524096f};
    LAUNCH(softmax_rows<4>,2,4,logits.ptr,prob.ptr,3);prob.expect("LDS softmax",ref);
    LAUNCH(softmax_wave64,2,64,logits.ptr,prob.ptr,3);prob.expect("wave softmax",ref);
}

void gather_cases() {
    Buffer<float> pages{30,31,99,99,40,41,50,51,10,11,20,21},values(16);
    Buffer<int> table{2,0,2,1},length{3,4};
    LAUNCH(gather_kv_batch,dim3(4,2),4,pages.ptr,table.ptr,length.ptr,values.ptr,2,2,4,2);
    values.expect("shared prefix and ragged length",{10,11,20,21,30,31,0,0,10,11,20,21,40,41,50,51});
    Buffer<int> packed{0,2,1,3},slot(4);
    Buffer<float> expert{4,6,6,7},gate{.75f,.25f,.25f,.75f},y(2);
    LAUNCH(inverse_slots,1,4,packed.ptr,slot.ptr,4);
    LAUNCH(combine_by_token,1,4,expert.ptr,gate.ptr,slot.ptr,y.ptr,2,2,1);
    y.expect("MoE gather combine",{4.5f,6.75f});
}

int main() {
    hipDeviceProp_t prop{};HIP_CHECK(hipGetDeviceProperties(&prop,0));
    if(prop.warpSize!=64) {std::fprintf(stderr,"This lab uses a 64-lane CDNA wave.\n");return 1;}
    gemm_case(false);gemm_case(true);attention_case();quant_case();
    routes_case({2,0,2,1,0,2,1,1});
    std::vector<int> many(70);
    for(int i=0;i<70;++i) many[i]=(i*7+i/3)%3;
    routes_case(many);scans_case();transpose_case32();
    matvec_and_conv();sparse_and_fused_rows();gather_cases();
}
