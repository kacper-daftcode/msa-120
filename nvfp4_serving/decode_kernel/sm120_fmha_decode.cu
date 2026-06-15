// SM120 Block-Sparse Paged Attention — DECODE-SPECIALIZED (flash-decoding)
//
// Companion to sm120_fmha_paged.cu (the PREFILL FA2 design). That kernel is
// 31x too slow at bs1 decode because at q_len==1 its grid collapses to
// (1, Hq=64)=64 blocks (0.34 blocks/SM) and 63/64 of every M-tile is a dead
// query row.
//
// This kernel mirrors vLLM's Triton flash-DECODING geometry:
//
//   M3 GQA: Hq=64 q-heads, Hkv=4 kv-heads -> 16 q-heads share each kv-head.
//   We tile the 16-head GQA GROUP as the MMA "M" dimension (16 REAL rows, zero
//   dead rows) instead of 64 query rows.  head_dim=128, page=64, topk pages.
//
//   SPLIT-K: the selected KV pages are partitioned across `split_chunks`
//   thread-blocks per (request, kv-head). Each block computes a PARTIAL
//   attention (partial O + running max/denom) over its page subset.
//   Grid = (num_kv_heads, split_chunks, num_requests) -> fills the 188 SMs.
//
//   LSE-MERGE epilogue: a second kernel reduces the split-K partials per
//   (request, q-head) into final O using the flash-decoding LSE merge.
//
// Numerics match sm120_fmha_paged.cu exactly: bf16 Q/K, fp8(e4m3) P and V for
// the PV GEMM (block-scaled MMA), fp32 accumulation, bf16 out.
//
// Compile: nvcc -gencode arch=compute_120f,code=sm_120f -O3
// SPDX-License-Identifier: MIT

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int GQA      = 16;   // q-heads per kv-head == MMA M rows
static constexpr int BLK_N    = 64;   // keys per page == PAGE_SIZE
static constexpr int PAGE_SIZE = 64;
static constexpr int HEAD_DIM = 128;
#ifndef NUM_WARPS
#define NUM_WARPS 4           // 4 warps key-split the 64 keys (16 keys/warp)
#endif
static constexpr int MMA_N = 8;
static constexpr int MMA_K_BF16 = 16;
static constexpr int MMA_K_FP8 = 32;

__device__ __forceinline__ void cp_async_16b(void *smem, const void *gmem) {
    uint32_t smem_addr;
    asm volatile("{\n"
        ".reg .u64 u;\n"
        "cvta.to.shared.u64 u, %1;\n"
        "cvt.u32.u64 %0, u;\n"
        "}\n" : "=r"(smem_addr) : "l"(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_async_wait(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else        asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ void hmma_bf16_m16n8k16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__device__ __forceinline__ void qmma_sf_fp8_m16n8k32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    uint32_t sfa = 0x7F7F7F7F;
    uint32_t sfb = 0x7F7F7F7F;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col"
        ".kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "%14, {0, 1}, %15, {0, 1};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(sfa), "r"(sfb));
}

__device__ __forceinline__ uint16_t cvt_2f32_to_e4m3x2(float lo, float hi) {
    float2 pair = make_float2(lo, hi);
    return __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ uint32_t f32x4_to_e4m3x4(float v0, float v1, float v2, float v3) {
    uint16_t lo = cvt_2f32_to_e4m3x2(v0, v1);
    uint16_t hi = cvt_2f32_to_e4m3x2(v2, v3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

// Pack two bf16 (held as floats) into one uint32 (lo,hi) for an HMMA operand.
__device__ __forceinline__ uint32_t bf16x2_pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)(*reinterpret_cast<uint16_t*>(&lo))
         | ((uint32_t)(*reinterpret_cast<uint16_t*>(&hi)) << 16);
}

__device__ __forceinline__ uint32_t lds_u32(const void *smem_ptr) {
    uint32_t smem_addr, r;
    asm volatile(
        "{\n.reg .u64 u;\n cvta.to.shared.u64 u, %1;\n cvt.u32.u64 %2, u;\n"
        " ld.shared.b32 %0, [%2];\n}\n" : "=r"(r), "+l"(smem_ptr), "=r"(smem_addr));
    return r;
}

// ---- ldmatrix (LDSM) helpers ---------------------------------------------
// Convert a generic shared pointer to the 32-bit shared-window address ldmatrix
// expects. The four .b16 fragments returned by an x4 op are distributed one
// uint32 (= 2 bf16) per lane.
__device__ __forceinline__ uint32_t cvta_shared(const void *p) {
    uint32_t a;
    asm volatile("{ .reg .u64 u; cvta.to.shared.u64 u, %1; cvt.u32.u64 %0, u; }\n"
                 : "=r"(a) : "l"(p));
    return a;
}
// ldmatrix.x4 (no transpose): 4 contiguous 8x8 b16 tiles. Used to feed the MMA
// A operand (row-major MxK 16x16) and the K (QK B) operand (8x16 col, frags d0,d2).
__device__ __forceinline__ void ldmatrix_x4(uint32_t &d0, uint32_t &d1,
                                            uint32_t &d2, uint32_t &d3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
// ldmatrix.x4.trans: transposes each 8x8 b16 tile on load. Feeds the PV B operand
// (V stored row-major [key][dim]; the transpose yields [dim][key]). For a 16x16
// V region, the 4 frags map (v0,v1)=dims 0-7 over 16 keys, (v2,v3)=dims 8-15.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t &d0, uint32_t &d1,
                                                  uint32_t &d2, uint32_t &d3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}

// Load one page (BLK_N x HEAD_DIM) for ONE kv head, cooperatively over `nthr`.
__device__ void load_page_async(__nv_bfloat16 *smem_tile, const __nv_bfloat16 *gmem,
                                int num_valid_rows, int stride_row, int tid, int nthr) {
    const int elems_per_copy = 8;
    const int total_elems = BLK_N * HEAD_DIM;
    const int copies = (total_elems + nthr * elems_per_copy - 1) / (nthr * elems_per_copy);
    for (int c = 0; c < copies; c++) {
        int elem_idx = (tid + c * nthr) * elems_per_copy;
        if (elem_idx >= total_elems) break;
        int row = elem_idx / HEAD_DIM, col = elem_idx % HEAD_DIM;
        __nv_bfloat16 *dst = smem_tile + row * HEAD_DIM + col;
        if (row < num_valid_rows) cp_async_16b(dst, gmem + row * stride_row + col);
        else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
}

// ===========================================================================
// SPLIT-K PARTIAL KERNEL
//   grid = (num_kv_heads, split_chunks, num_requests)
//   block = NUM_WARPS*32 threads. Warp 0 does the whole GQA group; warps 1..3
//   are spare (kept for smem-load bandwidth + future N-split). We launch 1 warp
//   of real MMA per block; the parallelism comes from the (kvh, chunk) grid.
//
//   Each block processes a CONTIGUOUS slice of the request's selected pages:
//     pages [chunk*pages_per_chunk, (chunk+1)*pages_per_chunk).
//   Output:
//     O_part : [num_requests, split_chunks, num_kv_heads, GQA, HEAD_DIM] f32
//     M_part : [num_requests, split_chunks, num_kv_heads, GQA] f32 (row max)
//     L_part : [num_requests, split_chunks, num_kv_heads, GQA] f32 (row denom)
// ===========================================================================
extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
sm120_fmha_decode_partial_bf16(
    const __nv_bfloat16 * __restrict__ Q,     // [num_requests, Hq, 128]
    const __nv_bfloat16 * __restrict__ Kc,    // [num_pages, 64, Hkv, 128]
    const __nv_bfloat16 * __restrict__ Vc,    // [num_pages, 64, Hkv, 128]
    __nv_bfloat16 * __restrict__ O_part,      // [R, C, Hkv, GQA, 128] bf16
    float * __restrict__ M_part,              // [R, C, Hkv, GQA]
    float * __restrict__ L_part,              // [R, C, Hkv, GQA]
    const int * __restrict__ block_ids,       // [num_requests, topk]  logical pages (-1 pad)
    const int * __restrict__ block_table,     // [num_requests, max_logical_blocks]
    int max_logical_blocks,
    int topk,
    int seq_len_k,
    int num_heads_q,
    int num_heads_kv,
    float softmax_scale,
    int split_chunks,
    int pages_per_chunk)
{
    extern __shared__ __nv_bfloat16 smem_raw[];

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    const int grp = lid / 4;     // 0..7  -> query-row pair {grp, grp+8}
    const int sub = lid % 4;     // 0..3  -> key/col pair
    const int nthr = NUM_WARPS * 32;

    const int hkv   = blockIdx.x;
    const int chunk = blockIdx.y;
    const int req   = blockIdx.z;

    const int qstride   = num_heads_q  * HEAD_DIM;
    const int kvstride  = num_heads_kv * HEAD_DIM;             // token stride in a page
    const int page_stride = PAGE_SIZE * num_heads_kv * HEAD_DIM;

    // The 16 q-heads sharing kv-head hkv are [hkv*GQA, hkv*GQA+16).
    const int hq0 = hkv * GQA;
    const __nv_bfloat16 *Qp    = Q + (int64_t)req * qstride + hq0 * HEAD_DIM;
    const __nv_bfloat16 *Kbase = Kc + hkv * HEAD_DIM;
    const __nv_bfloat16 *Vbase = Vc + hkv * HEAD_DIM;
    const int *bt_row  = block_table + req * max_logical_blocks;
    const int *blk_row = block_ids   + req * topk;

    // SMEM: sQ [16x128]  +  NSTAGE x (sK[64x128] sV[64x128])  +  sP[1KB]
    // NSTAGE=2 double-buffers K/V (page t+1 loads while page t computes);
    // NSTAGE=1 halves smem -> 2 resident blocks/SM (better latency hiding for
    // the 1-page-per-block regime). Selected by the host via -DNSTAGE.
#ifndef NSTAGE
#define NSTAGE 2
#endif
    __nv_bfloat16 *sQ  = smem_raw;
    __nv_bfloat16 *sKV = sQ + GQA * HEAD_DIM;
    const int STAGE_ELEMS = 2 * BLK_N * HEAD_DIM;      // K+V per stage
    uint8_t *sP_all = reinterpret_cast<uint8_t*>(sKV + NSTAGE * STAGE_ELEMS);
    #define SK(stage) (sKV + ((stage) % NSTAGE) * STAGE_ELEMS)
    #define SV(stage) (sKV + ((stage) % NSTAGE) * STAGE_ELEMS + BLK_N * HEAD_DIM)

    // Issue the 16-head GQA query group load (overlaps the first page load).
    {
        const int elems_per_copy = 8;
        const int total = GQA * HEAD_DIM;
        for (int e = tid * elems_per_copy; e < total; e += nthr * elems_per_copy) {
            int row = e / HEAD_DIM, col = e % HEAD_DIM;
            cp_async_16b(sQ + row * HEAD_DIM + col, Qp + row * HEAD_DIM + col);
        }
        cp_async_commit();   // group: Q
    }

    // ===== warp-key-split helpers =====
    // (n-tiles {2w, 2w+1} of the 8 MMA_N=8 columns). All 4 warps load + compute.
    const int sub2     = sub * 2;
    const int qbase0   = grp * HEAD_DIM;          // q-row grp
    const int qbase1   = (grp + 8) * HEAD_DIM;    // q-row grp+8
    const int grp_hdim = grp * HEAD_DIM;
    const int nt_base  = wid * 2;                 // this warp's first n-tile (0,2,4,6)
    const int key_base = wid * 16;                // this warp's first key

    const __nv_bfloat16 * __restrict__ sQ_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sQ, 16));

    // Per-warp flash accumulator over this warp's key subset.
    float Oa[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) Oa[i][0]=Oa[i][1]=Oa[i][2]=Oa[i][3]=0.f;
    float rm[2] = {-INFINITY, -INFINITY};
    float rl[2] = {0.f, 0.f};
    constexpr float LG2E = 1.4426950408889634f;

    const int p_begin = chunk * pages_per_chunk;
    const int p_end   = min(p_begin + pages_per_chunk, topk);

    // ---- page metadata helper (returns valid, kvv, page) ----
    auto page_meta = [&](int t, int &kvv, const __nv_bfloat16* &Kpage,
                         const __nv_bfloat16* &Vpage) -> bool {
        int kb = (t >= p_begin && t < p_end) ? blk_row[t] : -1;
        if (kb < 0) return false;
        int kvs = kb * BLK_N;
        if (kvs >= seq_len_k) return false;
        kvv = min(BLK_N, seq_len_k - kvs);
        int page = bt_row[kb];
        Kpage = Kbase + (int64_t)page * page_stride;
        Vpage = Vbase + (int64_t)page * page_stride;
        return true;
    };

    // ---- prologue: issue load of first page into stage 0 ----
    int cur = 0;
    int kvv_cur = 0; const __nv_bfloat16 *Kp0=nullptr,*Vp0=nullptr;
    bool valid_cur = page_meta(p_begin, kvv_cur, Kp0, Vp0);
    if (valid_cur) {
        load_page_async(SK(0), Kp0, kvv_cur, kvstride, tid, nthr);
        load_page_async(SV(0), Vp0, kvv_cur, kvstride, tid, nthr);
    }
    cp_async_commit();

    for (int t = p_begin; t < p_end; t++) {
        const int nxt = cur ^ 1;
#if NSTAGE > 1
        // prefetch next page into the other stage (overlaps this page's compute)
        int kvv_nxt = 0; const __nv_bfloat16 *Kpn=nullptr,*Vpn=nullptr;
        bool valid_nxt = (t+1 < p_end) && page_meta(t+1, kvv_nxt, Kpn, Vpn);
        if (valid_nxt) {
            load_page_async(SK(nxt), Kpn, kvv_nxt, kvstride, tid, nthr);
            load_page_async(SV(nxt), Vpn, kvv_nxt, kvstride, tid, nthr);
        }
        cp_async_commit();
        cp_async_wait(1);     // wait until only the just-issued (next) group remains
        __syncthreads();
#else
        // single buffer: wait for this page (Q+page in flight), no prefetch.
        cp_async_wait0();
        __syncthreads();
        int kvv_nxt = 0; const __nv_bfloat16 *Kpn=nullptr,*Vpn=nullptr;
        bool valid_nxt = false;
#endif

        const int kvv = kvv_cur;
        const bool valid = valid_cur;
        __nv_bfloat16 *sK = SK(cur);
        __nv_bfloat16 *sV = SV(cur);
        // advance pipeline state for next iteration
        cur = nxt; kvv_cur = kvv_nxt; valid_cur = valid_nxt;

        // Only warp 0 computes the 16 q-heads x 64 keys; warps 1-3 just load.
        if (wid == 0) {
        const __nv_bfloat16 * __restrict__ sKc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sK, 16));
        float Sr[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++) Sr[i][0]=Sr[i][1]=Sr[i][2]=Sr[i][3]=0.f;

        uint32_t b0n = lds_u32(&sKc_a[grp_hdim + sub2]);
        uint32_t b1n = lds_u32(&sKc_a[grp_hdim + sub2 + 8]);
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int qoff = ks * MMA_K_BF16;
            uint32_t a0 = lds_u32(&sQ_a[qbase0 + qoff + sub2]);
            uint32_t a1 = lds_u32(&sQ_a[qbase1 + qoff + sub2]);
            uint32_t a2 = lds_u32(&sQ_a[qbase0 + qoff + sub2 + 8]);
            uint32_t a3 = lds_u32(&sQ_a[qbase1 + qoff + sub2 + 8]);
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t b0 = b0n, b1 = b1n;
                hmma_bf16_m16n8k16(Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3],
                    a0,a1,a2,a3,b0,b1, Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3]);
                if (nt < 7) {
                    const int kr = ((nt+1)*MMA_N + grp)*HEAD_DIM + qoff;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                } else if (ks < 7) {
                    const int kr = grp_hdim + (ks+1)*MMA_K_BF16;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            Sr[i][0]*=softmax_scale; Sr[i][1]*=softmax_scale;
            Sr[i][2]*=softmax_scale; Sr[i][3]*=softmax_scale;
        }
        if (kvv < BLK_N) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int n0 = nt*MMA_N + sub*2, n1 = n0+1;
                if (n0 >= kvv) { Sr[nt][0]=-INFINITY; Sr[nt][2]=-INFINITY; }
                if (n1 >= kvv) { Sr[nt][1]=-INFINITY; Sr[nt][3]=-INFINITY; }
            }
        }
        // ---- online softmax ----
        float mx0=-INFINITY, mx1=-INFINITY;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            mx0=fmaxf(mx0,fmaxf(Sr[i][0],Sr[i][1]));
            mx1=fmaxf(mx1,fmaxf(Sr[i][2],Sr[i][3]));
        }
        { float t;
          t=__shfl_xor_sync(0xffffffff,mx0,1); mx0=fmaxf(mx0,t);
          t=__shfl_xor_sync(0xffffffff,mx0,2); mx0=fmaxf(mx0,t);
          t=__shfl_xor_sync(0xffffffff,mx1,1); mx1=fmaxf(mx1,t);
          t=__shfl_xor_sync(0xffffffff,mx1,2); mx1=fmaxf(mx1,t); }
        float mn0=fmaxf(rm[0],mx0), mn1=fmaxf(rm[1],mx1);
        float s0=(rm[0]==-INFINITY)?0.f:exp2f(LG2E*(rm[0]-mn0));
        float s1=(rm[1]==-INFINITY)?0.f:exp2f(LG2E*(rm[1]-mn1));
        rl[0]*=s0; rl[1]*=s1;
        #pragma unroll
        for (int d=0;d<16;d++){ Oa[d][0]*=s0;Oa[d][1]*=s0;Oa[d][2]*=s1;Oa[d][3]*=s1; }
        const bool dead0=(mn0==-INFINITY), dead1=(mn1==-INFINITY);
        float ls0=0.f, ls1=0.f;
        #pragma unroll
        for (int i=0;i<8;i++){
            Sr[i][0]=dead0?0.f:exp2f(LG2E*(Sr[i][0]-mn0));
            Sr[i][1]=dead0?0.f:exp2f(LG2E*(Sr[i][1]-mn0));
            Sr[i][2]=dead1?0.f:exp2f(LG2E*(Sr[i][2]-mn1));
            Sr[i][3]=dead1?0.f:exp2f(LG2E*(Sr[i][3]-mn1));
            ls0+=Sr[i][0]+Sr[i][1]; ls1+=Sr[i][2]+Sr[i][3];
        }
        { float t;
          t=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=t;
          t=__shfl_xor_sync(0xffffffff,ls0,2); ls0+=t;
          t=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=t;
          t=__shfl_xor_sync(0xffffffff,ls1,2); ls1+=t; }
        rm[0]=mn0; rm[1]=mn1; rl[0]+=ls0; rl[1]+=ls1;

        // ---- P -> fp8 in sP_all (BLK_N=64 wide) ----
        uint8_t *sP = sP_all;
        #pragma unroll
        for (int nt=0;nt<8;nt++){
            int kv_col = nt*MMA_N + sub*2;
            *reinterpret_cast<uint16_t*>(&sP[grp*BLK_N + kv_col])     = cvt_2f32_to_e4m3x2(Sr[nt][0],Sr[nt][1]);
            *reinterpret_cast<uint16_t*>(&sP[(grp+8)*BLK_N + kv_col]) = cvt_2f32_to_e4m3x2(Sr[nt][2],Sr[nt][3]);
        }
        __syncwarp();
        // ---- PV : O[16 x 128] (QMMA fp8 m16n8k32, 2 ns x 16 dt) ----
        const __nv_bfloat16 * __restrict__ sVc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sV, 16));
        #pragma unroll
        for (int ns=0; ns<2; ns++){
            const int pa_base0 = grp*BLK_N + ns*MMA_K_FP8;
            const int pa_base1 = (grp+8)*BLK_N + ns*MMA_K_FP8;
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4]);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4]);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4 + 16]);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4 + 16]);
            const int vr_lo = ns*MMA_K_FP8 + sub*4;
            const int vr_hi = vr_lo + 16;
            int vcol = grp;
            #define VLD(rb,vc) f32x4_to_e4m3x4( \
                __bfloat162float(sVc_a[(rb)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+1)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+2)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+3)*HEAD_DIM+(vc)]))
            uint32_t vb0n = VLD(vr_lo, vcol);
            uint32_t vb1n = VLD(vr_hi, vcol);
            #pragma unroll
            for (int dt=0; dt<16; dt++){
                uint32_t vb0=vb0n, vb1=vb1n;
                qmma_sf_fp8_m16n8k32(Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3],
                    pa0,pa1,pa2,pa3,vb0,vb1, Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3]);
                if (dt<15){
                    vcol=(dt+1)*MMA_N+grp;
                    vb0n = VLD(vr_lo, vcol); vb1n = VLD(vr_hi, vcol);
                }
            }
            #undef VLD
        }
        } // wid==0
        __syncthreads();
#if NSTAGE == 1
        // single buffer: issue next page's load now (after compute released sK/sV)
        if (t + 1 < p_end) {
            int kv2 = 0; const __nv_bfloat16 *Kp2=nullptr,*Vp2=nullptr;
            bool v2 = page_meta(t+1, kv2, Kp2, Vp2);
            if (v2) { load_page_async(SK(0), Kp2, kv2, kvstride, tid, nthr);
                      load_page_async(SV(0), Vp2, kv2, kvstride, tid, nthr); }
            cp_async_commit();
            kvv_cur = kv2; valid_cur = v2;
        }
#endif
    }

    // ===== epilogue: warp0 writes the chunk partial directly (no cross-warp) =====
    if (wid == 0) {
        const int64_t head_base = ((int64_t)req * num_heads_kv + hkv) * GQA;
        const int r0 = grp, r1 = grp + 8;
        #pragma unroll
        for (int dt=0; dt<16; dt++){
            int c0 = dt*MMA_N + sub*2, c1 = c0+1;
            int64_t ob0 = ((head_base + r0) * split_chunks + chunk) * HEAD_DIM;
            int64_t ob1 = ((head_base + r1) * split_chunks + chunk) * HEAD_DIM;
            O_part[ob0 + c0]=__float2bfloat16(Oa[dt][0]); O_part[ob0 + c1]=__float2bfloat16(Oa[dt][1]);
            O_part[ob1 + c0]=__float2bfloat16(Oa[dt][2]); O_part[ob1 + c1]=__float2bfloat16(Oa[dt][3]);
        }
        if (sub==0){
            M_part[(head_base + r0) * split_chunks + chunk] = rm[0];
            M_part[(head_base + r1) * split_chunks + chunk] = rm[1];
            L_part[(head_base + r0) * split_chunks + chunk] = rl[0];
            L_part[(head_base + r1) * split_chunks + chunk] = rl[1];
        }
    }
}

// ===========================================================================
// MERGE KERNEL v2 — multi-head per block to hide the O-load latency chain.
//   The single-head merge is LATENCY-bound: each block reduces 1 head's 128
//   dims over `split_chunks` partials, ~1 warp/scheduler, achieved occ 4.3%
//   vs 83% theoretical (ncu) => warps starved, the 32-chunk O-load chain is
//   exposed. Packing HQPB q-heads into one block gives HQPB independent
//   load chains per block (more eligible warps => better latency hiding)
//   while keeping each thread's coalesced O read. Grid = (R, Hq/HQPB, 1),
//   block = HQPB * (HEAD_DIM/2) threads laid out [head][dim-half-thread].
//   Each thread owns 2 head-dims (vectorized bf16x2 O load).
// ===========================================================================
#ifndef MERGE_HQPB
#define MERGE_HQPB 4
#endif
extern "C" __global__ void __launch_bounds__(MERGE_HQPB * (HEAD_DIM/2))
sm120_fmha_decode_merge_bf16_v2(
    const __nv_bfloat16 * __restrict__ O_part, // [R, Hkv, GQA, C, 128] bf16
    const float * __restrict__ M_part,        // [R, Hkv, GQA, C]
    const float * __restrict__ L_part,        // [R, Hkv, GQA, C]
    __nv_bfloat16 * __restrict__ O,           // [R, Hq, 128]
    float * __restrict__ LSE,                 // [R, Hq]  (optional)
    int split_chunks,
    int num_heads_q,
    int num_heads_kv)
{
    constexpr int HQPB = MERGE_HQPB;
    constexpr int DT   = HEAD_DIM / 2;        // threads per head (each owns 2 dims)
    const int lane    = threadIdx.x % DT;     // 0..63 -> dims {2*lane, 2*lane+1}
    const int hslot   = threadIdx.x / DT;     // 0..HQPB-1
    const int req     = blockIdx.x;
    const int hq      = blockIdx.y * HQPB + hslot;
    if (hq >= num_heads_q) return;
    const int gqa = num_heads_q / num_heads_kv;
    const int hkv = hq / gqa;
    const int g   = hq % gqa;
    const int d0  = lane * 2;                  // first of this thread's 2 dims
    constexpr float LG2E = 1.4426950408889634f;

    const int64_t head_row = ((int64_t)req*num_heads_kv + hkv)*gqa + g;
    const float * __restrict__ Mc = M_part + head_row*split_chunks;
    const float * __restrict__ Lc = L_part + head_row*split_chunks;
    const __nv_bfloat16 * __restrict__ Oc = O_part + head_row*split_chunks*HEAD_DIM;

    // gmax + denom over chunks (each thread recomputes from regs; M/L are L2-hot
    // and tiny — cheaper than a smem handshake at this low occupancy).
    float gmax = -INFINITY;
    #pragma unroll 4
    for (int c = 0; c < split_chunks; c++) gmax = fmaxf(gmax, Mc[c]);
    float denom = 0.f;
    // 4-way ILP over the 2-dim O loads (bf16x2 => 32-bit coalesced loads).
    float a0=0.f,a1=0.f,b0=0.f,b1=0.f;   // a*: dim d0, b*: dim d0+1 (2 accs each)
    int c = 0;
    for (; c + 2 <= split_chunks; c += 2) {
        float w0 = (Mc[c]   != -INFINITY) ? exp2f(LG2E*(Mc[c]  -gmax)) : 0.f;
        float w1 = (Mc[c+1] != -INFINITY) ? exp2f(LG2E*(Mc[c+1]-gmax)) : 0.f;
        float l0 = Lc[c], l1 = Lc[c+1];
        denom += (l0>0.f?l0*w0:0.f) + (l1>0.f?l1*w1:0.f);
        __nv_bfloat162 p0 = *reinterpret_cast<const __nv_bfloat162*>(Oc + (int64_t)c*HEAD_DIM + d0);
        __nv_bfloat162 p1 = *reinterpret_cast<const __nv_bfloat162*>(Oc + (int64_t)(c+1)*HEAD_DIM + d0);
        float2 f0 = __bfloat1622float2(p0);
        float2 f1 = __bfloat1622float2(p1);
        a0 += f0.x*w0; a1 += f1.x*w1;
        b0 += f0.y*w0; b1 += f1.y*w1;
    }
    for (; c < split_chunks; c++) {
        float w = (Mc[c] != -INFINITY) ? exp2f(LG2E*(Mc[c]-gmax)) : 0.f;
        float l = Lc[c]; denom += (l>0.f?l*w:0.f);
        __nv_bfloat162 p = *reinterpret_cast<const __nv_bfloat162*>(Oc + (int64_t)c*HEAD_DIM + d0);
        float2 f = __bfloat1622float2(p);
        a0 += f.x*w; b0 += f.y*w;
    }
    float inv = (denom > 0.f) ? (1.f/denom) : 0.f;
    __nv_bfloat162 res = __floats2bfloat162_rn((a0+a1)*inv, (b0+b1)*inv);
    *reinterpret_cast<__nv_bfloat162*>(O + ((int64_t)req*num_heads_q + hq)*HEAD_DIM + d0) = res;
    if (lane == 0 && LSE != nullptr)
        LSE[(int64_t)req*num_heads_q + hq] = (denom > 0.f) ? (gmax + logf(denom)) : -INFINITY;
}

// ===========================================================================
// MERGE KERNEL — flash-decoding LSE merge across split-K chunks.
//   grid = (num_requests, Hq).  block = HEAD_DIM threads (128).
//   For q-head hq (kv-head hkv=hq/GQA, local g=hq%GQA), read all chunks'
//   partial (M, L, O[.,g,.]) and combine into final O[req, hq, :].
// ===========================================================================
extern "C" __global__ void
sm120_fmha_decode_merge_bf16(
    const __nv_bfloat16 * __restrict__ O_part, // [R, C, Hkv, GQA, 128] bf16
    const float * __restrict__ M_part,        // [R, C, Hkv, GQA]
    const float * __restrict__ L_part,        // [R, C, Hkv, GQA]
    __nv_bfloat16 * __restrict__ O,           // [R, Hq, 128]
    float * __restrict__ LSE,                 // [R, Hq]  (optional)
    int split_chunks,
    int num_heads_q,
    int num_heads_kv)
{
    const int req = blockIdx.x;
    const int hq  = blockIdx.y;
    const int gqa = num_heads_q / num_heads_kv;
    const int hkv = hq / gqa;
    const int g   = hq % gqa;
    const int d   = blockIdx.z * blockDim.x + threadIdx.x;   // head-dim split over grid.z

    constexpr float LG2E = 1.4426950408889634f;

    // layout [R, Hkv, GQA, C]: this head's chunk-partials are contiguous.
    const int64_t head_row = ((int64_t)req*num_heads_kv + hkv)*gqa + g;
    const float * __restrict__ Mc = M_part + head_row*split_chunks;
    const float * __restrict__ Lc = L_part + head_row*split_chunks;
    const __nv_bfloat16 * __restrict__ Oc = O_part + head_row*split_chunks*HEAD_DIM;

    // Precompute per-chunk rescale weights in smem once (M/L are tiny), then the
    // per-thread O loop reads only coalesced O_part.
    extern __shared__ float sw[];           // [split_chunks] O rescale weights
    float gmax = -INFINITY;
    for (int c = 0; c < split_chunks; c++) gmax = fmaxf(gmax, Mc[c]);
    float denom = 0.f;
    for (int c = threadIdx.x; c < split_chunks; c += blockDim.x) {
        float m = Mc[c], l = Lc[c];
        float w = (m != -INFINITY) ? exp2f(LG2E*(m - gmax)) : 0.f;
        sw[c] = w;
    }
    __syncthreads();
    for (int c = 0; c < split_chunks; c++) {
        float l = Lc[c];
        denom += (l > 0.f) ? l * sw[c] : 0.f;
    }
    // Unroll with 4 independent accumulators to break the load->add dependency
    // chain and expose memory-level parallelism (issue many O loads in flight).
    float a0=0.f, a1=0.f, a2=0.f, a3=0.f;
    int c = 0;
    for (; c + 4 <= split_chunks; c += 4) {
        float v0 = __bfloat162float(Oc[(int64_t)(c+0)*HEAD_DIM + d]);
        float v1 = __bfloat162float(Oc[(int64_t)(c+1)*HEAD_DIM + d]);
        float v2 = __bfloat162float(Oc[(int64_t)(c+2)*HEAD_DIM + d]);
        float v3 = __bfloat162float(Oc[(int64_t)(c+3)*HEAD_DIM + d]);
        a0 += v0 * sw[c+0]; a1 += v1 * sw[c+1];
        a2 += v2 * sw[c+2]; a3 += v3 * sw[c+3];
    }
    for (; c < split_chunks; c++)
        a0 += __bfloat162float(Oc[(int64_t)c*HEAD_DIM + d]) * sw[c];
    float acc = (a0 + a1) + (a2 + a3);
    float inv = (denom > 0.f) ? (1.f/denom) : 0.f;
    O[((int64_t)req*num_heads_q + hq)*HEAD_DIM + d] = __float2bfloat16(acc * inv);
    if (d == 0 && LSE != nullptr)
        LSE[(int64_t)req*num_heads_q + hq] = (denom > 0.f) ? (gmax + logf(denom)) : -INFINITY;
}

// ===========================================================================
// PAGE-128 SPLIT-K PARTIAL KERNEL  (primary optimization)
//
//   The M3 paged KV cache is NATIVE 128-token pages (page == sparse block ==
//   128). The page-64 partial above splits each 128-block into two 64-pages,
//   doubling the cp.async loads, the ldmatrix round-trips, and the smem-stage
//   __syncthreads vs Triton's page-128 tiles. At bs1 this kernel is
//   global-load-latency bound, so halving the load/sync traffic is the win.
//
//   This variant consumes ONE 128-page as the inner-K tile (P128_N=128):
//     - QK: 16 MMA_N=8 n-tiles (vs 8) over 8 ks-steps of head_dim.
//     - PV: 4 ns-groups of m16n8k32 (128 keys = 4*k32) (vs 2).
//   Same numerics (bf16 QK, fp8 block-scaled PV, fp32 accum, bf16 partial-out).
//
//   K/V cache: [num_pages, 128, Hkv, 128] bf16 (page-128 native layout).
//   block_ids: [R, topk] selected LOGICAL 128-pages (-1 pad).
//   block_table: [R, max_logical_blocks] logical->physical 128-page map.
// ===========================================================================
static constexpr int P128_N = 128;            // keys per 128-page

// Load one P128_N x HEAD_DIM tile for one kv head, cooperatively over nthr.
__device__ __forceinline__ void load_page128_async(
    __nv_bfloat16 *smem_tile, const __nv_bfloat16 *gmem,
    int num_valid_rows, int stride_row, int tid, int nthr) {
    const int elems_per_copy = 8;
    const int total_elems = P128_N * HEAD_DIM;
    for (int e = tid * elems_per_copy; e < total_elems; e += nthr * elems_per_copy) {
        int row = e / HEAD_DIM, col = e % HEAD_DIM;
        __nv_bfloat16 *dst = smem_tile + row * HEAD_DIM + col;
        if (row < num_valid_rows) cp_async_16b(dst, gmem + row * stride_row + col);
        else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
}

// Padded smem row stride for the ldmatrix path: HEAD_DIM + 8 bf16. The +8
// (one 16B ldmatrix granule) shifts each row by 4 banks so the 16 per-lane
// addresses an ldmatrix issues land in distinct banks (kills the 5.4-way
// conflict the plain stride-128 layout produces). cp.async stores into the
// padded layout; ldmatrix reads from it. Used by the *_p128_ldsm kernel.
static constexpr int LDSM_PAD   = 8;
static constexpr int LDSM_KVLD  = HEAD_DIM + LDSM_PAD;   // K/V row stride (key->dim)
static constexpr int LDSM_PLD   = P128_N  + LDSM_PAD;    // P row stride (head->key)

// cp.async one P128_N x HEAD_DIM tile into a PADDED smem buffer (stride LDSM_KVLD).
__device__ __forceinline__ void load_page128_async_pad(
    __nv_bfloat16 *smem_tile, const __nv_bfloat16 *gmem,
    int num_valid_rows, int stride_row, int tid, int nthr) {
    const int total_elems = P128_N * HEAD_DIM;
    for (int e = tid * 8; e < total_elems; e += nthr * 8) {
        int row = e / HEAD_DIM, col = e % HEAD_DIM;
        __nv_bfloat16 *dst = smem_tile + row * LDSM_KVLD + col;
        if (row < num_valid_rows) cp_async_16b(dst, gmem + row * stride_row + col);
        else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
    }
}

// NSTAGE128=2 double-buffers two 128-pages (K+V = 64KB/stage); at 99KB optin
// smem that is 134KB and won't launch, so the page-128 tile is single-buffered
// by default (71KB, fits). The page is large enough that the 16-step QK + 64-step
// PV compute hides most of the next-page load latency anyway.
#ifndef NSTAGE128
#define NSTAGE128 1
#endif

extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
sm120_fmha_decode_partial_p128(
    const __nv_bfloat16 * __restrict__ Q,     // [num_requests, Hq, 128]
    const __nv_bfloat16 * __restrict__ Kc,    // [num_pages, 128, Hkv, 128]
    const __nv_bfloat16 * __restrict__ Vc,    // [num_pages, 128, Hkv, 128]
    __nv_bfloat16 * __restrict__ O_part,      // [R, Hkv, GQA, C, 128] bf16
    float * __restrict__ M_part,              // [R, Hkv, GQA, C]
    float * __restrict__ L_part,              // [R, Hkv, GQA, C]
    const int * __restrict__ block_ids,       // [R, topk] logical 128-pages (-1 pad)
    const int * __restrict__ block_table,     // [R, max_logical_blocks]
    int max_logical_blocks,
    int topk,
    int seq_len_k,
    int num_heads_q,
    int num_heads_kv,
    float softmax_scale,
    int split_chunks,
    int pages_per_chunk)
{
    extern __shared__ __nv_bfloat16 smem_raw[];

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    const int grp = lid / 4;     // 0..7  -> query-row pair {grp, grp+8}
    const int sub = lid % 4;     // 0..3  -> key/col pair
    const int nthr = NUM_WARPS * 32;

    const int hkv   = blockIdx.x;
    const int chunk = blockIdx.y;
    const int req   = blockIdx.z;

    const int qstride     = num_heads_q  * HEAD_DIM;
    const int kvstride    = num_heads_kv * HEAD_DIM;            // token stride in a page
    const int page_stride = P128_N * num_heads_kv * HEAD_DIM;   // 128-page stride

    const int hq0 = hkv * GQA;
    const __nv_bfloat16 *Qp    = Q + (int64_t)req * qstride + hq0 * HEAD_DIM;
    const __nv_bfloat16 *Kbase = Kc + hkv * HEAD_DIM;
    const __nv_bfloat16 *Vbase = Vc + hkv * HEAD_DIM;
    const int *bt_row  = block_table + req * max_logical_blocks;
    const int *blk_row = block_ids   + req * topk;

    // SMEM: sQ [16x128] + NSTAGE128 x (sK[128x128] sV[128x128]) + sP[16x128]
    __nv_bfloat16 *sQ  = smem_raw;
    __nv_bfloat16 *sKV = sQ + GQA * HEAD_DIM;
    const int STAGE_ELEMS = 2 * P128_N * HEAD_DIM;            // K+V per stage
    uint8_t *sP_all = reinterpret_cast<uint8_t*>(sKV + NSTAGE128 * STAGE_ELEMS);
    #define SK128(stage) (sKV + ((stage) % NSTAGE128) * STAGE_ELEMS)
    #define SV128(stage) (sKV + ((stage) % NSTAGE128) * STAGE_ELEMS + P128_N * HEAD_DIM)

    // Q load (overlaps first page load).
    {
        const int elems_per_copy = 8;
        const int total = GQA * HEAD_DIM;
        for (int e = tid * elems_per_copy; e < total; e += nthr * elems_per_copy) {
            cp_async_16b(sQ + e, Qp + e);
        }
        cp_async_commit();
    }

    const int sub2     = sub * 2;
    const int qbase0   = grp * HEAD_DIM;
    const int qbase1   = (grp + 8) * HEAD_DIM;
    const int grp_hdim = grp * HEAD_DIM;
    const __nv_bfloat16 * __restrict__ sQ_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sQ, 16));

    float Oa[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) Oa[i][0]=Oa[i][1]=Oa[i][2]=Oa[i][3]=0.f;
    float rm[2] = {-INFINITY, -INFINITY};
    float rl[2] = {0.f, 0.f};
    constexpr float LG2E = 1.4426950408889634f;

    const int p_begin = chunk * pages_per_chunk;
    const int p_end   = min(p_begin + pages_per_chunk, topk);

    auto page_meta = [&](int t, int &kvv, const __nv_bfloat16* &Kpage,
                         const __nv_bfloat16* &Vpage) -> bool {
        int kb = (t >= p_begin && t < p_end) ? blk_row[t] : -1;
        if (kb < 0) return false;
        int kvs = kb * P128_N;
        if (kvs >= seq_len_k) return false;
        kvv = min(P128_N, seq_len_k - kvs);
        int page = bt_row[kb];
        Kpage = Kbase + (int64_t)page * page_stride;
        Vpage = Vbase + (int64_t)page * page_stride;
        return true;
    };

    int cur = 0;
    int kvv_cur = 0; const __nv_bfloat16 *Kp0=nullptr,*Vp0=nullptr;
    bool valid_cur = page_meta(p_begin, kvv_cur, Kp0, Vp0);
    if (valid_cur) {
        load_page128_async(SK128(0), Kp0, kvv_cur, kvstride, tid, nthr);
        load_page128_async(SV128(0), Vp0, kvv_cur, kvstride, tid, nthr);
    }
    cp_async_commit();

    for (int t = p_begin; t < p_end; t++) {
        const int nxt = cur ^ 1;
#if NSTAGE128 > 1
        int kvv_nxt = 0; const __nv_bfloat16 *Kpn=nullptr,*Vpn=nullptr;
        bool valid_nxt = (t+1 < p_end) && page_meta(t+1, kvv_nxt, Kpn, Vpn);
        if (valid_nxt) {
            load_page128_async(SK128(nxt), Kpn, kvv_nxt, kvstride, tid, nthr);
            load_page128_async(SV128(nxt), Vpn, kvv_nxt, kvstride, tid, nthr);
        }
        cp_async_commit();
        cp_async_wait(1);
        __syncthreads();
#else
        cp_async_wait0();
        __syncthreads();
        int kvv_nxt = 0; const __nv_bfloat16 *Kpn=nullptr,*Vpn=nullptr;
        bool valid_nxt = false;
#endif
        const int kvv = kvv_cur;
        __nv_bfloat16 *sK = SK128(cur);
        __nv_bfloat16 *sV = SV128(cur);
        cur = nxt; kvv_cur = kvv_nxt; valid_cur = valid_nxt;

        if (wid == 0) {
        const __nv_bfloat16 * __restrict__ sKc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sK, 16));
        // QK: 16 n-tiles (128 keys) over 8 ks-steps.
        float Sr[16][4];
        #pragma unroll
        for (int i = 0; i < 16; i++) Sr[i][0]=Sr[i][1]=Sr[i][2]=Sr[i][3]=0.f;

        uint32_t b0n = lds_u32(&sKc_a[grp_hdim + sub2]);
        uint32_t b1n = lds_u32(&sKc_a[grp_hdim + sub2 + 8]);
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int qoff = ks * MMA_K_BF16;
            uint32_t a0 = lds_u32(&sQ_a[qbase0 + qoff + sub2]);
            uint32_t a1 = lds_u32(&sQ_a[qbase1 + qoff + sub2]);
            uint32_t a2 = lds_u32(&sQ_a[qbase0 + qoff + sub2 + 8]);
            uint32_t a3 = lds_u32(&sQ_a[qbase1 + qoff + sub2 + 8]);
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                uint32_t b0 = b0n, b1 = b1n;
                hmma_bf16_m16n8k16(Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3],
                    a0,a1,a2,a3,b0,b1, Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3]);
                if (nt < 15) {
                    const int kr = ((nt+1)*MMA_N + grp)*HEAD_DIM + qoff;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                } else if (ks < 7) {
                    const int kr = grp_hdim + (ks+1)*MMA_K_BF16;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            Sr[i][0]*=softmax_scale; Sr[i][1]*=softmax_scale;
            Sr[i][2]*=softmax_scale; Sr[i][3]*=softmax_scale;
        }
        if (kvv < P128_N) {
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                int n0 = nt*MMA_N + sub*2, n1 = n0+1;
                if (n0 >= kvv) { Sr[nt][0]=-INFINITY; Sr[nt][2]=-INFINITY; }
                if (n1 >= kvv) { Sr[nt][1]=-INFINITY; Sr[nt][3]=-INFINITY; }
            }
        }
        // ---- online softmax over 16 n-tiles ----
        float mx0=-INFINITY, mx1=-INFINITY;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            mx0=fmaxf(mx0,fmaxf(Sr[i][0],Sr[i][1]));
            mx1=fmaxf(mx1,fmaxf(Sr[i][2],Sr[i][3]));
        }
        { float t;
          t=__shfl_xor_sync(0xffffffff,mx0,1); mx0=fmaxf(mx0,t);
          t=__shfl_xor_sync(0xffffffff,mx0,2); mx0=fmaxf(mx0,t);
          t=__shfl_xor_sync(0xffffffff,mx1,1); mx1=fmaxf(mx1,t);
          t=__shfl_xor_sync(0xffffffff,mx1,2); mx1=fmaxf(mx1,t); }
        float mn0=fmaxf(rm[0],mx0), mn1=fmaxf(rm[1],mx1);
        float s0=(rm[0]==-INFINITY)?0.f:exp2f(LG2E*(rm[0]-mn0));
        float s1=(rm[1]==-INFINITY)?0.f:exp2f(LG2E*(rm[1]-mn1));
        rl[0]*=s0; rl[1]*=s1;
        #pragma unroll
        for (int d=0;d<16;d++){ Oa[d][0]*=s0;Oa[d][1]*=s0;Oa[d][2]*=s1;Oa[d][3]*=s1; }
        const bool dead0=(mn0==-INFINITY), dead1=(mn1==-INFINITY);
        float ls0=0.f, ls1=0.f;
        #pragma unroll
        for (int i=0;i<16;i++){
            Sr[i][0]=dead0?0.f:exp2f(LG2E*(Sr[i][0]-mn0));
            Sr[i][1]=dead0?0.f:exp2f(LG2E*(Sr[i][1]-mn0));
            Sr[i][2]=dead1?0.f:exp2f(LG2E*(Sr[i][2]-mn1));
            Sr[i][3]=dead1?0.f:exp2f(LG2E*(Sr[i][3]-mn1));
            ls0+=Sr[i][0]+Sr[i][1]; ls1+=Sr[i][2]+Sr[i][3];
        }
        { float t;
          t=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=t;
          t=__shfl_xor_sync(0xffffffff,ls0,2); ls0+=t;
          t=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=t;
          t=__shfl_xor_sync(0xffffffff,ls1,2); ls1+=t; }
        rm[0]=mn0; rm[1]=mn1; rl[0]+=ls0; rl[1]+=ls1;

        // ---- P -> fp8 in sP_all (P128_N=128 wide per row) ----
        uint8_t *sP = sP_all;
        #pragma unroll
        for (int nt=0;nt<16;nt++){
            int kv_col = nt*MMA_N + sub*2;
            *reinterpret_cast<uint16_t*>(&sP[grp*P128_N + kv_col])     = cvt_2f32_to_e4m3x2(Sr[nt][0],Sr[nt][1]);
            *reinterpret_cast<uint16_t*>(&sP[(grp+8)*P128_N + kv_col]) = cvt_2f32_to_e4m3x2(Sr[nt][2],Sr[nt][3]);
        }
        __syncwarp();
        // ---- PV : O[16 x 128] (QMMA fp8 m16n8k32, 4 ns-groups x 16 dt) ----
        const __nv_bfloat16 * __restrict__ sVc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sV, 16));
        #pragma unroll
        for (int ns=0; ns<4; ns++){
            const int pa_base0 = grp*P128_N + ns*MMA_K_FP8;
            const int pa_base1 = (grp+8)*P128_N + ns*MMA_K_FP8;
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4]);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4]);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4 + 16]);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4 + 16]);
            const int vr_lo = ns*MMA_K_FP8 + sub*4;
            const int vr_hi = vr_lo + 16;
            int vcol = grp;
            #define VLD128(rb,vc) f32x4_to_e4m3x4( \
                __bfloat162float(sVc_a[(rb)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+1)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+2)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+3)*HEAD_DIM+(vc)]))
            uint32_t vb0n = VLD128(vr_lo, vcol);
            uint32_t vb1n = VLD128(vr_hi, vcol);
            #pragma unroll
            for (int dt=0; dt<16; dt++){
                uint32_t vb0=vb0n, vb1=vb1n;
                qmma_sf_fp8_m16n8k32(Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3],
                    pa0,pa1,pa2,pa3,vb0,vb1, Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3]);
                if (dt<15){
                    vcol=(dt+1)*MMA_N+grp;
                    vb0n = VLD128(vr_lo, vcol); vb1n = VLD128(vr_hi, vcol);
                }
            }
            #undef VLD128
        }
        } // wid==0
        __syncthreads();
#if NSTAGE128 == 1
        if (t + 1 < p_end) {
            int kv2 = 0; const __nv_bfloat16 *Kp2=nullptr,*Vp2=nullptr;
            bool v2 = page_meta(t+1, kv2, Kp2, Vp2);
            if (v2) { load_page128_async(SK128(0), Kp2, kv2, kvstride, tid, nthr);
                      load_page128_async(SV128(0), Vp2, kv2, kvstride, tid, nthr); }
            cp_async_commit();
            kvv_cur = kv2; valid_cur = v2;
        }
#endif
    }

    // ===== epilogue: warp0 writes the chunk partial =====
    if (wid == 0) {
        const int64_t head_base = ((int64_t)req * num_heads_kv + hkv) * GQA;
        const int r0 = grp, r1 = grp + 8;
        #pragma unroll
        for (int dt=0; dt<16; dt++){
            int c0 = dt*MMA_N + sub*2, c1 = c0+1;
            int64_t ob0 = ((head_base + r0) * split_chunks + chunk) * HEAD_DIM;
            int64_t ob1 = ((head_base + r1) * split_chunks + chunk) * HEAD_DIM;
            O_part[ob0 + c0]=__float2bfloat16(Oa[dt][0]); O_part[ob0 + c1]=__float2bfloat16(Oa[dt][1]);
            O_part[ob1 + c0]=__float2bfloat16(Oa[dt][2]); O_part[ob1 + c1]=__float2bfloat16(Oa[dt][3]);
        }
        if (sub==0){
            M_part[(head_base + r0) * split_chunks + chunk] = rm[0];
            M_part[(head_base + r1) * split_chunks + chunk] = rm[1];
            L_part[(head_base + r0) * split_chunks + chunk] = rl[0];
            L_part[(head_base + r1) * split_chunks + chunk] = rl[1];
        }
    }
}

// ===========================================================================
// PAGE-128 4-WARP PARTIAL KERNEL  (limiter-2 fix: use all 4 warps to compute)
//
//   The single-warp page-128 kernel above leaves warps 1-3 load-only. Triton's
//   decode runs the SAME 64-block geometry (page-128, GQA-16, NUM_TOPK_CHUNKS=
//   16 at bs1) but uses every warp's tensor core, so it is ~2.25x faster per
//   block. This kernel reclaims warps 1-3:
//
//     QK  split by KEYS  : warp w computes the 32 keys [32w,32w+32) -> 4 n-tiles
//                          (m16n8k16), real work, no zero-pad.
//     softmax            : per-warp partial max/sum reduced ACROSS the 4 warps
//                          via a tiny smem scratch (16 heads x scalars).
//     PV  split by HEAD-D: warp w computes output dims [32w,32w+32) over ALL 128
//                          keys -> 4 ns-groups (k32) x 4 dt = 16 QMMA/warp.
//                          Disjoint output dims => NO cross-warp O merge.
//
//   Online softmax across pages is preserved (multi-page chunks still correct).
//   Same numerics. K/V cache + ids identical to the single-warp page-128 path.
// ===========================================================================
extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
sm120_fmha_decode_partial_p128_4w(
    const __nv_bfloat16 * __restrict__ Q,     // [num_requests, Hq, 128]
    const __nv_bfloat16 * __restrict__ Kc,    // [num_pages, 128, Hkv, 128]
    const __nv_bfloat16 * __restrict__ Vc,    // [num_pages, 128, Hkv, 128]
    __nv_bfloat16 * __restrict__ O_part,      // [R, Hkv, GQA, C, 128] bf16
    float * __restrict__ M_part,              // [R, Hkv, GQA, C]
    float * __restrict__ L_part,              // [R, Hkv, GQA, C]
    const int * __restrict__ block_ids,
    const int * __restrict__ block_table,
    int max_logical_blocks,
    int topk,
    int seq_len_k,
    int num_heads_q,
    int num_heads_kv,
    float softmax_scale,
    int split_chunks,
    int pages_per_chunk)
{
    extern __shared__ __nv_bfloat16 smem_raw[];

    const int tid = threadIdx.x;
    const int wid = tid / 32;        // 0..3 : this warp owns keys [32w,32w+32)
    const int lid = tid % 32;
    const int grp = lid / 4;         // 0..7  -> query-row pair {grp, grp+8}
    const int sub = lid % 4;         // 0..3
    const int nthr = NUM_WARPS * 32;

    const int hkv   = blockIdx.x;
    const int chunk = blockIdx.y;
    const int req   = blockIdx.z;

    const int qstride     = num_heads_q  * HEAD_DIM;
    const int kvstride    = num_heads_kv * HEAD_DIM;
    const int page_stride = P128_N * num_heads_kv * HEAD_DIM;

    const int hq0 = hkv * GQA;
    const __nv_bfloat16 *Qp    = Q + (int64_t)req * qstride + hq0 * HEAD_DIM;
    const __nv_bfloat16 *Kbase = Kc + hkv * HEAD_DIM;
    const __nv_bfloat16 *Vbase = Vc + hkv * HEAD_DIM;
    const int *bt_row  = block_table + req * max_logical_blocks;
    const int *blk_row = block_ids   + req * topk;

    // SMEM: sQ[16x128] + sK[128x128] + sV[128x128] + sP[16x128] + sRed scratch.
    __nv_bfloat16 *sQ  = smem_raw;
    __nv_bfloat16 *sK  = sQ + GQA * HEAD_DIM;
    __nv_bfloat16 *sV  = sK + P128_N * HEAD_DIM;
    uint8_t *sP        = reinterpret_cast<uint8_t*>(sV + P128_N * HEAD_DIM);  // 16*128 B
    // reduction scratch: per-warp (max,sum) for the 16 rows. [NUM_WARPS][16][2]
    float *sRed = reinterpret_cast<float*>(sP + GQA * P128_N);

    {   // Q load
        const int total = GQA * HEAD_DIM;
        for (int e = tid * 8; e < total; e += nthr * 8) cp_async_16b(sQ + e, Qp + e);
        cp_async_commit();
    }

    const int sub2     = sub * 2;
    const int qbase0   = grp * HEAD_DIM;
    const int qbase1   = (grp + 8) * HEAD_DIM;
    const int key0     = wid * 32;          // first key of this warp
    const __nv_bfloat16 * __restrict__ sQ_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sQ, 16));

    // Per-warp output accumulator: this warp owns output dims [32w,32w+32) = 4
    // dt-tiles (8 dims each). Oa[4][4] : [dt][{r0c0,r0c1,r1c0,r1c1}].
    float Oa[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) Oa[i][0]=Oa[i][1]=Oa[i][2]=Oa[i][3]=0.f;
    float rm[2] = {-INFINITY, -INFINITY};
    float rl[2] = {0.f, 0.f};
    constexpr float LG2E = 1.4426950408889634f;

    const int p_begin = chunk * pages_per_chunk;
    const int p_end   = min(p_begin + pages_per_chunk, topk);

    auto page_meta = [&](int t, int &kvv, const __nv_bfloat16* &Kpage,
                         const __nv_bfloat16* &Vpage) -> bool {
        int kb = (t >= p_begin && t < p_end) ? blk_row[t] : -1;
        if (kb < 0) return false;
        int kvs = kb * P128_N;
        if (kvs >= seq_len_k) return false;
        kvv = min(P128_N, seq_len_k - kvs);
        int page = bt_row[kb];
        Kpage = Kbase + (int64_t)page * page_stride;
        Vpage = Vbase + (int64_t)page * page_stride;
        return true;
    };

    // K and V are committed as SEPARATE cp.async groups so QK (which only needs
    // K) can start as soon as K lands while V is still streaming. wait_group(1)
    // leaves the V group in flight; wait_group(0) before PV drains it.
    int kvv_cur = 0; const __nv_bfloat16 *Kp0=nullptr,*Vp0=nullptr;
    bool valid_cur = page_meta(p_begin, kvv_cur, Kp0, Vp0);
    if (valid_cur) load_page128_async(sK, Kp0, kvv_cur, kvstride, tid, nthr);
    cp_async_commit();                                   // group: K
    if (valid_cur) load_page128_async(sV, Vp0, kvv_cur, kvstride, tid, nthr);
    cp_async_commit();                                   // group: V

    const __nv_bfloat16 * __restrict__ sKc_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sK, 16));
    const __nv_bfloat16 * __restrict__ sVc_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sV, 16));

    for (int t = p_begin; t < p_end; t++) {
        cp_async_wait(1);          // K landed; V still in flight
        __syncthreads();
        const int kvv = kvv_cur;

        // ---- QK: this warp's 32 keys [key0,key0+32) -> 4 n-tiles ----
        float Sr[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) Sr[i][0]=Sr[i][1]=Sr[i][2]=Sr[i][3]=0.f;
        const int kgrp0 = (key0 + grp) * HEAD_DIM;        // row key0+grp
        uint32_t b0n = lds_u32(&sKc_a[kgrp0 + sub2]);
        uint32_t b1n = lds_u32(&sKc_a[kgrp0 + sub2 + 8]);
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int qoff = ks * MMA_K_BF16;
            uint32_t a0 = lds_u32(&sQ_a[qbase0 + qoff + sub2]);
            uint32_t a1 = lds_u32(&sQ_a[qbase1 + qoff + sub2]);
            uint32_t a2 = lds_u32(&sQ_a[qbase0 + qoff + sub2 + 8]);
            uint32_t a3 = lds_u32(&sQ_a[qbase1 + qoff + sub2 + 8]);
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                hmma_bf16_m16n8k16(Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3],
                    a0,a1,a2,a3,b0n,b1n, Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3]);
                if (nt < 3) {
                    const int kr = (key0 + (nt+1)*MMA_N + grp)*HEAD_DIM + qoff;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                } else if (ks < 7) {
                    const int kr = kgrp0 + (ks+1)*MMA_K_BF16;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            Sr[i][0]*=softmax_scale; Sr[i][1]*=softmax_scale;
            Sr[i][2]*=softmax_scale; Sr[i][3]*=softmax_scale;
        }
        // mask keys >= kvv (global key index = key0 + nt*8 + sub*2 {,+1})
        if (kvv < P128_N) {
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                int n0 = key0 + nt*MMA_N + sub*2, n1 = n0+1;
                if (n0 >= kvv) { Sr[nt][0]=-INFINITY; Sr[nt][2]=-INFINITY; }
                if (n1 >= kvv) { Sr[nt][1]=-INFINITY; Sr[nt][3]=-INFINITY; }
            }
        }
        // ---- per-warp partial row-max (over its 32 keys), then cross-warp ----
        float mx0=-INFINITY, mx1=-INFINITY;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            mx0=fmaxf(mx0,fmaxf(Sr[i][0],Sr[i][1]));
            mx1=fmaxf(mx1,fmaxf(Sr[i][2],Sr[i][3]));
        }
        { float tt;
          tt=__shfl_xor_sync(0xffffffff,mx0,1); mx0=fmaxf(mx0,tt);
          tt=__shfl_xor_sync(0xffffffff,mx0,2); mx0=fmaxf(mx0,tt);
          tt=__shfl_xor_sync(0xffffffff,mx1,1); mx1=fmaxf(mx1,tt);
          tt=__shfl_xor_sync(0xffffffff,mx1,2); mx1=fmaxf(mx1,tt); }
        // each warp's lane (sub==0) writes its partial max for the 8 grp-rows
        if (sub == 0) { sRed[(wid*GQA + grp)*2 + 0] = mx0; sRed[(wid*GQA + grp + 8)*2 + 0] = mx1; }
        __syncthreads();
        // cross-warp max for rows grp, grp+8
        float pmx0 = mx0, pmx1 = mx1;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            pmx0 = fmaxf(pmx0, sRed[(w*GQA + grp)*2 + 0]);
            pmx1 = fmaxf(pmx1, sRed[(w*GQA + grp + 8)*2 + 0]);
        }
        // online: combine with running max
        float mn0=fmaxf(rm[0],pmx0), mn1=fmaxf(rm[1],pmx1);
        float s0=(rm[0]==-INFINITY)?0.f:exp2f(LG2E*(rm[0]-mn0));
        float s1=(rm[1]==-INFINITY)?0.f:exp2f(LG2E*(rm[1]-mn1));
        rl[0]*=s0; rl[1]*=s1;
        #pragma unroll
        for (int d=0;d<4;d++){ Oa[d][0]*=s0;Oa[d][1]*=s0;Oa[d][2]*=s1;Oa[d][3]*=s1; }
        const bool dead0=(mn0==-INFINITY), dead1=(mn1==-INFINITY);
        // exp this warp's 32 P-values against the GLOBAL max mn, accumulate local sum
        float ls0=0.f, ls1=0.f;
        #pragma unroll
        for (int i=0;i<4;i++){
            Sr[i][0]=dead0?0.f:exp2f(LG2E*(Sr[i][0]-mn0));
            Sr[i][1]=dead0?0.f:exp2f(LG2E*(Sr[i][1]-mn0));
            Sr[i][2]=dead1?0.f:exp2f(LG2E*(Sr[i][2]-mn1));
            Sr[i][3]=dead1?0.f:exp2f(LG2E*(Sr[i][3]-mn1));
            ls0+=Sr[i][0]+Sr[i][1]; ls1+=Sr[i][2]+Sr[i][3];
        }
        { float tt;
          tt=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls0,2); ls0+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls1,2); ls1+=tt; }
        // each warp writes its partial denom; sum across warps
        if (sub == 0) { sRed[(wid*GQA + grp)*2 + 1] = ls0; sRed[(wid*GQA + grp + 8)*2 + 1] = ls1; }
        // ---- write this warp's 32 P-fp8 into shared sP[16 x 128] ----
        #pragma unroll
        for (int nt=0;nt<4;nt++){
            int kv_col = key0 + nt*MMA_N + sub*2;
            *reinterpret_cast<uint16_t*>(&sP[grp*P128_N + kv_col])     = cvt_2f32_to_e4m3x2(Sr[nt][0],Sr[nt][1]);
            *reinterpret_cast<uint16_t*>(&sP[(grp+8)*P128_N + kv_col]) = cvt_2f32_to_e4m3x2(Sr[nt][2],Sr[nt][3]);
        }
        __syncthreads();
        // sum partial denoms across warps
        float gl0=0.f, gl1=0.f;
        #pragma unroll
        for (int w=0; w<NUM_WARPS; w++){
            gl0 += sRed[(w*GQA + grp)*2 + 1];
            gl1 += sRed[(w*GQA + grp + 8)*2 + 1];
        }
        rm[0]=mn0; rm[1]=mn1; rl[0]+=gl0; rl[1]+=gl1;

        // V must have landed by now (it loaded during the whole QK+softmax).
        cp_async_wait0();
        __syncthreads();

        // ---- PV : this warp owns output dims [key0,key0+32) (4 dt of 8) ----
        // (reuse key0 as the head-dim base since 32 dims/warp == 32 keys/warp)
        const int dbase = wid * 32;
        #pragma unroll
        for (int ns=0; ns<4; ns++){
            const int pa_base0 = grp*P128_N + ns*MMA_K_FP8;
            const int pa_base1 = (grp+8)*P128_N + ns*MMA_K_FP8;
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4]);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4]);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4 + 16]);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4 + 16]);
            const int vr_lo = ns*MMA_K_FP8 + sub*4;
            const int vr_hi = vr_lo + 16;
            int vcol = dbase + grp;
            #define VLD4W(rb,vc) f32x4_to_e4m3x4( \
                __bfloat162float(sVc_a[(rb)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+1)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+2)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+3)*HEAD_DIM+(vc)]))
            uint32_t vb0n = VLD4W(vr_lo, vcol);
            uint32_t vb1n = VLD4W(vr_hi, vcol);
            #pragma unroll
            for (int dt=0; dt<4; dt++){
                uint32_t vb0=vb0n, vb1=vb1n;
                qmma_sf_fp8_m16n8k32(Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3],
                    pa0,pa1,pa2,pa3,vb0,vb1, Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3]);
                if (dt<3){
                    vcol=dbase+(dt+1)*MMA_N+grp;
                    vb0n = VLD4W(vr_lo, vcol); vb1n = VLD4W(vr_hi, vcol);
                }
            }
            #undef VLD4W
        }
        __syncthreads();   // protect sP/sRed reuse + sK/sV reload

        // issue next page load (single-buffer), K and V as separate groups.
        if (t + 1 < p_end) {
            int kv2 = 0; const __nv_bfloat16 *Kp2=nullptr,*Vp2=nullptr;
            bool v2 = page_meta(t+1, kv2, Kp2, Vp2);
            if (v2) load_page128_async(sK, Kp2, kv2, kvstride, tid, nthr);
            cp_async_commit();
            if (v2) load_page128_async(sV, Vp2, kv2, kvstride, tid, nthr);
            cp_async_commit();
            kvv_cur = kv2; valid_cur = v2;
        }
    }

    // ===== epilogue: each warp writes its 32 output dims =====
    const int64_t head_base = ((int64_t)req * num_heads_kv + hkv) * GQA;
    const int r0 = grp, r1 = grp + 8;
    const int dbase = wid * 32;
    #pragma unroll
    for (int dt=0; dt<4; dt++){
        int c0 = dbase + dt*MMA_N + sub*2, c1 = c0+1;
        int64_t ob0 = ((head_base + r0) * split_chunks + chunk) * HEAD_DIM;
        int64_t ob1 = ((head_base + r1) * split_chunks + chunk) * HEAD_DIM;
        O_part[ob0 + c0]=__float2bfloat16(Oa[dt][0]); O_part[ob0 + c1]=__float2bfloat16(Oa[dt][1]);
        O_part[ob1 + c0]=__float2bfloat16(Oa[dt][2]); O_part[ob1 + c1]=__float2bfloat16(Oa[dt][3]);
    }
    // M/L written once (warp 0, sub 0) — identical across warps.
    if (wid == 0 && sub == 0) {
        M_part[(head_base + r0) * split_chunks + chunk] = rm[0];
        M_part[(head_base + r1) * split_chunks + chunk] = rm[1];
        L_part[(head_base + r0) * split_chunks + chunk] = rl[0];
        L_part[(head_base + r1) * split_chunks + chunk] = rl[1];
    }
}

// ===========================================================================
// PAGE-128 cache, 64-KEY SUB-TILE, 4-WARP PARTIAL  (high-occupancy variant)
//
//   Reads the NATIVE page-128 cache but splits split-K at 64-KEY granularity:
//   a "sub-tile" is 64 keys = half a 128-page. With topk=16 selected pages =
//   32 sub-tiles, split_chunks can reach 32 -> 4 kvh * 32 = 128 blocks
//   (0.68/SM) instead of 64 (0.34/SM). This recovers the occupancy the
//   full-page-128 path lost while staying on the integration-ready 128-page
//   layout (blocker A/B resolved). 4 warps split the 64 keys (16 keys/warp).
//
//   chunk c -> sub-tile range; sub-tile s -> page s/2, in-page key offset
//   (s&1)*64. Each block loads only its 64-key half (kvstride rows).
// ===========================================================================
static constexpr int SUB_N = 64;     // keys per sub-tile

extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
sm120_fmha_decode_partial_p128_sub64(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ Kc,    // [num_pages, 128, Hkv, 128]
    const __nv_bfloat16 * __restrict__ Vc,
    __nv_bfloat16 * __restrict__ O_part,      // [R, Hkv, GQA, C, 128]
    float * __restrict__ M_part,
    float * __restrict__ L_part,
    const int * __restrict__ block_ids,
    const int * __restrict__ block_table,
    int max_logical_blocks,
    int topk,
    int seq_len_k,
    int num_heads_q,
    int num_heads_kv,
    float softmax_scale,
    int split_chunks,
    int sub_per_chunk,
    int dh_split,            // 1 or 2: split the 128 output dims across dh_split blocks
    int pv_bf16,             // 0: fp8 block-scaled QMMA PV; 1: native bf16 HMMA PV
    __nv_bfloat16 * __restrict__ O_final,  // [R, Hq, 128] (fused merge target, or null)
    int * __restrict__ chunk_done)         // [R*Hkv] arrival counters (fused), or null
{
    extern __shared__ __nv_bfloat16 smem_raw[];

    const int tid = threadIdx.x;
    const int wid = tid / 32;        // 0..3 : warp owns keys [16w,16w+16) of the 64-tile
    const int lid = tid % 32;
    const int grp = lid / 4;
    const int sub = lid % 4;
    const int nthr = NUM_WARPS * 32;

    const int hkv   = blockIdx.x;
    const int chunk = blockIdx.y;
    const int req   = blockIdx.z / dh_split;
    const int dh_id = blockIdx.z - req * dh_split;   // 0..dh_split-1
    const int dims_per_blk = HEAD_DIM / dh_split;    // 128 or 64
    const int odim0 = dh_id * dims_per_blk;          // first output dim of this block
    const int dt_cnt = dims_per_blk / 32;            // dt-tiles per warp (4 or 2)

    const int qstride     = num_heads_q  * HEAD_DIM;
    const int kvstride    = num_heads_kv * HEAD_DIM;
    const int page_stride = P128_N * num_heads_kv * HEAD_DIM;

    const int hq0 = hkv * GQA;
    const __nv_bfloat16 *Qp    = Q + (int64_t)req * qstride + hq0 * HEAD_DIM;
    const __nv_bfloat16 *Kbase = Kc + hkv * HEAD_DIM;
    const __nv_bfloat16 *Vbase = Vc + hkv * HEAD_DIM;
    const int *bt_row  = block_table + req * max_logical_blocks;
    const int *blk_row = block_ids   + req * topk;
    const int n_sub = topk * 2;                   // total 64-key sub-tiles

    // SMEM: sQ[16x128] + sK[64x128] + sV[64x128] + sP[16x64] + sRed.
    __nv_bfloat16 *sQ  = smem_raw;
    __nv_bfloat16 *sK  = sQ + GQA * HEAD_DIM;
    __nv_bfloat16 *sV  = sK + SUB_N * HEAD_DIM;
    uint8_t *sP        = reinterpret_cast<uint8_t*>(sV + SUB_N * HEAD_DIM);   // 16*64
    float *sRed        = reinterpret_cast<float*>(sP + GQA * SUB_N);
    // bf16 P scores for the native-bf16 PV path (16 heads x 64 keys).
    __nv_bfloat16 *sPbf = reinterpret_cast<__nv_bfloat16*>(sRed + NUM_WARPS * GQA * 2);

    {   const int total = GQA * HEAD_DIM;
        for (int e = tid * 8; e < total; e += nthr * 8) cp_async_16b(sQ + e, Qp + e);
        cp_async_commit();
    }

    const int sub2     = sub * 2;
    const int qbase0   = grp * HEAD_DIM;
    const int qbase1   = (grp + 8) * HEAD_DIM;
    const int key0     = wid * 16;             // first key of this warp (within 64-tile)
    const __nv_bfloat16 * __restrict__ sQ_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sQ, 16));

    float Oa[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) Oa[i][0]=Oa[i][1]=Oa[i][2]=Oa[i][3]=0.f;
    float rm[2] = {-INFINITY, -INFINITY};
    float rl[2] = {0.f, 0.f};
    constexpr float LG2E = 1.4426950408889634f;

    const int s_begin = chunk * sub_per_chunk;
    const int s_end   = min(s_begin + sub_per_chunk, n_sub);

    // sub-tile s -> (page = blk_row[s/2], in-page key offset (s&1)*64)
    auto sub_meta = [&](int s, int &kvv, const __nv_bfloat16* &Kp,
                        const __nv_bfloat16* &Vp) -> bool {
        if (s < s_begin || s >= s_end) return false;
        int pi = s >> 1, half = s & 1;
        int kb = blk_row[pi];
        if (kb < 0) return false;
        int kvs = kb * P128_N + half * SUB_N;       // absolute key start
        if (kvs >= seq_len_k) return false;
        kvv = min(SUB_N, seq_len_k - kvs);
        int page = bt_row[kb];
        const __nv_bfloat16 *base = Kbase + (int64_t)page * page_stride + half * SUB_N * kvstride;
        Kp = base;
        Vp = Vbase + (int64_t)page * page_stride + half * SUB_N * kvstride;
        return true;
    };

    int kvv_cur = 0; const __nv_bfloat16 *Kp0=nullptr,*Vp0=nullptr;
    bool valid_cur = sub_meta(s_begin, kvv_cur, Kp0, Vp0);
    if (valid_cur) {
        for (int e = tid*8; e < SUB_N*HEAD_DIM; e += nthr*8) {
            int row=e/HEAD_DIM, col=e%HEAD_DIM;
            if (row<kvv_cur) cp_async_16b(sK+e, Kp0+row*kvstride+col);
            else *reinterpret_cast<uint4*>(sK+e)=make_uint4(0,0,0,0);
        }
    }
    cp_async_commit();
    if (valid_cur) {
        for (int e = tid*8; e < SUB_N*HEAD_DIM; e += nthr*8) {
            int row=e/HEAD_DIM, col=e%HEAD_DIM;
            if (row<kvv_cur) cp_async_16b(sV+e, Vp0+row*kvstride+col);
            else *reinterpret_cast<uint4*>(sV+e)=make_uint4(0,0,0,0);
        }
    }
    cp_async_commit();

    const __nv_bfloat16 * __restrict__ sKc_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sK, 16));
    const __nv_bfloat16 * __restrict__ sVc_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sV, 16));

    for (int s = s_begin; s < s_end; s++) {
        cp_async_wait(1);
        __syncthreads();
        const int kvv = kvv_cur;

        // ---- QK: this warp's 16 keys [key0,key0+16) -> 2 n-tiles ----
        float Sr[2][4];
        #pragma unroll
        for (int i = 0; i < 2; i++) Sr[i][0]=Sr[i][1]=Sr[i][2]=Sr[i][3]=0.f;
        const int kgrp0 = (key0 + grp) * HEAD_DIM;
        uint32_t b0n = lds_u32(&sKc_a[kgrp0 + sub2]);
        uint32_t b1n = lds_u32(&sKc_a[kgrp0 + sub2 + 8]);
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int qoff = ks * MMA_K_BF16;
            uint32_t a0 = lds_u32(&sQ_a[qbase0 + qoff + sub2]);
            uint32_t a1 = lds_u32(&sQ_a[qbase1 + qoff + sub2]);
            uint32_t a2 = lds_u32(&sQ_a[qbase0 + qoff + sub2 + 8]);
            uint32_t a3 = lds_u32(&sQ_a[qbase1 + qoff + sub2 + 8]);
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                hmma_bf16_m16n8k16(Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3],
                    a0,a1,a2,a3,b0n,b1n, Sr[nt][0],Sr[nt][1],Sr[nt][2],Sr[nt][3]);
                if (nt < 1) {
                    const int kr = (key0 + MMA_N + grp)*HEAD_DIM + qoff;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                } else if (ks < 7) {
                    const int kr = kgrp0 + (ks+1)*MMA_K_BF16;
                    b0n = lds_u32(&sKc_a[kr + sub2]); b1n = lds_u32(&sKc_a[kr + sub2 + 8]);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            Sr[i][0]*=softmax_scale; Sr[i][1]*=softmax_scale;
            Sr[i][2]*=softmax_scale; Sr[i][3]*=softmax_scale;
        }
        if (kvv < SUB_N) {
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                int n0 = key0 + nt*MMA_N + sub*2, n1 = n0+1;
                if (n0 >= kvv) { Sr[nt][0]=-INFINITY; Sr[nt][2]=-INFINITY; }
                if (n1 >= kvv) { Sr[nt][1]=-INFINITY; Sr[nt][3]=-INFINITY; }
            }
        }
        float mx0=-INFINITY, mx1=-INFINITY;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            mx0=fmaxf(mx0,fmaxf(Sr[i][0],Sr[i][1]));
            mx1=fmaxf(mx1,fmaxf(Sr[i][2],Sr[i][3]));
        }
        { float tt;
          tt=__shfl_xor_sync(0xffffffff,mx0,1); mx0=fmaxf(mx0,tt);
          tt=__shfl_xor_sync(0xffffffff,mx0,2); mx0=fmaxf(mx0,tt);
          tt=__shfl_xor_sync(0xffffffff,mx1,1); mx1=fmaxf(mx1,tt);
          tt=__shfl_xor_sync(0xffffffff,mx1,2); mx1=fmaxf(mx1,tt); }
        if (sub == 0) { sRed[(wid*GQA + grp)*2 + 0] = mx0; sRed[(wid*GQA + grp + 8)*2 + 0] = mx1; }
        __syncthreads();
        float pmx0 = mx0, pmx1 = mx1;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            pmx0 = fmaxf(pmx0, sRed[(w*GQA + grp)*2 + 0]);
            pmx1 = fmaxf(pmx1, sRed[(w*GQA + grp + 8)*2 + 0]);
        }
        float mn0=fmaxf(rm[0],pmx0), mn1=fmaxf(rm[1],pmx1);
        float s0=(rm[0]==-INFINITY)?0.f:exp2f(LG2E*(rm[0]-mn0));
        float s1=(rm[1]==-INFINITY)?0.f:exp2f(LG2E*(rm[1]-mn1));
        rl[0]*=s0; rl[1]*=s1;
        #pragma unroll
        for (int d=0;d<4;d++){ Oa[d][0]*=s0;Oa[d][1]*=s0;Oa[d][2]*=s1;Oa[d][3]*=s1; }
        const bool dead0=(mn0==-INFINITY), dead1=(mn1==-INFINITY);
        float ls0=0.f, ls1=0.f;
        #pragma unroll
        for (int i=0;i<2;i++){
            Sr[i][0]=dead0?0.f:exp2f(LG2E*(Sr[i][0]-mn0));
            Sr[i][1]=dead0?0.f:exp2f(LG2E*(Sr[i][1]-mn0));
            Sr[i][2]=dead1?0.f:exp2f(LG2E*(Sr[i][2]-mn1));
            Sr[i][3]=dead1?0.f:exp2f(LG2E*(Sr[i][3]-mn1));
            ls0+=Sr[i][0]+Sr[i][1]; ls1+=Sr[i][2]+Sr[i][3];
        }
        { float tt;
          tt=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls0,2); ls0+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls1,2); ls1+=tt; }
        if (sub == 0) { sRed[(wid*GQA + grp)*2 + 1] = ls0; sRed[(wid*GQA + grp + 8)*2 + 1] = ls1; }
        if (pv_bf16) {
            #pragma unroll
            for (int nt=0;nt<2;nt++){
                int kv_col = key0 + nt*MMA_N + sub*2;
                sPbf[grp*SUB_N + kv_col]       = __float2bfloat16(Sr[nt][0]);
                sPbf[grp*SUB_N + kv_col + 1]   = __float2bfloat16(Sr[nt][1]);
                sPbf[(grp+8)*SUB_N + kv_col]     = __float2bfloat16(Sr[nt][2]);
                sPbf[(grp+8)*SUB_N + kv_col + 1] = __float2bfloat16(Sr[nt][3]);
            }
        } else {
            #pragma unroll
            for (int nt=0;nt<2;nt++){
                int kv_col = key0 + nt*MMA_N + sub*2;
                *reinterpret_cast<uint16_t*>(&sP[grp*SUB_N + kv_col])     = cvt_2f32_to_e4m3x2(Sr[nt][0],Sr[nt][1]);
                *reinterpret_cast<uint16_t*>(&sP[(grp+8)*SUB_N + kv_col]) = cvt_2f32_to_e4m3x2(Sr[nt][2],Sr[nt][3]);
            }
        }
        __syncthreads();
        float gl0=0.f, gl1=0.f;
        #pragma unroll
        for (int w=0; w<NUM_WARPS; w++){
            gl0 += sRed[(w*GQA + grp)*2 + 1];
            gl1 += sRed[(w*GQA + grp + 8)*2 + 1];
        }
        rm[0]=mn0; rm[1]=mn1; rl[0]+=gl0; rl[1]+=gl1;

        cp_async_wait0();
        __syncthreads();
        // ---- PV : this warp owns output dims [odim0 + wid*(dims_per_blk/4) ..) ----
        const int dbase = odim0 + wid * (dims_per_blk / 4);
        if (pv_bf16) {
          // Native bf16 HMMA PV: A=P[16h x 64k] bf16, B=V[64k x dim] bf16. No
          // fp8 conversion on the critical path. 64 keys = 4 k16-tiles; B is
          // V transposed (V smem is [key][dim]) gathered as bf16x2 pairs.
          const __nv_bfloat16 * __restrict__ sPbf_a =
              static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sPbf, 16));
          #pragma unroll
          for (int kt=0; kt<4; kt++){
            const int koff = kt*MMA_K_BF16;            // first key of this k16-tile
            uint32_t pa0 = lds_u32(&sPbf_a[grp*SUB_N     + koff + sub*2]);
            uint32_t pa1 = lds_u32(&sPbf_a[(grp+8)*SUB_N + koff + sub*2]);
            uint32_t pa2 = lds_u32(&sPbf_a[grp*SUB_N     + koff + sub*2 + 8]);
            uint32_t pa3 = lds_u32(&sPbf_a[(grp+8)*SUB_N + koff + sub*2 + 8]);
            const int kr_lo = koff + sub*2;            // V rows for b0 (2 keys)
            const int kr_hi = koff + sub*2 + 8;        // V rows for b1 (2 keys)
            int vcol = dbase + grp;
            #define VLDB(kr,vc) bf16x2_pack(sVc_a[(kr)*HEAD_DIM+(vc)], \
                                            sVc_a[((kr)+1)*HEAD_DIM+(vc)])
            uint32_t vb0n = VLDB(kr_lo, vcol);
            uint32_t vb1n = VLDB(kr_hi, vcol);
            #pragma unroll
            for (int dt=0; dt<4; dt++){
                if (dt >= dt_cnt) break;
                uint32_t vb0=vb0n, vb1=vb1n;
                hmma_bf16_m16n8k16(Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3],
                    pa0,pa1,pa2,pa3,vb0,vb1, Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3]);
                if (dt+1<dt_cnt){
                    vcol=dbase+(dt+1)*MMA_N+grp;
                    vb0n = VLDB(kr_lo, vcol); vb1n = VLDB(kr_hi, vcol);
                }
            }
            #undef VLDB
          }
        } else {
        // dt_cnt = 4 (dh_split=1) or 2 (dh_split=2); 64 keys = 2 ns-groups.
        #pragma unroll
        for (int ns=0; ns<2; ns++){
            const int pa_base0 = grp*SUB_N + ns*MMA_K_FP8;
            const int pa_base1 = (grp+8)*SUB_N + ns*MMA_K_FP8;
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4]);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4]);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub*4 + 16]);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub*4 + 16]);
            const int vr_lo = ns*MMA_K_FP8 + sub*4;
            const int vr_hi = vr_lo + 16;
            int vcol = dbase + grp;
            #define VLDS(rb,vc) f32x4_to_e4m3x4( \
                __bfloat162float(sVc_a[(rb)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+1)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+2)*HEAD_DIM+(vc)]), \
                __bfloat162float(sVc_a[((rb)+3)*HEAD_DIM+(vc)]))
            uint32_t vb0n = VLDS(vr_lo, vcol);
            uint32_t vb1n = VLDS(vr_hi, vcol);
            #pragma unroll
            for (int dt=0; dt<4; dt++){
                if (dt >= dt_cnt) break;
                uint32_t vb0=vb0n, vb1=vb1n;
                qmma_sf_fp8_m16n8k32(Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3],
                    pa0,pa1,pa2,pa3,vb0,vb1, Oa[dt][0],Oa[dt][1],Oa[dt][2],Oa[dt][3]);
                if (dt+1<dt_cnt){
                    vcol=dbase+(dt+1)*MMA_N+grp;
                    vb0n = VLDS(vr_lo, vcol); vb1n = VLDS(vr_hi, vcol);
                }
            }
            #undef VLDS
        }
        }
        __syncthreads();
        if (s + 1 < s_end) {
            int kv2 = 0; const __nv_bfloat16 *Kp2=nullptr,*Vp2=nullptr;
            bool v2 = sub_meta(s+1, kv2, Kp2, Vp2);
            if (v2) for (int e=tid*8;e<SUB_N*HEAD_DIM;e+=nthr*8){int row=e/HEAD_DIM,col=e%HEAD_DIM;
                if(row<kv2) cp_async_16b(sK+e,Kp2+row*kvstride+col); else *reinterpret_cast<uint4*>(sK+e)=make_uint4(0,0,0,0);}
            cp_async_commit();
            if (v2) for (int e=tid*8;e<SUB_N*HEAD_DIM;e+=nthr*8){int row=e/HEAD_DIM,col=e%HEAD_DIM;
                if(row<kv2) cp_async_16b(sV+e,Vp2+row*kvstride+col); else *reinterpret_cast<uint4*>(sV+e)=make_uint4(0,0,0,0);}
            cp_async_commit();
            kvv_cur = kv2; valid_cur = v2;
        }
    }

    const int64_t head_base = ((int64_t)req * num_heads_kv + hkv) * GQA;
    const int r0 = grp, r1 = grp + 8;
    const int dbase = odim0 + wid * (dims_per_blk / 4);
    #pragma unroll
    for (int dt=0; dt<4; dt++){
        if (dt >= dt_cnt) break;
        int c0 = dbase + dt*MMA_N + sub*2, c1 = c0+1;
        int64_t ob0 = ((head_base + r0) * split_chunks + chunk) * HEAD_DIM;
        int64_t ob1 = ((head_base + r1) * split_chunks + chunk) * HEAD_DIM;
        O_part[ob0 + c0]=__float2bfloat16(Oa[dt][0]); O_part[ob0 + c1]=__float2bfloat16(Oa[dt][1]);
        O_part[ob1 + c0]=__float2bfloat16(Oa[dt][2]); O_part[ob1 + c1]=__float2bfloat16(Oa[dt][3]);
    }
    // M/L are per-(head) scalars, identical regardless of dim-half. Only dh_id==0
    // writes them (dh_id==1 would write the same values; restrict to avoid races
    // being a concern and to skip redundant work).
    if (dh_id == 0 && wid == 0 && sub == 0) {
        M_part[(head_base + r0) * split_chunks + chunk] = rm[0];
        M_part[(head_base + r1) * split_chunks + chunk] = rm[1];
        L_part[(head_base + r0) * split_chunks + chunk] = rl[0];
        L_part[(head_base + r1) * split_chunks + chunk] = rl[1];
    }

    // ===== FUSED MERGE (single launch): the LAST chunk-block to finish this
    // (req,kvh) reads all chunk partials (L2-hot, no 2nd launch) and writes the
    // final O. dh_split must be 1 (each block owns all 128 dims). =====
    if (O_final != nullptr) {
        __threadfence();
        __syncthreads();
        __shared__ bool s_last;
        if (tid == 0) {
            int prev = atomicAdd(&chunk_done[req * num_heads_kv + hkv], 1);
            s_last = (prev == split_chunks - 1);
        }
        __syncthreads();
        if (!s_last) return;

        // This block now combines all `split_chunks` partials for (req,hkv)
        // over the 16 GQA heads x 128 dims. Precompute per-(head,chunk) rescale
        // weights ONCE in smem (16*C exp2f total), then the dim loop is pure FMA.
        const float * __restrict__ Mb = M_part + head_base * split_chunks;
        const float * __restrict__ Lb = L_part + head_base * split_chunks;
        const __nv_bfloat16 * __restrict__ Ob = O_part + head_base * split_chunks * HEAD_DIM;
        __shared__ float s_inv[GQA];
        // reuse sRed scratch (>= 16*C floats? sized NUM_WARPS*GQA*2=128; C up to 32
        // => 16*32=512 > 128). Use a dedicated smem buffer via dynamic extension:
        // store weights in sV region (free now: 64*128 bf16 = 16KB = 4096 f32).
        float *s_w = reinterpret_cast<float*>(sV);   // [GQA * split_chunks]
        constexpr float LG2E_M = 1.4426950408889634f;
        for (int h = tid; h < GQA; h += nthr) {
            const float *Mh = Mb + (int64_t)h * split_chunks;
            const float *Lh = Lb + (int64_t)h * split_chunks;
            float gmax = -INFINITY;
            for (int c = 0; c < split_chunks; c++) gmax = fmaxf(gmax, Mh[c]);
            float denom = 0.f;
            for (int c = 0; c < split_chunks; c++) {
                float m = Mh[c];
                float w = (m != -INFINITY) ? exp2f(LG2E_M * (m - gmax)) : 0.f;
                s_w[h * split_chunks + c] = w;
                denom += Lh[c] * w;
            }
            s_inv[h] = (denom > 0.f) ? (1.f / denom) : 0.f;
        }
        __syncthreads();
        // each thread owns a strided set of (head,dim) outputs; weights are smem.
        for (int idx = tid; idx < GQA * HEAD_DIM; idx += nthr) {
            int h = idx / HEAD_DIM, dcol = idx % HEAD_DIM;
            const __nv_bfloat16 *Oh = Ob + (int64_t)h * split_chunks * HEAD_DIM + dcol;
            const float *wh = s_w + h * split_chunks;
            float a0=0.f, a1=0.f, a2=0.f, a3=0.f;
            int c = 0;
            for (; c + 4 <= split_chunks; c += 4) {
                a0 += __bfloat162float(Oh[(int64_t)(c+0)*HEAD_DIM]) * wh[c+0];
                a1 += __bfloat162float(Oh[(int64_t)(c+1)*HEAD_DIM]) * wh[c+1];
                a2 += __bfloat162float(Oh[(int64_t)(c+2)*HEAD_DIM]) * wh[c+2];
                a3 += __bfloat162float(Oh[(int64_t)(c+3)*HEAD_DIM]) * wh[c+3];
            }
            for (; c < split_chunks; c++)
                a0 += __bfloat162float(Oh[(int64_t)c*HEAD_DIM]) * wh[c];
            float acc = (a0+a1) + (a2+a3);
            int hq = hkv * GQA + h;
            O_final[((int64_t)req * num_heads_q + hq) * HEAD_DIM + dcol] =
                __float2bfloat16(acc * s_inv[h]);
        }
    }
}

// ===========================================================================
// PAGE-128 4-WARP LDMATRIX PARTIAL KERNEL  (the load-path rewrite)
//
//   The smem->tensor-core FEED is hardware (ldmatrix / LDSM), not the software
//   gather of the other variants. This is the named #1 limiter fix:
//     QK : A=Q, B=K both fed by `ldmatrix.x4` (no transpose) from the
//          row-major [row][head_dim] smem the cp.async lands.
//     PV : NATIVE bf16 HMMA (m16n8k16). A=P[16head x key] bf16 fed by
//          `ldmatrix.x4`. B=V fed by `ldmatrix.x4.trans` — the transpose that
//          was the 362-LDS column-wise V gather is now ONE hardware op.
//
//   Geometry = Triton's: one block owns one kv-head's full 128-key page (all 16
//   GQA heads). split_chunks over the topk selected 128-pages (default 16 ->
//   1 page/chunk -> 16-chunk merge). Grid = (Hkv, split_chunks, R) = 64 blocks
//   at bs1 (chunks=16, Hkv=4).
//
//   4 warps:
//     QK  split by KEY  : warp w computes keys [32w,32w+32) -> 4 n-tiles. Q is
//                         shared (ldmatrix.x4 per k16-step); K's 32 keys = 2
//                         ldmatrix.x4 (each covers 16 keys = 2 n-tiles, frags
//                         key0-7=(v0,v2) key8-15=(v1,v3)).
//     softmax           : per-warp partial max/sum reduced ACROSS warps via a
//                         tiny smem scratch (identical to the _4w variant).
//     PV  split by DIM  : warp w computes output dims [32w,32w+32) over ALL 128
//                         keys. P[16h x key] bf16 fed by ldmatrix.x4; V fed by
//                         ldmatrix.x4.trans (16 dims x 16 keys/op; dims0-7=
//                         (v0,v1) dims8-15=(v2,v3)). Disjoint dims -> no merge.
//
//   K/V cp.async load path is UNCHANGED (128-bit LDGSTS, [key][dim] row-major).
//   Single-buffered V (2x128x128 bf16 = 64KB won't double-buffer under 99KB
//   optin; the 8-step QK+PV compute hides the next-page load). Same numerics as
//   PV_BF16=1: bf16 QK, bf16 PV, fp32 accum, bf16 partial-out (rms ~1.3e-3).
// ===========================================================================
extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
sm120_fmha_decode_partial_p128_ldsm(
    const __nv_bfloat16 * __restrict__ Q,     // [R, Hq, 128]
    const __nv_bfloat16 * __restrict__ Kc,    // [num_pages, 128, Hkv, 128]
    const __nv_bfloat16 * __restrict__ Vc,    // [num_pages, 128, Hkv, 128]
    __nv_bfloat16 * __restrict__ O_part,      // [R, Hkv, GQA, C, 128] bf16
    float * __restrict__ M_part,              // [R, Hkv, GQA, C]
    float * __restrict__ L_part,              // [R, Hkv, GQA, C]
    const int * __restrict__ block_ids,       // [R, topk] logical 128-pages (-1 pad)
    const int * __restrict__ block_table,     // [R, max_logical_blocks]
    int max_logical_blocks,
    int topk,
    int seq_len_k,
    int num_heads_q,
    int num_heads_kv,
    float softmax_scale,
    int split_chunks,
    int pages_per_chunk)
{
    extern __shared__ __nv_bfloat16 smem_raw[];

    const int tid = threadIdx.x;
    const int wid = tid / 32;        // 0..3 : warp owns keys/dims [32w,32w+32)
    const int lid = tid % 32;
    const int grp = lid / 4;         // 0..7  -> output row pair {grp, grp+8}
    const int sub = lid % 4;         // 0..3  -> output col pair
    const int nthr = NUM_WARPS * 32;
    const int ldlane = lid % 16;     // ldmatrix address lane within the 16x16 tile
    const int ldhalf = lid / 16;     // 0/1 -> which 8-col block of the tile

    const int hkv   = blockIdx.x;
    const int chunk = blockIdx.y;
    const int req   = blockIdx.z;

    const int qstride     = num_heads_q  * HEAD_DIM;
    const int kvstride    = num_heads_kv * HEAD_DIM;
    const int page_stride = P128_N * num_heads_kv * HEAD_DIM;

    const int hq0 = hkv * GQA;
    const __nv_bfloat16 *Qp    = Q + (int64_t)req * qstride + hq0 * HEAD_DIM;
    const __nv_bfloat16 *Kbase = Kc + hkv * HEAD_DIM;
    const __nv_bfloat16 *Vbase = Vc + hkv * HEAD_DIM;
    const int *bt_row  = block_table + req * max_logical_blocks;
    const int *blk_row = block_ids   + req * topk;

    // SMEM (PADDED for ldmatrix, stride +8 bf16 per row):
    //   sQ[16 x LDSM_KVLD] + sK[128 x LDSM_KVLD] + sV[128 x LDSM_KVLD]
    //   + sPbf[16 x LDSM_PLD] + sRed.
    __nv_bfloat16 *sQ  = smem_raw;
    __nv_bfloat16 *sK  = sQ + GQA   * LDSM_KVLD;
    __nv_bfloat16 *sV  = sK + P128_N* LDSM_KVLD;
    __nv_bfloat16 *sPbf= sV + P128_N* LDSM_KVLD;            // 16 x LDSM_PLD bf16
    float *sRed        = reinterpret_cast<float*>(sPbf + GQA * LDSM_PLD);

    {   // Q load into padded [head][hd] layout.
        const int total = GQA * HEAD_DIM;
        for (int e = tid * 8; e < total; e += nthr * 8) {
            int row = e / HEAD_DIM, col = e % HEAD_DIM;
            cp_async_16b(sQ + row * LDSM_KVLD + col, Qp + e);
        }
        cp_async_commit();
    }

    const int key0 = wid * 32;             // first key of this warp (QK)
    const int dbase= wid * 32;             // first output dim of this warp (PV)

    float Oa[4][4];                        // 4 dt-tiles (8 dims each) x {r0c0,r0c1,r1c0,r1c1}
    #pragma unroll
    for (int i = 0; i < 4; i++) Oa[i][0]=Oa[i][1]=Oa[i][2]=Oa[i][3]=0.f;
    float rm[2] = {-INFINITY, -INFINITY};
    float rl[2] = {0.f, 0.f};
    constexpr float LG2E = 1.4426950408889634f;

    const int p_begin = chunk * pages_per_chunk;
    const int p_end   = min(p_begin + pages_per_chunk, topk);

    auto page_meta = [&](int t, int &kvv, const __nv_bfloat16* &Kpage,
                         const __nv_bfloat16* &Vpage) -> bool {
        int kb = (t >= p_begin && t < p_end) ? blk_row[t] : -1;
        if (kb < 0) return false;
        int kvs = kb * P128_N;
        if (kvs >= seq_len_k) return false;
        kvv = min(P128_N, seq_len_k - kvs);
        int page = bt_row[kb];
        Kpage = Kbase + (int64_t)page * page_stride;
        Vpage = Vbase + (int64_t)page * page_stride;
        return true;
    };

    // K and V committed as SEPARATE cp.async groups (QK can start when K lands).
    int kvv_cur = 0; const __nv_bfloat16 *Kp0=nullptr,*Vp0=nullptr;
    bool valid_cur = page_meta(p_begin, kvv_cur, Kp0, Vp0);
    if (valid_cur) load_page128_async_pad(sK, Kp0, kvv_cur, kvstride, tid, nthr);
    cp_async_commit();                                   // group: K
    if (valid_cur) load_page128_async_pad(sV, Vp0, kvv_cur, kvstride, tid, nthr);
    cp_async_commit();                                   // group: V

    for (int t = p_begin; t < p_end; t++) {
        cp_async_wait(1);          // K landed; V still in flight
        __syncthreads();
        const int kvv = kvv_cur;

        // ---- QK: this warp's 32 keys [key0,key0+32) -> 4 n-tiles ----
        // Q ldmatrix addr: row = head (ldlane), col-block within the k16-step.
        // K ldmatrix addr: row = key (key0 + ldlane), col-block within k16-step.
        // Two K tiles per step cover keys [key0..+16) and [key0+16..+32).
        float Sr[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) Sr[i][0]=Sr[i][1]=Sr[i][2]=Sr[i][3]=0.f;
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int koff = ks * MMA_K_BF16;             // head-dim offset of this k16-step
            uint32_t qa0,qa1,qa2,qa3;
            ldmatrix_x4(qa0,qa1,qa2,qa3,
                cvta_shared(&sQ[ldlane*LDSM_KVLD + koff + ldhalf*8]));
            uint32_t ka0,ka1,ka2,ka3, kb0,kb1,kb2,kb3;
            ldmatrix_x4(ka0,ka1,ka2,ka3,
                cvta_shared(&sK[(key0 + ldlane)*LDSM_KVLD + koff + ldhalf*8]));
            ldmatrix_x4(kb0,kb1,kb2,kb3,
                cvta_shared(&sK[(key0 + 16 + ldlane)*LDSM_KVLD + koff + ldhalf*8]));
            // n-tile0=keys[key0..7]=(ka0,ka2); n1=keys[key0+8..15]=(ka1,ka3);
            // n2=keys[key0+16..23]=(kb0,kb2); n3=keys[key0+24..31]=(kb1,kb3).
            hmma_bf16_m16n8k16(Sr[0][0],Sr[0][1],Sr[0][2],Sr[0][3], qa0,qa1,qa2,qa3, ka0,ka2, Sr[0][0],Sr[0][1],Sr[0][2],Sr[0][3]);
            hmma_bf16_m16n8k16(Sr[1][0],Sr[1][1],Sr[1][2],Sr[1][3], qa0,qa1,qa2,qa3, ka1,ka3, Sr[1][0],Sr[1][1],Sr[1][2],Sr[1][3]);
            hmma_bf16_m16n8k16(Sr[2][0],Sr[2][1],Sr[2][2],Sr[2][3], qa0,qa1,qa2,qa3, kb0,kb2, Sr[2][0],Sr[2][1],Sr[2][2],Sr[2][3]);
            hmma_bf16_m16n8k16(Sr[3][0],Sr[3][1],Sr[3][2],Sr[3][3], qa0,qa1,qa2,qa3, kb1,kb3, Sr[3][0],Sr[3][1],Sr[3][2],Sr[3][3]);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            Sr[i][0]*=softmax_scale; Sr[i][1]*=softmax_scale;
            Sr[i][2]*=softmax_scale; Sr[i][3]*=softmax_scale;
        }
        if (kvv < P128_N) {
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                int n0 = key0 + nt*MMA_N + sub*2, n1 = n0+1;
                if (n0 >= kvv) { Sr[nt][0]=-INFINITY; Sr[nt][2]=-INFINITY; }
                if (n1 >= kvv) { Sr[nt][1]=-INFINITY; Sr[nt][3]=-INFINITY; }
            }
        }
        // ---- per-warp partial row-max over its 32 keys, then cross-warp ----
        float mx0=-INFINITY, mx1=-INFINITY;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            mx0=fmaxf(mx0,fmaxf(Sr[i][0],Sr[i][1]));
            mx1=fmaxf(mx1,fmaxf(Sr[i][2],Sr[i][3]));
        }
        { float tt;
          tt=__shfl_xor_sync(0xffffffff,mx0,1); mx0=fmaxf(mx0,tt);
          tt=__shfl_xor_sync(0xffffffff,mx0,2); mx0=fmaxf(mx0,tt);
          tt=__shfl_xor_sync(0xffffffff,mx1,1); mx1=fmaxf(mx1,tt);
          tt=__shfl_xor_sync(0xffffffff,mx1,2); mx1=fmaxf(mx1,tt); }
        if (sub == 0) { sRed[(wid*GQA + grp)*2 + 0] = mx0; sRed[(wid*GQA + grp + 8)*2 + 0] = mx1; }
        __syncthreads();
        float pmx0 = mx0, pmx1 = mx1;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            pmx0 = fmaxf(pmx0, sRed[(w*GQA + grp)*2 + 0]);
            pmx1 = fmaxf(pmx1, sRed[(w*GQA + grp + 8)*2 + 0]);
        }
        float mn0=fmaxf(rm[0],pmx0), mn1=fmaxf(rm[1],pmx1);
        float s0=(rm[0]==-INFINITY)?0.f:exp2f(LG2E*(rm[0]-mn0));
        float s1=(rm[1]==-INFINITY)?0.f:exp2f(LG2E*(rm[1]-mn1));
        rl[0]*=s0; rl[1]*=s1;
        #pragma unroll
        for (int d=0;d<4;d++){ Oa[d][0]*=s0;Oa[d][1]*=s0;Oa[d][2]*=s1;Oa[d][3]*=s1; }
        const bool dead0=(mn0==-INFINITY), dead1=(mn1==-INFINITY);
        float ls0=0.f, ls1=0.f;
        #pragma unroll
        for (int i=0;i<4;i++){
            Sr[i][0]=dead0?0.f:exp2f(LG2E*(Sr[i][0]-mn0));
            Sr[i][1]=dead0?0.f:exp2f(LG2E*(Sr[i][1]-mn0));
            Sr[i][2]=dead1?0.f:exp2f(LG2E*(Sr[i][2]-mn1));
            Sr[i][3]=dead1?0.f:exp2f(LG2E*(Sr[i][3]-mn1));
            ls0+=Sr[i][0]+Sr[i][1]; ls1+=Sr[i][2]+Sr[i][3];
        }
        { float tt;
          tt=__shfl_xor_sync(0xffffffff,ls0,1); ls0+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls0,2); ls0+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls1,1); ls1+=tt;
          tt=__shfl_xor_sync(0xffffffff,ls1,2); ls1+=tt; }
        if (sub == 0) { sRed[(wid*GQA + grp)*2 + 1] = ls0; sRed[(wid*GQA + grp + 8)*2 + 1] = ls1; }
        // ---- write this warp's 32 P-bf16 into shared sPbf[16 x 128] ----
        //   P[head][key], key = key0 + nt*8 + {sub*2, sub*2+1}.
        #pragma unroll
        for (int nt=0;nt<4;nt++){
            int kv_col = key0 + nt*MMA_N + sub*2;
            sPbf[grp*LDSM_PLD + kv_col]         = __float2bfloat16(Sr[nt][0]);
            sPbf[grp*LDSM_PLD + kv_col + 1]     = __float2bfloat16(Sr[nt][1]);
            sPbf[(grp+8)*LDSM_PLD + kv_col]     = __float2bfloat16(Sr[nt][2]);
            sPbf[(grp+8)*LDSM_PLD + kv_col + 1] = __float2bfloat16(Sr[nt][3]);
        }
        __syncthreads();
        float gl0=0.f, gl1=0.f;
        #pragma unroll
        for (int w=0; w<NUM_WARPS; w++){
            gl0 += sRed[(w*GQA + grp)*2 + 1];
            gl1 += sRed[(w*GQA + grp + 8)*2 + 1];
        }
        rm[0]=mn0; rm[1]=mn1; rl[0]+=gl0; rl[1]+=gl1;

        // V must have landed (it streamed during QK+softmax).
        cp_async_wait0();
        __syncthreads();

        // ---- PV (bf16 HMMA): O[16h x 32dim] = P[16h x 128k] @ V[128k x 32dim] ----
        //   8 k16-steps over the 128 keys. Per step:
        //     A = P[:, koff..koff+16] ldmatrix.x4 (row=head, k-block=ldhalf).
        //     B = V[koff..koff+16][dbase..dbase+32] via 2 ldmatrix.x4.trans
        //         (each covers 16 dims x 16 keys: dims0-7=(v0,v1) dims8-15=(v2,v3)).
        //   dt-tiles: dim n-tile dt (8 dims) -> Oa[dt]. 4 dt = 32 dims.
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int koff = ks * MMA_K_BF16;             // key offset of this k16-step
            uint32_t pa0,pa1,pa2,pa3;
            ldmatrix_x4(pa0,pa1,pa2,pa3,
                cvta_shared(&sPbf[ldlane*LDSM_PLD + koff + ldhalf*8]));
            // V trans tile A: keys [koff..+16] x dims [dbase..dbase+16]
            uint32_t va0,va1,va2,va3;
            ldmatrix_x4_trans(va0,va1,va2,va3,
                cvta_shared(&sV[(koff + ldlane)*LDSM_KVLD + dbase + ldhalf*8]));
            // V trans tile B: keys [koff..+16] x dims [dbase+16..dbase+32]
            uint32_t vb0,vb1,vb2,vb3;
            ldmatrix_x4_trans(vb0,vb1,vb2,vb3,
                cvta_shared(&sV[(koff + ldlane)*LDSM_KVLD + dbase + 16 + ldhalf*8]));
            // dims0-7=(va0,va1) dims8-15=(va2,va3) dims16-23=(vb0,vb1) dims24-31=(vb2,vb3)
            hmma_bf16_m16n8k16(Oa[0][0],Oa[0][1],Oa[0][2],Oa[0][3], pa0,pa1,pa2,pa3, va0,va1, Oa[0][0],Oa[0][1],Oa[0][2],Oa[0][3]);
            hmma_bf16_m16n8k16(Oa[1][0],Oa[1][1],Oa[1][2],Oa[1][3], pa0,pa1,pa2,pa3, va2,va3, Oa[1][0],Oa[1][1],Oa[1][2],Oa[1][3]);
            hmma_bf16_m16n8k16(Oa[2][0],Oa[2][1],Oa[2][2],Oa[2][3], pa0,pa1,pa2,pa3, vb0,vb1, Oa[2][0],Oa[2][1],Oa[2][2],Oa[2][3]);
            hmma_bf16_m16n8k16(Oa[3][0],Oa[3][1],Oa[3][2],Oa[3][3], pa0,pa1,pa2,pa3, vb2,vb3, Oa[3][0],Oa[3][1],Oa[3][2],Oa[3][3]);
        }
        __syncthreads();   // protect sPbf/sRed reuse + sK/sV reload

        if (t + 1 < p_end) {
            int kv2 = 0; const __nv_bfloat16 *Kp2=nullptr,*Vp2=nullptr;
            bool v2 = page_meta(t+1, kv2, Kp2, Vp2);
            if (v2) load_page128_async_pad(sK, Kp2, kv2, kvstride, tid, nthr);
            cp_async_commit();
            if (v2) load_page128_async_pad(sV, Vp2, kv2, kvstride, tid, nthr);
            cp_async_commit();
            kvv_cur = kv2; valid_cur = v2;
        }
    }

    // ===== epilogue: each warp writes its 32 output dims =====
    const int64_t head_base = ((int64_t)req * num_heads_kv + hkv) * GQA;
    const int r0 = grp, r1 = grp + 8;
    #pragma unroll
    for (int dt=0; dt<4; dt++){
        int c0 = dbase + dt*MMA_N + sub*2, c1 = c0+1;
        int64_t ob0 = ((head_base + r0) * split_chunks + chunk) * HEAD_DIM;
        int64_t ob1 = ((head_base + r1) * split_chunks + chunk) * HEAD_DIM;
        O_part[ob0 + c0]=__float2bfloat16(Oa[dt][0]); O_part[ob0 + c1]=__float2bfloat16(Oa[dt][1]);
        O_part[ob1 + c0]=__float2bfloat16(Oa[dt][2]); O_part[ob1 + c1]=__float2bfloat16(Oa[dt][3]);
    }
    if (wid == 0 && sub == 0) {
        M_part[(head_base + r0) * split_chunks + chunk] = rm[0];
        M_part[(head_base + r1) * split_chunks + chunk] = rm[1];
        L_part[(head_base + r0) * split_chunks + chunk] = rl[0];
        L_part[(head_base + r1) * split_chunks + chunk] = rl[1];
    }
}

// ============================ torch binding ============================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// q: [R, Hq, 128] bf16 (q_len==1 per request, R requests).
// k_cache,v_cache: [num_pages, 64, Hkv, 128] bf16.
// block_table: int32 [R, max_logical_blocks].
// block_ids:  int32 [R, topk] selected LOGICAL pages (-1 pad).
std::vector<torch::Tensor> forward_sparse_decode_bf16(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor block_ids,
    double softmax_scale, int64_t seq_len_k, int64_t split_chunks_in)
{
    TORCH_CHECK(q.is_cuda() && q.dtype()==torch::kBFloat16 && q.dim()==3 && q.size(2)==128,
                "q must be CUDA bf16 [R,Hq,128]");
    TORCH_CHECK(k_cache.dim()==4 && k_cache.size(1)==64 && k_cache.size(3)==128,
                "k_cache must be [num_pages,64,Hkv,128]");
    q=q.contiguous(); k_cache=k_cache.contiguous(); v_cache=v_cache.contiguous();
    block_table=block_table.contiguous(); block_ids=block_ids.contiguous();

    const int R = q.size(0);
    const int Hq = q.size(1);
    const int Hkv = k_cache.size(2);
    const int topk = block_ids.size(1);
    const int max_logical_blocks = block_table.size(1);
    const int seq_k = (int)seq_len_k;
    const int GQA_ = Hq / Hkv;
    TORCH_CHECK(GQA_ == GQA, "this decode kernel is specialized for GQA group == 16");

    // pick split_chunks: default ceil so each chunk ~ a few pages, fill SMs.
    int split_chunks = (int)split_chunks_in;
    if (split_chunks <= 0) split_chunks = topk;        // max parallelism: 1 page/chunk
    if (split_chunks > topk) split_chunks = topk;
    if (split_chunks < 1) split_chunks = 1;
    int pages_per_chunk = (topk + split_chunks - 1) / split_chunks;
    // recompute split_chunks so no empty trailing chunk wastes a block
    split_chunks = (topk + pages_per_chunk - 1) / pages_per_chunk;

    auto fopt = torch::dtype(torch::kFloat32).device(q.device());
    auto bopt = torch::dtype(torch::kBFloat16).device(q.device());
    // layout [R, Hkv, GQA, C] (O: +HEAD_DIM) for coalesced merge reads.
    auto O_part = torch::empty({R, Hkv, GQA, split_chunks, HEAD_DIM}, bopt);
    auto M_part = torch::empty({R, Hkv, GQA, split_chunks}, fopt);
    auto L_part = torch::empty({R, Hkv, GQA, split_chunks}, fopt);
    auto o   = torch::empty({R, Hq, HEAD_DIM}, bopt);
    auto lse = torch::empty({R, Hq}, fopt);

#ifndef NSTAGE
#define NSTAGE 2
#endif
    int smem_bytes = (GQA*HEAD_DIM + NSTAGE * 2 * BLK_N*HEAD_DIM) * 2  // Q + NSTAGEx(K+V)
                   + NUM_WARPS * GQA * 32;                            // sP_all
    cudaFuncSetAttribute(sm120_fmha_decode_partial_bf16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid_p(Hkv, split_chunks, R);
    dim3 block_p(NUM_WARPS*32);
    sm120_fmha_decode_partial_bf16<<<grid_p, block_p, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O_part.data_ptr()),
        M_part.data_ptr<float>(), L_part.data_ptr<float>(),
        block_ids.data_ptr<int>(), block_table.data_ptr<int>(),
        max_logical_blocks, topk, seq_k, Hq, Hkv, (float)softmax_scale,
        split_chunks, pages_per_chunk);

    const int HD_SPLIT = 2;                   // 64 dims/block (full 128B line), 2x blocks
    dim3 grid_m(R, Hq, HD_SPLIT);
    dim3 block_m(HEAD_DIM / HD_SPLIT);        // 64 threads
    int merge_smem = split_chunks * (int)sizeof(float);
    sm120_fmha_decode_merge_bf16<<<grid_m, block_m, merge_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(O_part.data_ptr()),
        M_part.data_ptr<float>(), L_part.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), lse.data_ptr<float>(),
        split_chunks, Hq, Hkv);

    return {o, lse};
}

// ----- PAGE-128 entrypoint (primary optimization, integration-ready) -----
// q: [R, Hq, 128] bf16. k_cache,v_cache: [num_pages, 128, Hkv, 128] bf16.
// block_table: int32 [R, max_logical_blocks]. block_ids: int32 [R, topk]
// selected LOGICAL 128-pages (-1 pad). page == sparse block == 128.
std::vector<torch::Tensor> forward_sparse_decode_p128_bf16(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor block_ids,
    double softmax_scale, int64_t seq_len_k, int64_t split_chunks_in,
    int64_t use_4warp)
{
    TORCH_CHECK(q.is_cuda() && q.dtype()==torch::kBFloat16 && q.dim()==3 && q.size(2)==128,
                "q must be CUDA bf16 [R,Hq,128]");
    TORCH_CHECK(k_cache.dim()==4 && k_cache.size(1)==128 && k_cache.size(3)==128,
                "k_cache must be [num_pages,128,Hkv,128]");
    q=q.contiguous(); k_cache=k_cache.contiguous(); v_cache=v_cache.contiguous();
    block_table=block_table.contiguous(); block_ids=block_ids.contiguous();

    const int R = q.size(0);
    const int Hq = q.size(1);
    const int Hkv = k_cache.size(2);
    const int topk = block_ids.size(1);
    const int max_logical_blocks = block_table.size(1);
    const int seq_k = (int)seq_len_k;
    const int GQA_ = Hq / Hkv;
    TORCH_CHECK(GQA_ == GQA, "this decode kernel is specialized for GQA group == 16");

    // use_4warp==2 -> 64-key sub-tile kernel (split-K granularity 64, up to
    // 2*topk chunks for 128 blocks). Otherwise split-K granularity is a full
    // 128-page (up to topk chunks).
    const bool sub64 = (use_4warp == 2);
    const int units = sub64 ? topk * 2 : topk;          // splittable units
    int split_chunks = (int)split_chunks_in;
    if (split_chunks <= 0) split_chunks = units;
    if (split_chunks > units) split_chunks = units;
    if (split_chunks < 1) split_chunks = 1;
    int units_per_chunk = (units + split_chunks - 1) / split_chunks;
    split_chunks = (units + units_per_chunk - 1) / units_per_chunk;
    int pages_per_chunk = units_per_chunk;              // (= sub_per_chunk when sub64)

    auto fopt = torch::dtype(torch::kFloat32).device(q.device());
    auto bopt = torch::dtype(torch::kBFloat16).device(q.device());
    auto O_part = torch::empty({R, Hkv, GQA, split_chunks, HEAD_DIM}, bopt);
    auto M_part = torch::empty({R, Hkv, GQA, split_chunks}, fopt);
    auto L_part = torch::empty({R, Hkv, GQA, split_chunks}, fopt);
    auto o   = torch::empty({R, Hq, HEAD_DIM}, bopt);
    auto lse = torch::empty({R, Hq}, fopt);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block_p(NUM_WARPS*32);
    if (sub64) {
        // dh_split=2 splits the 128 output dims across 2 blocks => doubles the
        // grid (4*chunks*2 blocks) so bs1 reaches ~256 blocks (1.36/SM) instead
        // of 128 (0.68/SM). The decode partial is occupancy-starved at bs1
        // (measured: partial/request drops 5.0->3.5->3.0us as blocks 128->256->
        // 512), so the redundant QK on otherwise-idle SMs is a net win.
        // dh_split=1 measured best: at bs1 there is a fixed amount of REAL work
        // (16 pages x 16 heads); splitting output dims doubles blocks but the
        // extra blocks redo the full QK + reload V, which costs more than the
        // idle-SM occupancy it buys (partial 5.0->6.0us at dh=2). Kept as a knob.
        const char *dhe = getenv("DH_SPLIT");
        int dh_split = dhe ? atoi(dhe) : 1;
        const char *pve = getenv("PV_BF16");
        int pv_bf16 = pve ? atoi(pve) : 0;   // 0: fp8 PV (fastest at bs1); 1: bf16 PV
        // FUSED single-launch: the last chunk-block of each (req,kvh) does the
        // LSE-merge in-kernel (L2-hot partials, no 2nd launch / DRAM round trip).
        // Requires dh_split==1 (each block owns all 128 output dims).
        // NOTE: fusion MEASURED SLOWER at every batch size (bs1: 8.0->15us). At
        // bs1 there are only Hkv*R (req,kvh) groups, so the "last block does the
        // merge" replaces the cheap 128-way-parallel merge kernel with a 4-way-
        // serial in-kernel merge. Default OFF; flag retained for the record.
        const char *fe = getenv("FUSED_MERGE");
        int fused = fe ? atoi(fe) : 0;
        if (dh_split != 1) fused = 0;
        dim3 grid_p(Hkv, split_chunks, R * dh_split);
        // sQ + sK[64x128] + sV[64x128] + sP[16x64] + sRed + sPbf[16x64 bf16]
        int smem_bytes = (GQA*HEAD_DIM + 2*SUB_N*HEAD_DIM) * 2
                       + GQA * SUB_N + NUM_WARPS * GQA * 2 * (int)sizeof(float)
                       + GQA * SUB_N * (int)sizeof(__nv_bfloat16);
        cudaFuncSetAttribute(sm120_fmha_decode_partial_p128_sub64,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        __nv_bfloat16 *o_final_ptr = nullptr;
        int *counter_ptr = nullptr;
        torch::Tensor counter;
        if (fused) {
            counter = torch::zeros({R * Hkv}, torch::dtype(torch::kInt32).device(q.device()));
            counter_ptr = counter.data_ptr<int>();
            o_final_ptr = reinterpret_cast<__nv_bfloat16*>(o.data_ptr());
        }
        sm120_fmha_decode_partial_p128_sub64<<<grid_p, block_p, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(O_part.data_ptr()),
            M_part.data_ptr<float>(), L_part.data_ptr<float>(),
            block_ids.data_ptr<int>(), block_table.data_ptr<int>(),
            max_logical_blocks, topk, seq_k, Hq, Hkv, (float)softmax_scale,
            split_chunks, units_per_chunk, dh_split, pv_bf16,
            o_final_ptr, counter_ptr);
        if (fused) return {o, lse};
    } else if (use_4warp == 3) {
        // LDMATRIX path: page-128, 4-warp, ldmatrix-fed QK + bf16 ldmatrix.trans PV.
        dim3 grid_p(Hkv, split_chunks, R);
        // PADDED smem (stride +8 bf16/row): sQ[16xKVLD] + sK[128xKVLD] +
        // sV[128xKVLD] + sPbf[16xPLD] + sRed(NUM_WARPS*16*2 f32).
        const int KVLD = HEAD_DIM + 8, PLD = P128_N + 8;
        int smem_bytes = (GQA*KVLD + 2*P128_N*KVLD + GQA*PLD) * (int)sizeof(__nv_bfloat16)
                       + NUM_WARPS * GQA * 2 * (int)sizeof(float);
        cudaFuncSetAttribute(sm120_fmha_decode_partial_p128_ldsm,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        sm120_fmha_decode_partial_p128_ldsm<<<grid_p, block_p, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(O_part.data_ptr()),
            M_part.data_ptr<float>(), L_part.data_ptr<float>(),
            block_ids.data_ptr<int>(), block_table.data_ptr<int>(),
            max_logical_blocks, topk, seq_k, Hq, Hkv, (float)softmax_scale,
            split_chunks, pages_per_chunk);
    } else if (use_4warp) {
        dim3 grid_p(Hkv, split_chunks, R);
        // sQ + sK + sV + sP(16x128) + sRed(NUM_WARPS*16*2 f32)
        int smem_bytes = (GQA*HEAD_DIM + 2*P128_N*HEAD_DIM) * 2
                       + GQA * P128_N + NUM_WARPS * GQA * 2 * (int)sizeof(float);
        cudaFuncSetAttribute(sm120_fmha_decode_partial_p128_4w,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        sm120_fmha_decode_partial_p128_4w<<<grid_p, block_p, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(O_part.data_ptr()),
            M_part.data_ptr<float>(), L_part.data_ptr<float>(),
            block_ids.data_ptr<int>(), block_table.data_ptr<int>(),
            max_logical_blocks, topk, seq_k, Hq, Hkv, (float)softmax_scale,
            split_chunks, pages_per_chunk);
    } else {
    dim3 grid_p(Hkv, split_chunks, R);
    int smem_bytes = (GQA*HEAD_DIM + NSTAGE128 * 2 * P128_N*HEAD_DIM) * 2  // Q + NSTAGE128x(K+V)
                   + GQA * P128_N;                                        // sP_all (16x128)
    cudaFuncSetAttribute(sm120_fmha_decode_partial_p128,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    sm120_fmha_decode_partial_p128<<<grid_p, block_p, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O_part.data_ptr()),
        M_part.data_ptr<float>(), L_part.data_ptr<float>(),
        block_ids.data_ptr<int>(), block_table.data_ptr<int>(),
        max_logical_blocks, topk, seq_k, Hq, Hkv, (float)softmax_scale,
        split_chunks, pages_per_chunk);
    }

    const char *mv = getenv("MERGE_V2");
    const int merge_v2 = mv ? atoi(mv) : 0;   // v2 multi-head merge: MEASURED SLOWER (4.2 vs 2.9), default OFF
    if (merge_v2) {
        const int HQPB = 4;                    // must match MERGE_HQPB
        dim3 grid_m(R, (Hq + HQPB - 1) / HQPB, 1);
        dim3 block_m(HQPB * (HEAD_DIM / 2));
        sm120_fmha_decode_merge_bf16_v2<<<grid_m, block_m, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(O_part.data_ptr()),
            M_part.data_ptr<float>(), L_part.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), lse.data_ptr<float>(),
            split_chunks, Hq, Hkv);
        return {o, lse};
    }
    const char *hs = getenv("MERGE_HD_SPLIT");
    const int HD_SPLIT = hs ? atoi(hs) : 2;
    dim3 grid_m(R, Hq, HD_SPLIT);
    dim3 block_m(HEAD_DIM / HD_SPLIT);
    int merge_smem = split_chunks * (int)sizeof(float);
    sm120_fmha_decode_merge_bf16<<<grid_m, block_m, merge_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(O_part.data_ptr()),
        M_part.data_ptr<float>(), L_part.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), lse.data_ptr<float>(),
        split_chunks, Hq, Hkv);

    return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_sparse_decode", &forward_sparse_decode_bf16,
          "SM120 block-sparse paged-KV flash-DECODING forward (BF16, page-64)",
          pybind11::arg("q"), pybind11::arg("k_cache"), pybind11::arg("v_cache"),
          pybind11::arg("block_table"), pybind11::arg("block_ids"),
          pybind11::arg("softmax_scale"), pybind11::arg("seq_len_k"),
          pybind11::arg("split_chunks") = 0);
    m.def("forward_sparse_decode_p128", &forward_sparse_decode_p128_bf16,
          "SM120 block-sparse paged-KV flash-DECODING forward (BF16, page-128)",
          pybind11::arg("q"), pybind11::arg("k_cache"), pybind11::arg("v_cache"),
          pybind11::arg("block_table"), pybind11::arg("block_ids"),
          pybind11::arg("softmax_scale"), pybind11::arg("seq_len_k"),
          pybind11::arg("split_chunks") = 0, pybind11::arg("use_4warp") = 0);
}
