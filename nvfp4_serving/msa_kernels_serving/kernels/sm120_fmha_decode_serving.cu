// SM120 block-sparse FUSED-CACHE DECODE -- SERVING variant (graph-capture safe).
// Derived from decode_kernel/sm120_fmha_decode.cu (validated W4=3 _ldsm partial
// + flat LSE merge). Serving-only changes vs source:
//   (1) block_ids [R,Hkv,topk] per-kv-head -> one launch covers GQA-shared topk.
//   (2) seq_lens DEVICE int32 [R] read in-kernel -> no host .item().
//   (3) KV cache is the M3 fused [num_blocks,2,128,Hkv,128]; K/V base pointers +
//       REAL strides (NHD or HND) are passed -> no cache copy, allocation-free.
// Numerics identical to the source W4=3 kernel. nvcc -gencode arch=compute_120f,code=sm_120f -O3
// SPDX-License-Identifier: MIT
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
// The cross-warp softmax reduction reads the NUM_WARPS per-warp partials as one
// float4 LDS.128. That vectorization assumes exactly 4 warps.
static_assert(NUM_WARPS == 4, "cross-warp float4 sRed reduction requires NUM_WARPS==4");

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
sm120_serve_decode_partial_p128_ldsm(
    const __nv_bfloat16 * __restrict__ Q,     // [R, Hq, 128]
    const __nv_bfloat16 * __restrict__ Kc,    // [num_pages, 128, Hkv, 128]
    const __nv_bfloat16 * __restrict__ Vc,    // [num_pages, 128, Hkv, 128]
    __nv_bfloat16 * __restrict__ O_part,      // [R, Hkv, GQA, C, 128] bf16
    float * __restrict__ M_part,              // [R, Hkv, GQA, C]
    float * __restrict__ L_part,              // [R, Hkv, GQA, C]
    const int * __restrict__ block_ids,       // [R, Hkv, topk] logical 128-pages (-1 pad)
    const int * __restrict__ block_table,     // [R, max_logical_blocks]
    int max_logical_blocks,
    int topk,
    const int * __restrict__ seq_lens,   // [R] device per-request KV length
    int num_heads_q,
    int num_heads_kv,
    int64_t kv_page_stride,              // elements between consecutive physical pages (K or V tensor)
    int64_t kv_pos_stride,               // elements between consecutive tokens within a page
    int64_t kv_head_stride,              // elements between consecutive kv-heads
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
    const int seq_len_k = seq_lens[req];

    const int qstride       = num_heads_q  * HEAD_DIM;
    const int64_t kvstride    = kv_pos_stride;     // token stride within a page
    const int64_t page_stride = kv_page_stride;    // physical-page stride

    const int hq0 = hkv * GQA;
    const __nv_bfloat16 *Qp    = Q + (int64_t)req * qstride + hq0 * HEAD_DIM;
    const __nv_bfloat16 *Kbase = Kc + (int64_t)hkv * kv_head_stride;
    const __nv_bfloat16 *Vbase = Vc + (int64_t)hkv * kv_head_stride;
    const int *bt_row  = block_table + req * max_logical_blocks;
    const int *blk_row = block_ids   + ((int64_t)req * num_heads_kv + hkv) * topk;

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
        kvv = 0; Kpage = Kbase; Vpage = Vbase;   // safe defaults (0 valid rows)
        int kb = (t >= p_begin && t < p_end) ? blk_row[t] : -1;
        if (kb < 0) return false;                // -1 pad page (fewer than topk selected)
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
    // ALWAYS load: an invalid (-1 pad) page loads with kvv_cur==0, which
    // zero-fills the K/V smem tile. The softmax masks all rows to -inf (kvv==0)
    // and PV then multiplies P(=0) by V(=0) -> 0, never garbage*0 -> NaN. This is
    // required for partial selections (fewer than topk blocks early in decode).
    load_page128_async_pad(sK, Kp0, kvv_cur, (int)kvstride, tid, nthr);
    cp_async_commit();                                   // group: K
    load_page128_async_pad(sV, Vp0, kvv_cur, (int)kvstride, tid, nthr);
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
        // ---- cross-warp max merge: VECTORIZED-LDS layout -------------------
        // sRed is laid out [row(16) x {max,sum}(2) x warp(NUM_WARPS)] so the
        // NUM_WARPS partials a thread reduces for one head-row are CONTIGUOUS in
        // smem. With NUM_WARPS==4 that's one 128-bit LDS.128 (float4) per row
        // instead of 4 scalar LDS.32 -> the cross-warp read issues 4x fewer LDS
        // instructions on the softmax critical path (the +23,808-LDS limiter).
        // Each warp's sub==0 lane writes its scalar partial (the 4 sub-lanes of a
        // grp already hold the SAME max after the intra-warp shuffle above).
        if (sub == 0) {
            sRed[(grp*2 + 0)*NUM_WARPS + wid]     = mx0;
            sRed[((grp+8)*2 + 0)*NUM_WARPS + wid] = mx1;
        }
        __syncthreads();
        float4 vmx0 = *reinterpret_cast<const float4*>(&sRed[(grp*2 + 0)*NUM_WARPS]);
        float4 vmx1 = *reinterpret_cast<const float4*>(&sRed[((grp+8)*2 + 0)*NUM_WARPS]);
        float pmx0 = fmaxf(fmaxf(vmx0.x, vmx0.y), fmaxf(vmx0.z, vmx0.w));
        float pmx1 = fmaxf(fmaxf(vmx1.x, vmx1.y), fmaxf(vmx1.z, vmx1.w));
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
        // ---- cross-warp denom merge: same VECTORIZED-LDS layout ({sum} slot).
        // sub==0 lane writes its partial denom; the sPbf P-write below overlaps
        // the smem latency between the __syncthreads and the float4 read-back.
        if (sub == 0) {
            sRed[(grp*2 + 1)*NUM_WARPS + wid]     = ls0;
            sRed[((grp+8)*2 + 1)*NUM_WARPS + wid] = ls1;
        }
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
        float4 vsl0 = *reinterpret_cast<const float4*>(&sRed[(grp*2 + 1)*NUM_WARPS]);
        float4 vsl1 = *reinterpret_cast<const float4*>(&sRed[((grp+8)*2 + 1)*NUM_WARPS]);
        float gl0 = (vsl0.x + vsl0.y) + (vsl0.z + vsl0.w);
        float gl1 = (vsl1.x + vsl1.y) + (vsl1.z + vsl1.w);
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
            load_page128_async_pad(sK, Kp2, kv2, (int)kvstride, tid, nthr);
            cp_async_commit();
            load_page128_async_pad(sV, Vp2, kv2, (int)kvstride, tid, nthr);
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
// ===========================================================================
// MERGE KERNEL — flash-decoding LSE merge across split-K chunks.
//   grid = (num_requests, Hq).  block = HEAD_DIM threads (128).
//   For q-head hq (kv-head hkv=hq/GQA, local g=hq%GQA), read all chunks'
//   partial (M, L, O[.,g,.]) and combine into final O[req, hq, :].
// ===========================================================================
extern "C" __global__ void
sm120_serve_decode_merge_bf16(
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

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <limits>

// q          : [R, Hq, 128] bf16
// kv_cache   : the M3 fused main cache, logical [num_blocks, 2, 128, Hkv, 128].
//              Physical layout may be NHD or HND; we pass K=[:,0] and V=[:,1]
//              base pointers + the REAL strides read from the tensor, so no copy.
// block_table: [R, max_logical_blocks] int32  (logical->physical page map)
// block_ids  : [R, Hkv, topk] int32           (per-kv-head selected pages, -1 pad)
// seq_lens   : [R] int32 DEVICE tensor        (per-request KV length)
// Returns {o[R,Hq,128] bf16, lse[R,Hq] f32}.  Graph-capture safe (no .item(),
// no .contiguous(); allocations go through the caching allocator).
std::vector<torch::Tensor> forward_sparse_decode_serving(
    torch::Tensor q, torch::Tensor kv_cache,
    torch::Tensor block_table, torch::Tensor block_ids,
    torch::Tensor seq_lens, double softmax_scale,
    int64_t num_kv_heads, int64_t split_chunks_in)
{
    TORCH_CHECK(q.is_cuda() && q.dtype()==torch::kBFloat16 && q.dim()==3 && q.size(2)==128,
                "q must be CUDA bf16 [R,Hq,128]");
    TORCH_CHECK(kv_cache.dim()==5 && kv_cache.size(1)==2 && kv_cache.size(2)==128
                && kv_cache.size(4)==128,
                "kv_cache must be fused [num_blocks,2,128,Hkv,128]");
    TORCH_CHECK(block_ids.dim()==3, "block_ids must be [R,Hkv,topk]");
    TORCH_CHECK(seq_lens.is_cuda() && seq_lens.dtype()==torch::kInt32,
                "seq_lens must be CUDA int32 [R]");
    const int R    = q.size(0);
    const int Hq   = q.size(1);
    const int Hkv  = (int)num_kv_heads;
    const int topk = block_ids.size(2);
    const int max_logical_blocks = block_table.size(1);
    const int GQA_ = Hq / Hkv;
    TORCH_CHECK(GQA_ == GQA, "decode kernel specialized for GQA group == 16");
    TORCH_CHECK(block_ids.size(0)==R && block_ids.size(1)==Hkv, "block_ids [R,Hkv,topk]");

    // Real strides (in elements) of the fused cache. dims: [blk, 2, pos, head, d].
    const auto st = kv_cache.strides();
    const int64_t blk_stride  = st[0];     // physical-page stride
    const int64_t kv_stride   = st[1];     // K vs V (the `2` axis)
    const int64_t pos_stride  = st[2];     // token within a page
    const int64_t head_stride = st[3];     // kv-head
    // K base = cache + 0*kv_stride ; V base = cache + 1*kv_stride.
    auto *base = reinterpret_cast<const __nv_bfloat16*>(kv_cache.data_ptr());
    const __nv_bfloat16 *k_base = base;
    const __nv_bfloat16 *v_base = base + kv_stride;

    int split_chunks = (int)split_chunks_in;
    if (split_chunks <= 0) split_chunks = topk;
    if (split_chunks > topk) split_chunks = topk;
    if (split_chunks < 1) split_chunks = 1;
    int pages_per_chunk = (topk + split_chunks - 1) / split_chunks;
    split_chunks = (topk + pages_per_chunk - 1) / pages_per_chunk;

    auto fopt = torch::dtype(torch::kFloat32).device(q.device());
    auto bopt = torch::dtype(torch::kBFloat16).device(q.device());
    // O_part zeroed and M_part=-inf / L_part=0 so any chunk the partial kernel
    // leaves unwritten (a chunk whose selected pages are all -1 pad, which
    // happens early in generation when fewer than `topk` blocks are selected) is
    // INERT in the LSE merge (weight exp2(-inf-gmax)=0). torch::zeros/full are
    // graph-safe (caching allocator); negligible at bs1.
    auto O_part = torch::zeros({R, Hkv, GQA, split_chunks, HEAD_DIM}, bopt);
    auto M_part = torch::full({R, Hkv, GQA, split_chunks},
                              -std::numeric_limits<float>::infinity(), fopt);
    auto L_part = torch::zeros({R, Hkv, GQA, split_chunks}, fopt);
    auto o   = torch::empty({R, Hq, HEAD_DIM}, bopt);
    auto lse = torch::empty({R, Hq}, fopt);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block_p(NUM_WARPS*32);
    dim3 grid_p(Hkv, split_chunks, R);
    const int KVLD = HEAD_DIM + 8, PLD = P128_N + 8;
    int smem_bytes = (GQA*KVLD + 2*P128_N*KVLD + GQA*PLD) * (int)sizeof(__nv_bfloat16)
                   + NUM_WARPS * GQA * 2 * (int)sizeof(float);
    cudaFuncSetAttribute(sm120_serve_decode_partial_p128_ldsm,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    sm120_serve_decode_partial_p128_ldsm<<<grid_p, block_p, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        k_base, v_base,
        reinterpret_cast<__nv_bfloat16*>(O_part.data_ptr()),
        M_part.data_ptr<float>(), L_part.data_ptr<float>(),
        block_ids.data_ptr<int>(), block_table.data_ptr<int>(),
        max_logical_blocks, topk, seq_lens.data_ptr<int>(), Hq, Hkv,
        blk_stride, pos_stride, head_stride,
        (float)softmax_scale, split_chunks, pages_per_chunk);

    dim3 grid_m(R, Hq, 2);
    dim3 block_m(HEAD_DIM / 2);
    int merge_smem = split_chunks * (int)sizeof(float);
    sm120_serve_decode_merge_bf16<<<grid_m, block_m, merge_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(O_part.data_ptr()),
        M_part.data_ptr<float>(), L_part.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()), lse.data_ptr<float>(),
        split_chunks, Hq, Hkv);
    return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_sparse_decode_serving", &forward_sparse_decode_serving,
          "SM120 block-sparse fused-cache decode (page-128, ldmatrix W4=3), graph-safe",
          pybind11::arg("q"), pybind11::arg("kv_cache"),
          pybind11::arg("block_table"), pybind11::arg("block_ids"),
          pybind11::arg("seq_lens"), pybind11::arg("softmax_scale"),
          pybind11::arg("num_kv_heads"), pybind11::arg("split_chunks") = 0);
}
