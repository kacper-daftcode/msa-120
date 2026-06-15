// SM120 Block-Sparse FlashAttention-2 Forward Kernel — PAGED-KV variant
//
// This is a paged-KV adaptation of sm120_fmha_sparse.cu. KV is no longer a
// contiguous [Skv, Hkv, 128] tensor; instead it lives in a page pool:
//   k_cache, v_cache : [num_pages, PAGE_SIZE, Hkv, 128] bf16
// with PAGE_SIZE == BLK_N == 64 for v1 (one logical KV block == one physical
// page). A block_table maps logical KV block index -> physical page id:
//   block_table : int32 [num_m_blocks, max_logical_blocks]
//
// When the kernel would load logical block `kb`, it instead loads physical
// page block_table[m_blk, kb]. Everything else (QK, softmax, PV, causal via
// absolute kpos = kb*BLK_N + local) is byte-for-byte identical to the
// contiguous kernel — so paged output must EQUAL contiguous output exactly.
//
// Compile: nvcc -gencode arch=compute_120f,code=sm_120f -O3
// SPDX-License-Identifier: MIT

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int BLK_M = 64;
static constexpr int BLK_N = 64;
static constexpr int PAGE_SIZE = 64;   // v1: one logical block == one page
static constexpr int HEAD_DIM = 128;
static constexpr int NUM_WARPS = 4;
static constexpr int WARP_M = BLK_M / NUM_WARPS;
static constexpr int NUM_STAGES = 1;

static constexpr int MMA_M = 16;
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

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait(int n = 0) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else if (n == 1) asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ void hmma_bf16_m16n8k16(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
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
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "%14, {0, 1}, "
        "%15, {0, 1};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "r"(sfb)
    );
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

// Load one BLK_N x HEAD_DIM tile from a single physical page.
// `gmem` already points at the page base for the correct kv head, i.e.
// page_base = cache + page*(PAGE_SIZE*Hkv*128) + hkv*128.
// stride_row == Hkv*128 (token-to-token stride within the page).
__device__ void load_page_async(
    __nv_bfloat16 *smem_tile,
    const __nv_bfloat16 *gmem,
    int num_valid_rows,
    int stride_row,
    int tid,
    int num_threads)
{
    const int elems_per_copy = 8;
    const int total_elems = BLK_N * HEAD_DIM;
    const int copies_per_thread = (total_elems + num_threads * elems_per_copy - 1)
                                / (num_threads * elems_per_copy);
    for (int c = 0; c < copies_per_thread; c++) {
        int elem_idx = (tid + c * num_threads) * elems_per_copy;
        if (elem_idx >= total_elems) break;
        int row = elem_idx / HEAD_DIM;
        int col = elem_idx % HEAD_DIM;
        __nv_bfloat16 *dst = smem_tile + row * HEAD_DIM + col;
        if (row < num_valid_rows) {
            const __nv_bfloat16 *src = gmem + row * stride_row + col;
            cp_async_16b(dst, src);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
}

__device__ void load_tile_async(
    __nv_bfloat16 *smem_tile,
    const __nv_bfloat16 *gmem,
    int row_start,
    int num_valid_rows,
    int stride_row,
    int tid,
    int num_threads)
{
    const int elems_per_copy = 8;
    const int total_elems = BLK_N * HEAD_DIM;
    const int copies_per_thread = (total_elems + num_threads * elems_per_copy - 1)
                                / (num_threads * elems_per_copy);
    for (int c = 0; c < copies_per_thread; c++) {
        int elem_idx = (tid + c * num_threads) * elems_per_copy;
        if (elem_idx >= total_elems) break;
        int row = elem_idx / HEAD_DIM;
        int col = elem_idx % HEAD_DIM;
        __nv_bfloat16 *dst = smem_tile + row * HEAD_DIM + col;
        if (row < num_valid_rows) {
            const __nv_bfloat16 *src = gmem + (row_start + row) * stride_row + col;
            cp_async_16b(dst, src);
        } else {
            *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ uint32_t lds_u32(const void *smem_ptr) {
    uint32_t smem_addr, r;
    asm volatile(
        "{\n"
        ".reg .u64 u;\n"
        "cvta.to.shared.u64 u, %1;\n"
        "cvt.u32.u64 %2, u;\n"
        "ld.shared.b32 %0, [%2];\n"
        "}\n" : "=r"(r), "+l"(smem_ptr), "=r"(smem_addr));
    return r;
}

// ===========================================================================
extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
sm120_fmha_sparse_paged_bf16(
    const __nv_bfloat16 * __restrict__ Q,        // [Sq, Hq, 128]
    const __nv_bfloat16 * __restrict__ Kc,       // [num_pages, PAGE_SIZE, Hkv, 128]
    const __nv_bfloat16 * __restrict__ Vc,       // [num_pages, PAGE_SIZE, Hkv, 128]
    __nv_bfloat16 * __restrict__ O,
    float * __restrict__ LSE,
    const int * __restrict__ block_ids,          // [num_m_blocks|Sq, topk] (-1=pad)
    const int * __restrict__ block_table,        // [num_m_blocks, max_logical_blocks]
    int max_logical_blocks,
    int topk,
    int seq_len_q,
    int seq_len_k,
    int num_heads_q,
    int num_heads_kv,
    float softmax_scale,
    int causal,
    int per_query)
{
    extern __shared__ __nv_bfloat16 smem_raw[];

    const int tid   = threadIdx.x;
    const int wid   = tid / 32;
    const int lid   = tid % 32;
    const int grp   = lid / 4;
    const int sub   = lid % 4;
    const int nthr  = NUM_WARPS * 32;

    const int m_blk     = blockIdx.x;
    const int hq        = blockIdx.y;
    const int hkv       = hq / (num_heads_q / num_heads_kv);
    const int qstart    = m_blk * BLK_M;
    const int qvalid    = min(BLK_M, seq_len_q - qstart);

    const int qstride   = num_heads_q  * HEAD_DIM;
    const int kvstride  = num_heads_kv * HEAD_DIM;             // token stride in a page
    const int page_stride = PAGE_SIZE * num_heads_kv * HEAD_DIM; // page-to-page stride

    const __nv_bfloat16 *Qp = Q + hq  * HEAD_DIM;
    const __nv_bfloat16 *Kbase = Kc + hkv * HEAD_DIM;          // + page*page_stride later
    const __nv_bfloat16 *Vbase = Vc + hkv * HEAD_DIM;
    __nv_bfloat16       *Op = O + hq  * HEAD_DIM;

    const int *bt_row = block_table + m_blk * max_logical_blocks;

    __nv_bfloat16 *sQ = smem_raw;
    __nv_bfloat16 *sK = sQ + BLK_M * HEAD_DIM;
    __nv_bfloat16 *sV = sK + BLK_N * HEAD_DIM * NUM_STAGES;

    load_tile_async(sQ, Qp, qstart, qvalid, qstride, tid, nthr);
    cp_async_commit();
    cp_async_wait(0);
    __syncthreads();

    const int wm = wid * WARP_M;

    const int sub2     = sub * 2;
    const int qbase0   = (wm + grp) * HEAD_DIM;
    const int qbase1   = (wm + grp + 8) * HEAD_DIM;
    const int grp_hdim = grp * HEAD_DIM;

    const __nv_bfloat16 * __restrict__ sQ_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sQ, 16));

    float Oa[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        Oa[i][0] = Oa[i][1] = Oa[i][2] = Oa[i][3] = 0.f;

    float rm[2] = {-INFINITY, -INFINITY};
    float rl[2] = {0.f, 0.f};

    constexpr float LG2E = 1.4426950408889634f;

    // v1: KV blocks are exactly BLK_N=64 = PAGE_SIZE.
    const int num_kv_blocks = (seq_len_k + BLK_N - 1) / BLK_N;

    int *s_present = reinterpret_cast<int*>(sV + BLK_N * HEAD_DIM);
    int *s_union   = s_present + num_kv_blocks;
    int *s_meta    = s_union + num_kv_blocks;
    int n_iter;
    const int *iter_blocks;
    const int *blk_row = block_ids + m_blk * topk;

    if (per_query) {
        for (int b = tid; b < num_kv_blocks; b += nthr) s_present[b] = 0;
        if (tid == 0) s_meta[0] = 0;
        __syncthreads();
        const int my_r0 = qstart + wm + grp;
        const int my_r1 = qstart + wm + grp + 8;
        if (sub == 0) {
            #pragma unroll 1
            for (int t = 0; t < topk; t++) {
                if (my_r0 < seq_len_q) {
                    int b = block_ids[my_r0 * topk + t];
                    if (b >= 0 && b < num_kv_blocks) atomicExch(&s_present[b], 1);
                }
                if (my_r1 < seq_len_q) {
                    int b = block_ids[my_r1 * topk + t];
                    if (b >= 0 && b < num_kv_blocks) atomicExch(&s_present[b], 1);
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            int c = 0;
            for (int b = 0; b < num_kv_blocks; b++)
                if (s_present[b]) s_union[c++] = b;
            s_meta[0] = c;
        }
        __syncthreads();
        n_iter = s_meta[0];
        iter_blocks = s_union;
    } else {
        n_iter = topk;
        iter_blocks = blk_row;
    }

    for (int t = 0; t < n_iter; t++) {
        const int kb = iter_blocks[t];
        if (kb < 0) continue;                       // -1 padding
        const int kvs = kb * BLK_N;                 // absolute logical token offset
        if (kvs >= seq_len_k) continue;
        const int kvv = min(BLK_N, seq_len_k - kvs);

        // Translate logical block -> physical page via block_table.
        const int page = bt_row[kb];
        const __nv_bfloat16 *Kpage = Kbase + (int64_t)page * page_stride;
        const __nv_bfloat16 *Vpage = Vbase + (int64_t)page * page_stride;

        // (1) Load K page -> sK and wait
        load_page_async(sK, Kpage, kvv, kvstride, tid, nthr);
        cp_async_commit();
        cp_async_wait(0);
        __syncthreads();

        // (2) Start V page load -> sV (overlaps with QK compute)
        __nv_bfloat16 *sVc = sV;
        load_page_async(sVc, Vpage, kvv, kvstride, tid, nthr);
        cp_async_commit();

        // (3) QK GEMM
        __nv_bfloat16 *sKc = sK;
        const __nv_bfloat16 * __restrict__ sKc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sKc, 16));

        float Sr[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            Sr[i][0] = Sr[i][1] = Sr[i][2] = Sr[i][3] = 0.f;

        uint32_t b0_next = lds_u32(&sKc_a[grp_hdim + sub2]);
        uint32_t b1_next = lds_u32(&sKc_a[grp_hdim + sub2 + 8]);

        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int qoff = ks * MMA_K_BF16;

            uint32_t a0 = lds_u32(&sQ_a[qbase0 + qoff + sub2]);
            uint32_t a1 = lds_u32(&sQ_a[qbase1 + qoff + sub2]);
            uint32_t a2 = lds_u32(&sQ_a[qbase0 + qoff + sub2 + 8]);
            uint32_t a3 = lds_u32(&sQ_a[qbase1 + qoff + sub2 + 8]);

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t b0 = b0_next;
                uint32_t b1 = b1_next;

                hmma_bf16_m16n8k16(
                    Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);

                if (nt < 7) {
                    const int kr_n = ((nt + 1) * MMA_N + grp) * HEAD_DIM + qoff;
                    b0_next = lds_u32(&sKc_a[kr_n + sub2]);
                    b1_next = lds_u32(&sKc_a[kr_n + sub2 + 8]);
                } else if (ks < 7) {
                    const int kr_n = grp_hdim + (ks + 1) * MMA_K_BF16;
                    b0_next = lds_u32(&sKc_a[kr_n + sub2]);
                    b1_next = lds_u32(&sKc_a[kr_n + sub2 + 8]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            Sr[i][0] *= softmax_scale;
            Sr[i][1] *= softmax_scale;
            Sr[i][2] *= softmax_scale;
            Sr[i][3] *= softmax_scale;
        }

        if (kvv < BLK_N) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int n0 = nt * MMA_N + sub * 2;
                int n1 = n0 + 1;
                if (n0 >= kvv) { Sr[nt][0] = -INFINITY; Sr[nt][2] = -INFINITY; }
                if (n1 >= kvv) { Sr[nt][1] = -INFINITY; Sr[nt][3] = -INFINITY; }
            }
        }

        if (causal) {
            const int q_r0 = qstart + wm + grp;
            const int q_r1 = qstart + wm + grp + 8;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int kp0 = kvs + nt * MMA_N + sub * 2;
                int kp1 = kp0 + 1;
                if (kp0 > q_r0) Sr[nt][0] = -INFINITY;
                if (kp1 > q_r0) Sr[nt][1] = -INFINITY;
                if (kp0 > q_r1) Sr[nt][2] = -INFINITY;
                if (kp1 > q_r1) Sr[nt][3] = -INFINITY;
            }
        }

        if (per_query) {
            const int q_r0 = qstart + wm + grp;
            const int q_r1 = qstart + wm + grp + 8;
            bool sel0 = false, sel1 = false;
            #pragma unroll 1
            for (int tt = 0; tt < topk; tt++) {
                if (q_r0 < seq_len_q && block_ids[q_r0 * topk + tt] == kb) sel0 = true;
                if (q_r1 < seq_len_q && block_ids[q_r1 * topk + tt] == kb) sel1 = true;
            }
            if (!sel0) {
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) { Sr[nt][0] = -INFINITY; Sr[nt][1] = -INFINITY; }
            }
            if (!sel1) {
                #pragma unroll
                for (int nt = 0; nt < 8; nt++) { Sr[nt][2] = -INFINITY; Sr[nt][3] = -INFINITY; }
            }
        }

        // (4) Online softmax
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            mx0 = fmaxf(mx0, fmaxf(Sr[i][0], Sr[i][1]));
            mx1 = fmaxf(mx1, fmaxf(Sr[i][2], Sr[i][3]));
        }
        {
            float t;
            t = __shfl_xor_sync(0xffffffff, mx0, 1); mx0 = fmaxf(mx0, t);
            t = __shfl_xor_sync(0xffffffff, mx0, 2); mx0 = fmaxf(mx0, t);
            t = __shfl_xor_sync(0xffffffff, mx1, 1); mx1 = fmaxf(mx1, t);
            t = __shfl_xor_sync(0xffffffff, mx1, 2); mx1 = fmaxf(mx1, t);
        }

        float mn0 = fmaxf(rm[0], mx0);
        float mn1 = fmaxf(rm[1], mx1);

        float s0 = (rm[0] == -INFINITY) ? 0.f : exp2f(LG2E * (rm[0] - mn0));
        float s1 = (rm[1] == -INFINITY) ? 0.f : exp2f(LG2E * (rm[1] - mn1));
        rl[0] *= s0;
        rl[1] *= s1;
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            Oa[d][0] *= s0; Oa[d][1] *= s0;
            Oa[d][2] *= s1; Oa[d][3] *= s1;
        }

        const bool dead0 = (mn0 == -INFINITY);
        const bool dead1 = (mn1 == -INFINITY);
        float ls0 = 0.f, ls1 = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            Sr[i][0] = dead0 ? 0.f : exp2f(LG2E * (Sr[i][0] - mn0));
            Sr[i][1] = dead0 ? 0.f : exp2f(LG2E * (Sr[i][1] - mn0));
            Sr[i][2] = dead1 ? 0.f : exp2f(LG2E * (Sr[i][2] - mn1));
            Sr[i][3] = dead1 ? 0.f : exp2f(LG2E * (Sr[i][3] - mn1));
            ls0 += Sr[i][0] + Sr[i][1];
            ls1 += Sr[i][2] + Sr[i][3];
        }
        {
            float t;
            t = __shfl_xor_sync(0xffffffff, ls0, 1); ls0 += t;
            t = __shfl_xor_sync(0xffffffff, ls0, 2); ls0 += t;
            t = __shfl_xor_sync(0xffffffff, ls1, 1); ls1 += t;
            t = __shfl_xor_sync(0xffffffff, ls1, 2); ls1 += t;
        }

        rm[0] = mn0; rm[1] = mn1;
        rl[0] += ls0; rl[1] += ls1;

        // (4b) Write P to SMEM as FP8 — reuse sK buffer
        uint8_t *sP = reinterpret_cast<uint8_t*>(sKc);
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int kv_col = nt * MMA_N + sub * 2;
            uint16_t pair_lo = cvt_2f32_to_e4m3x2(Sr[nt][0], Sr[nt][1]);
            uint16_t pair_hi = cvt_2f32_to_e4m3x2(Sr[nt][2], Sr[nt][3]);
            *reinterpret_cast<uint16_t*>(&sP[(wm + grp) * BLK_N + kv_col]) = pair_lo;
            *reinterpret_cast<uint16_t*>(&sP[(wm + grp + 8) * BLK_N + kv_col]) = pair_hi;
        }

        // (5) Wait for V page
        cp_async_wait(0);
        __syncthreads();

        // (7) PV GEMM — QMMA.SF FP8 m16n8k32
        const __nv_bfloat16 * __restrict__ sVc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sVc, 16));

        #pragma unroll
        for (int ns = 0; ns < 2; ns++) {
            const int pa_base0 = (wm + grp) * BLK_N + ns * MMA_K_FP8;
            const int pa_base1 = (wm + grp + 8) * BLK_N + ns * MMA_K_FP8;
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub * 4]);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub * 4]);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub * 4 + 16]);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub * 4 + 16]);

            const int vr_lo = ns * MMA_K_FP8 + sub * 4;
            const int vr_hi = vr_lo + 16;

            int vcol = grp;
            uint32_t vb0_next = f32x4_to_e4m3x4(
                __bfloat162float(sVc_a[vr_lo * HEAD_DIM + vcol]),
                __bfloat162float(sVc_a[(vr_lo + 1) * HEAD_DIM + vcol]),
                __bfloat162float(sVc_a[(vr_lo + 2) * HEAD_DIM + vcol]),
                __bfloat162float(sVc_a[(vr_lo + 3) * HEAD_DIM + vcol]));
            uint32_t vb1_next = f32x4_to_e4m3x4(
                __bfloat162float(sVc_a[vr_hi * HEAD_DIM + vcol]),
                __bfloat162float(sVc_a[(vr_hi + 1) * HEAD_DIM + vcol]),
                __bfloat162float(sVc_a[(vr_hi + 2) * HEAD_DIM + vcol]),
                __bfloat162float(sVc_a[(vr_hi + 3) * HEAD_DIM + vcol]));

            #pragma unroll
            for (int dt = 0; dt < 16; dt++) {
                uint32_t vb0 = vb0_next;
                uint32_t vb1 = vb1_next;

                qmma_sf_fp8_m16n8k32(
                    Oa[dt][0], Oa[dt][1], Oa[dt][2], Oa[dt][3],
                    pa0, pa1, pa2, pa3, vb0, vb1,
                    Oa[dt][0], Oa[dt][1], Oa[dt][2], Oa[dt][3]);

                if (dt < 15) {
                    vcol = (dt + 1) * MMA_N + grp;
                    vb0_next = f32x4_to_e4m3x4(
                        __bfloat162float(sVc_a[vr_lo * HEAD_DIM + vcol]),
                        __bfloat162float(sVc_a[(vr_lo + 1) * HEAD_DIM + vcol]),
                        __bfloat162float(sVc_a[(vr_lo + 2) * HEAD_DIM + vcol]),
                        __bfloat162float(sVc_a[(vr_lo + 3) * HEAD_DIM + vcol]));
                    vb1_next = f32x4_to_e4m3x4(
                        __bfloat162float(sVc_a[vr_hi * HEAD_DIM + vcol]),
                        __bfloat162float(sVc_a[(vr_hi + 1) * HEAD_DIM + vcol]),
                        __bfloat162float(sVc_a[(vr_hi + 2) * HEAD_DIM + vcol]),
                        __bfloat162float(sVc_a[(vr_hi + 3) * HEAD_DIM + vcol]));
                }
            }
        }

        __syncthreads();
    }

    // ----- epilogue -----
    const float inv0 = (rl[0] > 0.f) ? (1.f / rl[0]) : 0.f;
    const float inv1 = (rl[1] > 0.f) ? (1.f / rl[1]) : 0.f;
    const int r0 = qstart + wm + grp;
    const int r1 = qstart + wm + grp + 8;

    #pragma unroll
    for (int dt = 0; dt < 16; dt++) {
        int c0 = dt * MMA_N + sub * 2;
        int c1 = c0 + 1;
        if (r0 < seq_len_q) {
            Op[r0 * qstride + c0] = __float2bfloat16(Oa[dt][0] * inv0);
            Op[r0 * qstride + c1] = __float2bfloat16(Oa[dt][1] * inv0);
        }
        if (r1 < seq_len_q) {
            Op[r1 * qstride + c0] = __float2bfloat16(Oa[dt][2] * inv1);
            Op[r1 * qstride + c1] = __float2bfloat16(Oa[dt][3] * inv1);
        }
    }

    if (sub == 0) {
        if (r0 < seq_len_q)
            LSE[r0 * num_heads_q + hq] = rm[0] + logf(rl[0]);
        if (r1 < seq_len_q)
            LSE[r1 * num_heads_q + hq] = rm[1] + logf(rl[1]);
    }
}

// ============================ torch binding ============================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// q: [Sq, Hq, 128] bf16.
// k_cache, v_cache: [num_pages, PAGE_SIZE=64, Hkv, 128] bf16.
// block_table: int32 [num_m_blocks, max_logical_blocks] logical block -> page id.
// block_ids: int32 [num_m_blocks|Sq, topk] selected LOGICAL KV blocks (-1=pad).
std::vector<torch::Tensor> forward_sparse_paged_bf16(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor block_ids,
    double softmax_scale, bool causal, int64_t seq_len_k)
{
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16 && q.dim() == 3 && q.size(2) == 128,
                "q must be CUDA bf16 [Sq,Hq,128]");
    TORCH_CHECK(k_cache.is_cuda() && k_cache.dtype() == torch::kBFloat16 && k_cache.dim() == 4 &&
                k_cache.size(1) == 64 && k_cache.size(3) == 128,
                "k_cache must be CUDA bf16 [num_pages,64,Hkv,128]");
    TORCH_CHECK(v_cache.is_cuda() && v_cache.dtype() == torch::kBFloat16 && v_cache.dim() == 4 &&
                v_cache.size(1) == 64 && v_cache.size(3) == 128,
                "v_cache must be CUDA bf16 [num_pages,64,Hkv,128]");
    TORCH_CHECK(block_table.is_cuda() && block_table.dtype() == torch::kInt32 && block_table.dim() == 2,
                "block_table must be CUDA int32 [num_m_blocks,max_logical_blocks]");
    TORCH_CHECK(block_ids.is_cuda() && block_ids.dtype() == torch::kInt32 && block_ids.dim() == 2,
                "block_ids must be CUDA int32 [num_m_blocks,topk] or [Sq,topk]");

    q = q.contiguous(); k_cache = k_cache.contiguous(); v_cache = v_cache.contiguous();
    block_table = block_table.contiguous(); block_ids = block_ids.contiguous();

    const int seq_q = q.size(0), num_heads_q = q.size(1);
    const int num_heads_kv = k_cache.size(2);
    const int topk = block_ids.size(1);
    const int BLK_M = 64;
    const int num_m_blocks = (seq_q + BLK_M - 1) / BLK_M;
    const int max_logical_blocks = block_table.size(1);
    // seq_len_k for masking (partial last block). Caller passes the TRUE KV
    // length so that the partial-last-block mask matches the contiguous kernel.
    const int seq_k = (int)seq_len_k;

    TORCH_CHECK(block_table.size(0) == num_m_blocks,
                "block_table rows must equal num_m_blocks");

    int per_query;
    if (block_ids.size(0) == num_m_blocks)      per_query = 0;
    else if (block_ids.size(0) == seq_q)        per_query = 1;
    else { TORCH_CHECK(false, "block_ids rows must equal num_m_blocks or seq_q"); per_query = 0; }

    auto o = torch::zeros_like(q);
    auto lse = torch::zeros({seq_q, num_heads_q},
                            torch::dtype(torch::kFloat32).device(q.device()));

    const int num_kv_blocks = max_logical_blocks;

    dim3 grid(num_m_blocks, num_heads_q);
    dim3 block(128);
    int smem_bytes = 64 * 128 * 2 * 3;   // 48KB single-buffer (Q/K/V)
    if (per_query)
        smem_bytes += (2 * num_kv_blocks + 1) * (int)sizeof(int);
    cudaFuncSetAttribute(sm120_fmha_sparse_paged_bf16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    auto stream = at::cuda::getCurrentCUDAStream();
    sm120_fmha_sparse_paged_bf16<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
        lse.data_ptr<float>(),
        block_ids.data_ptr<int>(),
        block_table.data_ptr<int>(),
        max_logical_blocks, topk,
        seq_q, seq_k, num_heads_q, num_heads_kv, (float)softmax_scale,
        causal ? 1 : 0, per_query);
    return {o, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_sparse_paged", &forward_sparse_paged_bf16,
          "SM120 block-sparse paged-KV FA2 forward (BF16)",
          pybind11::arg("q"), pybind11::arg("k_cache"), pybind11::arg("v_cache"),
          pybind11::arg("block_table"), pybind11::arg("block_ids"),
          pybind11::arg("softmax_scale"), pybind11::arg("causal") = false,
          pybind11::arg("seq_len_k"));
}
