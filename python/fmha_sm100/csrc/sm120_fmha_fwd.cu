// SM120 Dense FlashAttention-2 Forward Kernel
// Per-warp HMMA BF16 m16n8k16 with cp.async pipeline
//
// Architecture:
//   - 4 warps per CTA, each handles 16 Q rows
//   - cp.async pipeline for K/V loads
//   - Online softmax in registers (FA2 algorithm)
//   - Tile: M=64, N=64, D=128
//
// Compile: nvcc -gencode arch=compute_120f,code=sm_120f -O3
// SPDX-License-Identifier: MIT

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int BLK_M = 64;
static constexpr int BLK_N = 64;
static constexpr int HEAD_DIM = 128;
static constexpr int NUM_WARPS = 4;
static constexpr int WARP_M = BLK_M / NUM_WARPS;
static constexpr int NUM_STAGES = 2;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 16;

static constexpr int SMEM_Q_BYTES = BLK_M * HEAD_DIM * sizeof(__nv_bfloat16);
static constexpr int SMEM_K_BYTES = BLK_N * HEAD_DIM * sizeof(__nv_bfloat16);
static constexpr int SMEM_V_BYTES = BLK_N * HEAD_DIM * sizeof(__nv_bfloat16);
static constexpr int SMEM_TOTAL = SMEM_Q_BYTES + (SMEM_K_BYTES + SMEM_V_BYTES) * NUM_STAGES;

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

__device__ __forceinline__ uint32_t f32x2_to_bf16x2(float lo, float hi) {
    uint32_t r;
    reinterpret_cast<__nv_bfloat16*>(&r)[0] = __float2bfloat16(lo);
    reinterpret_cast<__nv_bfloat16*>(&r)[1] = __float2bfloat16(hi);
    return r;
}

__device__ __forceinline__ uint32_t smem_bf16x2(
    const __nv_bfloat16 *p0, const __nv_bfloat16 *p1)
{
    uint32_t r;
    reinterpret_cast<__nv_bfloat16*>(&r)[0] = *p0;
    reinterpret_cast<__nv_bfloat16*>(&r)[1] = *p1;
    return r;
}

// ===========================================================================
extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32)
sm120_fmha_fwd_bf16(
    const __nv_bfloat16 * __restrict__ Q,
    const __nv_bfloat16 * __restrict__ K,
    const __nv_bfloat16 * __restrict__ V,
    __nv_bfloat16 * __restrict__ O,
    float * __restrict__ LSE,
    int seq_len_q,
    int seq_len_k,
    int num_heads_q,
    int num_heads_kv,
    float softmax_scale)
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
    const int kvstride  = num_heads_kv * HEAD_DIM;

    const __nv_bfloat16 *Qp = Q + hq  * HEAD_DIM;
    const __nv_bfloat16 *Kp = K + hkv * HEAD_DIM;
    const __nv_bfloat16 *Vp = V + hkv * HEAD_DIM;
    __nv_bfloat16       *Op = O + hq  * HEAD_DIM;

    __nv_bfloat16 *sQ = smem_raw;
    __nv_bfloat16 *sK = sQ + BLK_M * HEAD_DIM;
    __nv_bfloat16 *sV = sK + BLK_N * HEAD_DIM * NUM_STAGES;

    load_tile_async(sQ, Qp, qstart, qvalid, qstride, tid, nthr);
    cp_async_commit();
    cp_async_wait(0);
    __syncthreads();

    const int wm = wid * WARP_M;

    float Oa[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        Oa[i][0] = Oa[i][1] = Oa[i][2] = Oa[i][3] = 0.f;

    float rm[2] = {-INFINITY, -INFINITY};
    float rl[2] = {0.f, 0.f};

    const int nkvblk = (seq_len_k + BLK_N - 1) / BLK_N;
    constexpr float LG2E = 1.4426950408889634f;

    for (int kb = 0; kb < nkvblk; kb++) {
        const int kvs = kb * BLK_N;
        const int kvv = min(BLK_N, seq_len_k - kvs);

        __nv_bfloat16 *sKc = sK;
        load_tile_async(sKc, Kp, kvs, kvv, kvstride, tid, nthr);
        cp_async_commit();
        cp_async_wait(0);
        __syncthreads();

        // ----- QK GEMM -----
        float Sr[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            Sr[i][0] = Sr[i][1] = Sr[i][2] = Sr[i][3] = 0.f;

        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int qb0 = (wm + grp)     * HEAD_DIM + ks * MMA_K;
            const int qb1 = (wm + grp + 8) * HEAD_DIM + ks * MMA_K;
            // SM120 fragment layout: a0→D[0/1](rows 0-7), a1→D[2/3](rows 8-15)
            // a0/a2 = k-positions sub*2:sub*2+1 and sub*2+8:sub*2+9 for rows 0-7
            // a1/a3 = same k-positions for rows 8-15
            uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sQ[qb0 + sub*2]);
            uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sQ[qb1 + sub*2]);
            uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sQ[qb0 + sub*2 + 8]);
            uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sQ[qb1 + sub*2 + 8]);

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int kr = (nt * MMA_N + grp) * HEAD_DIM + ks * MMA_K;
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sKc[kr + sub*2]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sKc[kr + sub*2 + 8]);
                hmma_bf16_m16n8k16(
                    Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            Sr[i][0] *= softmax_scale;
            Sr[i][1] *= softmax_scale;
            Sr[i][2] *= softmax_scale;
            Sr[i][3] *= softmax_scale;
        }

        // ----- mask invalid KV positions (padding) to -inf -----
        // Output fragment N-position: Sr[nt][0/2] at col nt*8+sub*2, Sr[nt][1/3] at nt*8+sub*2+1
        if (kvv < BLK_N) {
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int n0 = nt * MMA_N + sub * 2;
                int n1 = n0 + 1;
                if (n0 >= kvv) { Sr[nt][0] = -INFINITY; Sr[nt][2] = -INFINITY; }
                if (n1 >= kvv) { Sr[nt][1] = -INFINITY; Sr[nt][3] = -INFINITY; }
            }
        }

        // ----- online softmax -----
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

        // Protect against -inf - (-inf) = NaN when row has no valid scores
        float s0 = (rm[0] == -INFINITY) ? 0.f : exp2f(LG2E * (rm[0] - mn0));
        float s1 = (rm[1] == -INFINITY) ? 0.f : exp2f(LG2E * (rm[1] - mn1));
        rl[0] *= s0;
        rl[1] *= s1;
        #pragma unroll
        for (int d = 0; d < 16; d++) {
            Oa[d][0] *= s0; Oa[d][1] *= s0;
            Oa[d][2] *= s1; Oa[d][3] *= s1;
        }

        float ls0 = 0.f, ls1 = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            Sr[i][0] = exp2f(LG2E * (Sr[i][0] - mn0));
            Sr[i][1] = exp2f(LG2E * (Sr[i][1] - mn0));
            Sr[i][2] = exp2f(LG2E * (Sr[i][2] - mn1));
            Sr[i][3] = exp2f(LG2E * (Sr[i][3] - mn1));
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

        // ----- load V -----
        __nv_bfloat16 *sVc = sV;
        load_tile_async(sVc, Vp, kvs, kvv, kvstride, tid, nthr);
        cp_async_commit();
        cp_async_wait(0);
        __syncthreads();

        // ----- PV GEMM -----
        // SM120 HMMA m16n8k16 has coupled a0/a2 fragments: both contribute
        // to D[0]/D[1] AND D[2]/D[3]. Cannot zero one without affecting the other.
        // Standard single-MMA call with all fragments populated.
        #pragma unroll
        for (int ns = 0; ns < 4; ns++) {
            const int tl = 2 * ns;
            const int th = 2 * ns + 1;
            // SM120 layout: a0→D[0/1], a1→D[2/3], a2/a3 = extended k
            uint32_t pa0 = f32x2_to_bf16x2(Sr[tl][0], Sr[tl][1]);  // rows 0-7
            uint32_t pa1 = f32x2_to_bf16x2(Sr[tl][2], Sr[tl][3]);  // rows 8-15 (was a2 on SM80)
            uint32_t pa2 = f32x2_to_bf16x2(Sr[th][0], Sr[th][1]);  // rows 0-7 extended k
            uint32_t pa3 = f32x2_to_bf16x2(Sr[th][2], Sr[th][3]);  // rows 8-15 extended k

            #pragma unroll
            for (int dt = 0; dt < 16; dt++) {
                const int vc = dt * MMA_N + grp;
                const int vr = ns * MMA_K + sub * 2;
                uint32_t vb0 = smem_bf16x2(
                    &sVc[ vr      * HEAD_DIM + vc],
                    &sVc[(vr + 1) * HEAD_DIM + vc]);
                uint32_t vb1 = smem_bf16x2(
                    &sVc[(vr + 8) * HEAD_DIM + vc],
                    &sVc[(vr + 9) * HEAD_DIM + vc]);
                hmma_bf16_m16n8k16(
                    Oa[dt][0], Oa[dt][1], Oa[dt][2], Oa[dt][3],
                    pa0, pa1, pa2, pa3, vb0, vb1,
                    Oa[dt][0], Oa[dt][1], Oa[dt][2], Oa[dt][3]);
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
