// SM120 Dense FlashAttention-2 Forward Kernel
// Per-warp HMMA BF16 m16n8k16 for QK + QMMA.SF FP8 m16n8k32 for PV
//
// KEY OPTIMIZATION: PV GEMM uses QMMA.SF on INT_ARITH pipeline (pipe 37),
// which executes in PARALLEL with FP_ARITH/MUFU operations (softmax rescale,
// BF16→FP8 conversion). This is possible because on SM120:
//   - HMMA BF16 (mma.sync m16n8k16) → pipe_class 30
//   - QMMA.SF FP8 (mma.sync m16n8k32 block_scale) → pipe_class 37 (INT_ARITH)
//   - These are DIFFERENT execution units and can overlap!
//
// Additionally, QMMA.SF processes K=32 per instruction (vs K=16 for HMMA),
// halving the PV inner loop from 4 ns-steps to 2 ns-steps.
//
// Architecture:
//   - 4 warps per CTA, each handles 16 Q rows
//   - cp.async pipeline for K/V loads
//   - Online softmax in registers (FA2 algorithm)
//   - Tile: M=64, N=64, D=128
//   - QK GEMM: HMMA BF16 m16n8k16 (pipe 30)
//   - PV GEMM: QMMA.SF FP8 m16n8k32 (pipe 37) — overlaps with BF16→FP8
//              conversion on FP_ARITH pipe 36 and SMEM loads on pipe 6
//
// Pipeline overlap in PV inner loop:
//   pipe 37 (INT_ARITH): QMMA.SF executing
//   pipe 36 (FP_ARITH):  BF16→FP32→FP8 conversion for next V fragment
//   pipe  6 (LDSM):      SMEM loads for next V fragment
//   All three execute simultaneously!
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
static constexpr int HEAD_DIM = 128;
static constexpr int NUM_WARPS = 4;
static constexpr int WARP_M = BLK_M / NUM_WARPS;
static constexpr int NUM_STAGES = 1;   // single-buffer K: 64KB->48KB SMEM,
                                       // occupancy 1->2 blocks/SM. The 2-stage
                                       // K prefetch bought nothing (ncu: DRAM
                                       // 1.3%, L2 hit 97.5% — not memory bound).

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K_BF16 = 16;
static constexpr int MMA_K_FP8 = 32;

static constexpr int SMEM_Q_BYTES = BLK_M * HEAD_DIM * sizeof(__nv_bfloat16);
static constexpr int SMEM_K_BYTES = BLK_N * HEAD_DIM * sizeof(__nv_bfloat16);
static constexpr int SMEM_V_BYTES = BLK_N * HEAD_DIM * sizeof(__nv_bfloat16);
static constexpr int SMEM_TOTAL = SMEM_Q_BYTES + SMEM_K_BYTES * NUM_STAGES + SMEM_V_BYTES;

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

// QMMA.SF FP8 m16n8k32 — executes on INT_ARITH pipeline (pipe 37)
__device__ __forceinline__ void qmma_sf_fp8_m16n8k32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    uint32_t sfa = 0x7F7F7F7F;   // UE8M0 bias=127 → 2^0 = 1.0 (neutral scale)
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

// Convert 2 FP32 → 2 packed FP8 E4M3 using CUDA intrinsics
// Result: uint16 with byte0=fp8(lo), byte1=fp8(hi)
__device__ __forceinline__ uint16_t cvt_2f32_to_e4m3x2(float lo, float hi) {
    float2 pair = make_float2(lo, hi);
    return __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);
}

// Convert 4 FP32 → 4 packed FP8 E4M3 in a uint32
// Result: byte0=fp8(v0), byte1=fp8(v1), byte2=fp8(v2), byte3=fp8(v3)
__device__ __forceinline__ uint32_t f32x4_to_e4m3x4(float v0, float v1, float v2, float v3) {
    uint16_t lo = cvt_2f32_to_e4m3x2(v0, v1);
    uint16_t hi = cvt_2f32_to_e4m3x2(v2, v3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
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

// V SMEM swizzle: the PV GEMM reads V transposed (fixed column, varying row),
// which on a plain row-major [row*128+col] layout collides every row in the
// same bank (catastrophic conflicts). XOR the column within each row by
// ((row&7)<<3): a multiple of 8, so the 16-byte (8-elem) cp.async store chunks
// stay contiguous & aligned, while 4 consecutive rows of one column now land in
// 4 distinct banks. Logical (row,col) is unchanged — MMA fragment layout intact.
__device__ __forceinline__ int v_swz(int row, int col) {
    return row * HEAD_DIM + (col ^ ((row & 7) << 3));
}

// Swizzled V store (same data placement as load_tile_async but XOR-permuted
// columns so PV reads are conflict-free). Stores 8 elems (16B) per cp.async.
__device__ void load_v_async_swz(
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
        // XOR mask is a multiple of 8, so the 8-wide chunk [col..col+7] maps to
        // [col^mask .. col^mask+7] contiguously and 16B-aligned.
        __nv_bfloat16 *dst = smem_tile + v_swz(row, col);
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

// Convert a generic shared-memory pointer to the 32-bit .shared address space
// offset that ldmatrix expects.
__device__ __forceinline__ uint32_t smem_u32(const void *smem_ptr) {
    uint32_t addr;
    asm volatile(
        "{\n"
        ".reg .u64 u;\n"
        "cvta.to.shared.u64 u, %1;\n"
        "cvt.u32.u64 %0, u;\n"
        "}\n" : "=r"(addr) : "l"(smem_ptr));
    return addr;
}

// ldmatrix.x4: each lane supplies the 16-byte-aligned row base address for one
// of the 32 rows across 4 8x8 b16 matrices. Loads a full 16x16 BF16 fragment
// (4 regs) matching the m16n8k16 A-operand thread distribution. `row_ptr` must
// already point at this lane's row (lane l -> matrix l/8, row l%8).
__device__ __forceinline__ void ldmatrix_x4(
    uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
    const void *row_ptr)
{
    uint32_t a = smem_u32(row_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// ldmatrix.x2: loads two 8x8 b16 matrices (2 regs) matching the m16n8k16
// B-operand thread distribution. Lanes 0-7 supply matrix 0 rows, 8-15 matrix 1.
__device__ __forceinline__ void ldmatrix_x2(
    uint32_t &r0, uint32_t &r1, const void *row_ptr)
{
    uint32_t a = smem_u32(row_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(a));
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

    const int sub2     = sub * 2;
    const int qbase0   = (wm + grp) * HEAD_DIM;
    const int qbase1   = (wm + grp + 8) * HEAD_DIM;
    const int grp_hdim = grp * HEAD_DIM;

    // ldmatrix lane mapping. For the m16n8k16 A operand (Q, 16x16, row-major)
    // each lane supplies the base of one of 32 rows across 4 8x8 matrices:
    //   matrix = lid/8, row-in-matrix = lid%8, k-half = (lid>=16)?8:0.
    // For the B operand (K, 8x16, row-major), x2 uses lanes 0-15:
    //   matrix = lid/8 selects the k-half, row-in-matrix = lid%8 selects K row.
    const int q_ld_row  = wm + ((lid & 8) ? 8 : 0) + (lid & 7);
    const int q_ld_koff = (lid & 16) ? 8 : 0;
    const int k_ld_lo   = lid & 7;            // K row within the nt group
    const int k_ld_koff = (lid & 8) ? 8 : 0;  // k-half for B

    const __nv_bfloat16 * __restrict__ sQ_a =
        static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sQ, 16));

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

        // (1) Load K[kb] → sK (single-buffer) and wait
        load_tile_async(sK, Kp, kvs, kvv, kvstride, tid, nthr);
        cp_async_commit();
        cp_async_wait(0);
        __syncthreads();

        // (2) Start V[kb] load → sV (overlaps with QK compute), swizzled layout
        __nv_bfloat16 *sVc = sV;
        load_v_async_swz(sVc, Vp, kvs, kvv, kvstride, tid, nthr);
        cp_async_commit();

        // (3) QK GEMM — interleaved B-fragment loads overlapping HMMA (pipe 30)
        __nv_bfloat16 *sKc = sK;
        const __nv_bfloat16 * __restrict__ sKc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sKc, 16));

        float Sr[8][4];
        #pragma unroll
        for (int i = 0; i < 8; i++)
            Sr[i][0] = Sr[i][1] = Sr[i][2] = Sr[i][3] = 0.f;

        // B (K) fragments via ldmatrix.x2 — one instruction per (nt,ks) replaces
        // two scalar lds_u32. Prefetch the first (nt=0, ks=0) fragment.
        uint32_t b0_next, b1_next;
        ldmatrix_x2(b0_next, b1_next, &sKc_a[k_ld_lo * HEAD_DIM + k_ld_koff]);

        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const int qoff = ks * MMA_K_BF16;

            // A (Q) fragment via ldmatrix.x4 — one instruction replaces 4 lds_u32.
            uint32_t a0, a1, a2, a3;
            ldmatrix_x4(a0, a1, a2, a3,
                        &sQ_a[q_ld_row * HEAD_DIM + qoff + q_ld_koff]);

            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                uint32_t b0 = b0_next;
                uint32_t b1 = b1_next;

                hmma_bf16_m16n8k16(
                    Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3],
                    a0, a1, a2, a3, b0, b1,
                    Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3]);

                if (nt < 7) {
                    const int kr = ((nt + 1) * MMA_N + k_ld_lo) * HEAD_DIM
                                   + qoff + k_ld_koff;
                    ldmatrix_x2(b0_next, b1_next, &sKc_a[kr]);
                } else if (ks < 7) {
                    const int kr = k_ld_lo * HEAD_DIM
                                   + (ks + 1) * MMA_K_BF16 + k_ld_koff;
                    ldmatrix_x2(b0_next, b1_next, &sKc_a[kr]);
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

        // (4b) Write P to SMEM as FP8 — reuse sK[stage] buffer (QK is done)
        // Layout: sP[row * BLK_N + col], row-major, FP8 E4M3
        // Each thread writes its P values at positions (grp, nt*8+sub*2) etc.
        uint8_t *sP = reinterpret_cast<uint8_t*>(sKc);
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int kv_col = nt * MMA_N + sub * 2;
            uint16_t pair_lo = cvt_2f32_to_e4m3x2(Sr[nt][0], Sr[nt][1]);
            uint16_t pair_hi = cvt_2f32_to_e4m3x2(Sr[nt][2], Sr[nt][3]);
            *reinterpret_cast<uint16_t*>(&sP[(wm + grp) * BLK_N + kv_col]) = pair_lo;
            *reinterpret_cast<uint16_t*>(&sP[(wm + grp + 8) * BLK_N + kv_col]) = pair_hi;
        }

        // (5) Wait for V[kb] (single-buffer: K[kb+1] loads at top of next iter,
        //     after PV consumes sP which aliases sK)
        cp_async_wait(0);
        __syncthreads();

        // (7) PV GEMM — QMMA.SF FP8 m16n8k32 on pipe 37 (INT_ARITH)
        //     2 ns-steps (K=32 each) covering BLK_N=64 KV tokens
        //     A-fragment: P from sP (FP8, row-major)
        //     B-fragment: V from sVc (BF16), converted to FP8 in registers
        //     The BF16→FP8 conversion uses FP_ARITH pipe 36 — overlaps with QMMA!
        const __nv_bfloat16 * __restrict__ sVc_a =
            static_cast<const __nv_bfloat16*>(__builtin_assume_aligned(sVc, 16));

        #pragma unroll
        for (int ns = 0; ns < 2; ns++) {
            // Load A-fragment from sP (SM120 a1↔a2 swap layout)
            // a0: P(row_lo, k=ns*32+sub*4..sub*4+3)
            // a1: P(row_hi, k=ns*32+sub*4..sub*4+3)   [SM120 swap]
            // a2: P(row_lo, k=ns*32+sub*4+16..sub*4+19)
            // a3: P(row_hi, k=ns*32+sub*4+16..sub*4+19)
            const int pa_base0 = (wm + grp) * BLK_N + ns * MMA_K_FP8;
            const int pa_base1 = (wm + grp + 8) * BLK_N + ns * MMA_K_FP8;
            uint32_t pa0 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub * 4]);
            uint32_t pa1 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub * 4]);
            uint32_t pa2 = *reinterpret_cast<const uint32_t*>(&sP[pa_base0 + sub * 4 + 16]);
            uint32_t pa3 = *reinterpret_cast<const uint32_t*>(&sP[pa_base1 + sub * 4 + 16]);

            // B-fragment rows: V[ns*32+sub*4+offset, col] for b0 (k_lo),
            //                  V[ns*32+sub*4+16+offset, col] for b1 (k_hi)
            const int vr_lo = ns * MMA_K_FP8 + sub * 4;
            const int vr_hi = vr_lo + 16;

            // Preload first V fragment (dt=0, col = 0*8+grp = grp). Reads use the
            // swizzled V layout (v_swz) — conflict-free transposed access.
            int vcol = grp;
            uint32_t vb0_next = f32x4_to_e4m3x4(
                __bfloat162float(sVc_a[v_swz(vr_lo,     vcol)]),
                __bfloat162float(sVc_a[v_swz(vr_lo + 1, vcol)]),
                __bfloat162float(sVc_a[v_swz(vr_lo + 2, vcol)]),
                __bfloat162float(sVc_a[v_swz(vr_lo + 3, vcol)]));
            uint32_t vb1_next = f32x4_to_e4m3x4(
                __bfloat162float(sVc_a[v_swz(vr_hi,     vcol)]),
                __bfloat162float(sVc_a[v_swz(vr_hi + 1, vcol)]),
                __bfloat162float(sVc_a[v_swz(vr_hi + 2, vcol)]),
                __bfloat162float(sVc_a[v_swz(vr_hi + 3, vcol)]));

            #pragma unroll
            for (int dt = 0; dt < 16; dt++) {
                uint32_t vb0 = vb0_next;
                uint32_t vb1 = vb1_next;

                // Issue QMMA.SF on pipe 37 (INT_ARITH) — non-blocking on pipes 6/36
                qmma_sf_fp8_m16n8k32(
                    Oa[dt][0], Oa[dt][1], Oa[dt][2], Oa[dt][3],
                    pa0, pa1, pa2, pa3, vb0, vb1,
                    Oa[dt][0], Oa[dt][1], Oa[dt][2], Oa[dt][3]);

                // Prefetch next V fragment — BF16→FP8 conversion on FP_ARITH (pipe 36)
                // SMEM loads on pipe 6. Both overlap with QMMA.SF on pipe 37!
                if (dt < 15) {
                    vcol = (dt + 1) * MMA_N + grp;
                    vb0_next = f32x4_to_e4m3x4(
                        __bfloat162float(sVc_a[v_swz(vr_lo,     vcol)]),
                        __bfloat162float(sVc_a[v_swz(vr_lo + 1, vcol)]),
                        __bfloat162float(sVc_a[v_swz(vr_lo + 2, vcol)]),
                        __bfloat162float(sVc_a[v_swz(vr_lo + 3, vcol)]));
                    vb1_next = f32x4_to_e4m3x4(
                        __bfloat162float(sVc_a[v_swz(vr_hi,     vcol)]),
                        __bfloat162float(sVc_a[v_swz(vr_hi + 1, vcol)]),
                        __bfloat162float(sVc_a[v_swz(vr_hi + 2, vcol)]),
                        __bfloat162float(sVc_a[v_swz(vr_hi + 3, vcol)]));
                }
            }
        }

        // (8) Ensure all warps done before next iteration reuses shared memory
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
