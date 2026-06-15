// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT
//
// SM120 (RTX PRO 6000, compute_120f) CUDA kernel reproducing the MiniMax-M3
// learned "lightning indexer" block-score output -- see
// python/fmha_sm100/indexer_ref.py (verified torch reference) and
// docs/M3_INDEXER_SPEC.md.
//
// Pipeline:
//   1. Project hidden states X[N,hidden] with q_proj[H*d,hidden] / k_proj[d,hidden].
//   2. Gemma RMSNorm per head over d (gain = 1 + w), computed in fp32.
//   3. Partial NeoX (split-half) RoPE on the first rotary_dim channels.
//   4. Score S[r,i,j] = scale * (q_idx[i,r] . k_idx[j,0]) with single shared key head.
//   5. Causal block max-pool over block_size key tiles -> max_score[H, nblk, N].
//
// PERF (HMMA tensor-core path, replacing the original one-thread-per-channel naive
// version):
//   project_norm_rope_hmma : X -> q_idx[N,H,d] / k_idx[N,d] via BF16 HMMA m16n8k16
//                            GEMM with cp.async SMEM tiling, then fused fp32
//                            RMSNorm + partial-NeoX RoPE in shared memory.
//   block_score_hmma       : q_idx,k_idx -> max_score[H,nblk,N] via BF16 HMMA
//                            m16n8k16 GEMM (Q tile @ K tile^T) with on-the-fly
//                            causal block max-pool.
//
// The fp32 input path (no tensor cores for bf16) falls back to a simple but
// correct scalar kernel; bf16/fp16 use HMMA.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>

namespace {

constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
// PTX helpers (mirrors sm120_fmha_fwd.cu)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16b(void* smem, const void* gmem) {
  uint32_t smem_addr;
  asm volatile(
      "{\n"
      ".reg .u64 u;\n"
      "cvta.to.shared.u64 u, %1;\n"
      "cvt.u32.u64 %0, u;\n"
      "}\n"
      : "=r"(smem_addr)
      : "l"(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gmem));
}
__device__ __forceinline__ void cp_async_zero_16b(void* smem) {
  *reinterpret_cast<uint4*>(smem) = make_uint4(0, 0, 0, 0);
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}
__device__ __forceinline__ void cp_async_wait1() {
  asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ uint32_t lds_u32(const void* smem_ptr) {
  uint32_t smem_addr, r;
  asm volatile(
      "{\n"
      ".reg .u64 u;\n"
      "cvta.to.shared.u64 u, %1;\n"
      "cvt.u32.u64 %2, u;\n"
      "ld.shared.b32 %0, [%2];\n"
      "}\n"
      : "=r"(r), "+l"(smem_ptr), "=r"(smem_addr));
  return r;
}

__device__ __forceinline__ void hmma_bf16_m16n8k16(float& d0, float& d1,
                                                    float& d2, float& d3,
                                                    uint32_t a0, uint32_t a1,
                                                    uint32_t a2, uint32_t a3,
                                                    uint32_t b0, uint32_t b1,
                                                    float c0, float c1, float c2,
                                                    float c3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3));
}

__device__ __forceinline__ void store_out(float* o, int c, float v) { o[c] = v; }
__device__ __forceinline__ void store_out(__nv_bfloat16* o, int c, float v) {
  o[c] = __float2bfloat16(v);
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}

// atomicMax for float on SMEM (monotone bit trick on ordered IEEE encoding).
// Values folded here are always finite scores (>= -inf init handled below).
__device__ __forceinline__ float atomicMaxFloat(float* addr, float val) {
  int* a = reinterpret_cast<int*>(addr);
  int old = *a, assumed;
  do {
    assumed = old;
    float cur = __int_as_float(assumed);
    if (cur >= val) break;
    old = atomicCAS(a, assumed, __float_as_int(val));
  } while (old != assumed);
  return __int_as_float(old);
}

// ===========================================================================
// PROJECTION + RMSNorm + RoPE  (HMMA BF16 path)
//
// One CTA computes a tile of [BLK_M tokens x d channels] = ONE output head for
// BLK_M tokens.  out[token, head, channel].  Grid = (ceil(N/BLK_M), H).
//   M = BLK_M (64) tokens, N = d (128) channels, K = hidden (6144).
//   A = X[token, k]            (row-major, [N, hidden])
//   B = W[head*d + n, k]       (row-major, [H*d, hidden])  -> col layout for HMMA
// After accumulation, the BLK_M x d tile lives across the 4 warps (each warp
// owns WARP_M=16 rows).  We then do Gemma RMSNorm + partial-NeoX RoPE per row.
// ===========================================================================
template <int BLK_M, int DHEAD, typename OutT>
__global__ void __launch_bounds__(128) project_norm_rope_hmma(
    const __nv_bfloat16* __restrict__ X,     // [N, hidden]
    const __nv_bfloat16* __restrict__ W,     // [H*d, hidden]
    const __nv_bfloat16* __restrict__ norm,  // [d]
    const int64_t* __restrict__ positions,   // [N]
    OutT* __restrict__ out,                  // [N, H, d]
    int N, int hidden, int H, int rotary_dim, float theta, float eps,
    bool apply_rope) {
  static_assert(BLK_M == 64, "BLK_M must be 64");
  static_assert(DHEAD == 128, "DHEAD must be 128");
  constexpr int NUM_WARPS = 4;
  constexpr int WARP_M = BLK_M / NUM_WARPS;  // 16
  constexpr int K_TILE = 32;                 // hidden tile per cp.async stage
  constexpr int NUM_STAGES = 2;

  const int m_blk = blockIdx.x;
  const int head = blockIdx.y;
  const int m0 = m_blk * BLK_M;

  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const int grp = lid / 4;  // 0..7
  const int sub = lid % 4;  // 0..3
  const int wm = wid * WARP_M;

  extern __shared__ __nv_bfloat16 smem_proj[];
  // sX: NUM_STAGES x [BLK_M x K_TILE], sW: NUM_STAGES x [DHEAD x K_TILE].
  // sAcc (BLK_M x DHEAD fp32) ALIASES the start of smem: the GEMM tiles are dead
  // by the time RMSNorm/RoPE runs, so total smem = max(tiles, sAcc) not the sum.
  __nv_bfloat16* sX = smem_proj;
  __nv_bfloat16* sW = sX + NUM_STAGES * BLK_M * K_TILE;
  float* sAcc = reinterpret_cast<float*>(smem_proj);

  const __nv_bfloat16* Xp = X + (int64_t)m0 * hidden;
  const __nv_bfloat16* Wp = W + (int64_t)head * DHEAD * hidden;

  // Accumulators: each warp owns 16 rows (grp, grp+8) x (16 n-tiles of 8).
  // Acc[nt][0..3]: rows {grp, grp+8} x cols {nt*8+sub*2, +1}
  constexpr int NT = DHEAD / 8;  // 16
  float Acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++)
    Acc[i][0] = Acc[i][1] = Acc[i][2] = Acc[i][3] = 0.f;

  const int nktiles = (hidden + K_TILE - 1) / K_TILE;

  // cp.async loader: copy [ROWS x K_TILE] bf16 into smem (8 bf16 per cp.async).
  // 128 threads, each does 8 elems => 1024 elems/iter.
  auto load_tile = [&](__nv_bfloat16* dst, const __nv_bfloat16* src_base,
                       int row_count, int row_stride, int kt) {
    const int total = row_count * K_TILE;  // e.g. 64*32=2048 or 128*32=4096
    const int kbase = kt * K_TILE;
    for (int e = tid * 8; e < total; e += 128 * 8) {
      int r = e / K_TILE;
      int c = e % K_TILE;
      __nv_bfloat16* d = dst + r * K_TILE + c;
      int gk = kbase + c;
      // K_TILE divides hidden (6144 % 32 == 0) so no k bounds check needed.
      const __nv_bfloat16* s = src_base + (int64_t)r * row_stride + gk;
      cp_async_16b(d, s);
    }
  };

  auto load_tile_x = [&](__nv_bfloat16* dst, int kt) {
    // X rows may exceed N at the tail -> zero-fill.
    const int kbase = kt * K_TILE;
    for (int e = tid * 8; e < BLK_M * K_TILE; e += 128 * 8) {
      int r = e / K_TILE;
      int c = e % K_TILE;
      __nv_bfloat16* d = dst + r * K_TILE + c;
      if (m0 + r < N) {
        const __nv_bfloat16* s = Xp + (int64_t)r * hidden + kbase + c;
        cp_async_16b(d, s);
      } else {
        cp_async_zero_16b(d);
      }
    }
  };

  // Prologue: stage 0.
  load_tile_x(sX, 0);
  load_tile(sW, Wp, DHEAD, hidden, 0);
  cp_async_commit();

  for (int kt = 0; kt < nktiles; kt++) {
    int stage = kt & 1;
    int nstage = (kt + 1) & 1;
    // prefetch next
    if (kt + 1 < nktiles) {
      load_tile_x(sX + nstage * BLK_M * K_TILE, kt + 1);
      load_tile(sW + nstage * DHEAD * K_TILE, Wp, DHEAD, hidden, kt + 1);
      cp_async_commit();
      cp_async_wait1();
    } else {
      cp_async_wait_all();
    }
    __syncthreads();

    __nv_bfloat16* sXc = sX + stage * BLK_M * K_TILE;
    __nv_bfloat16* sWc = sW + stage * DHEAD * K_TILE;

    // K_TILE=32 -> 2 k-substeps of 16.
#pragma unroll
    for (int kk = 0; kk < K_TILE / 16; kk++) {
      const int koff = kk * 16;
      // A-fragment: X rows {wm+grp, wm+grp+8}, k = koff + {sub*2..+1, +8..}
      uint32_t a0 = lds_u32(&sXc[(wm + grp) * K_TILE + koff + sub * 2]);
      uint32_t a1 = lds_u32(&sXc[(wm + grp + 8) * K_TILE + koff + sub * 2]);
      uint32_t a2 = lds_u32(&sXc[(wm + grp) * K_TILE + koff + sub * 2 + 8]);
      uint32_t a3 = lds_u32(&sXc[(wm + grp + 8) * K_TILE + koff + sub * 2 + 8]);
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        // B-fragment: W rows (n = nt*8 + grp), k = koff + {sub*2.., +8..}
        const int wrow = nt * 8 + grp;
        uint32_t b0 = lds_u32(&sWc[wrow * K_TILE + koff + sub * 2]);
        uint32_t b1 = lds_u32(&sWc[wrow * K_TILE + koff + sub * 2 + 8]);
        hmma_bf16_m16n8k16(Acc[nt][0], Acc[nt][1], Acc[nt][2], Acc[nt][3], a0,
                           a1, a2, a3, b0, b1, Acc[nt][0], Acc[nt][1],
                           Acc[nt][2], Acc[nt][3]);
      }
    }
    __syncthreads();
  }

  // ---- Store raw accumulators to sAcc[row(0..63)][col(0..127)] ----
  // Row r0 = wm+grp, r1 = wm+grp+8; cols {nt*8+sub*2, +1}.
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    int c0 = nt * 8 + sub * 2;
    int r0 = wm + grp;
    int r1 = wm + grp + 8;
    sAcc[r0 * DHEAD + c0] = Acc[nt][0];
    sAcc[r0 * DHEAD + c0 + 1] = Acc[nt][1];
    sAcc[r1 * DHEAD + c0] = Acc[nt][2];
    sAcc[r1 * DHEAD + c0 + 1] = Acc[nt][3];
  }
  __syncthreads();

  // ---- RMSNorm + RoPE per row, 1 warp-group handles rows; use all 128 threads.
  // Assign one row per (warp,grp) pair? Simpler: each thread handles a subset of
  // (row) doing the full 128-dim norm. There are BLK_M=64 rows; with 128 threads
  // use the natural row = wm + grp (each (warp,grp) maps to 2 rows r and r+8 are
  // owned by lanes with same grp but different sub). To avoid redundant work, let
  // exactly one thread per row do the reduction over sAcc.
  // Use threads 0..63 => one row each.
  const int half = rotary_dim / 2;
  if (tid < BLK_M) {
    int r = tid;
    if (m0 + r < N) {
      const float* row = sAcc + r * DHEAD;
      float ss = 0.f;
#pragma unroll
      for (int c = 0; c < DHEAD; c++) ss += row[c] * row[c];
      float inv_rms = rsqrtf(ss / (float)DHEAD + eps);

      long pos = positions ? positions[m0 + r] : 0;
      float normed[DHEAD];
#pragma unroll
      for (int c = 0; c < DHEAD; c++)
        normed[c] = row[c] * inv_rms * (1.0f + __bfloat162float(norm[c]));

      OutT* o = out + ((int64_t)(m0 + r) * H + head) * DHEAD;
      if (apply_rope) {
#pragma unroll
        for (int c = 0; c < DHEAD; c++) {
          if (c < rotary_dim) {
            int fi = (c < half) ? c : (c - half);
            float ang = (float)pos *
                        (1.0f / powf(theta, (float)fi / (float)half));
            float cosv = cosf(ang), sinv = sinf(ang);
            float x1 = normed[fi];
            float x2 = normed[fi + half];
            store_out(o, c, (c < half) ? (x1 * cosv - x2 * sinv)
                                       : (x2 * cosv + x1 * sinv));
          } else {
            store_out(o, c, normed[c]);
          }
        }
      } else {
#pragma unroll
        for (int c = 0; c < DHEAD; c++) store_out(o, c, normed[c]);
      }
    }
  }
}

// ===========================================================================
// BLOCK SCORE  (HMMA BF16 path)
//
// max_score[h, b, i] = max_{j in block b, j<=i} scale * (q_idx[i,h] . k_idx[j]).
// One CTA per (head h, query-tile of BLK_M=64 queries).  For each key tile of
// BLK_N=64 keys we compute S[64q x 64k] = Q_tile @ K_tile^T via HMMA, apply
// causal mask + scale, and fold into the per-(query, kv-block) running max.
//
// q_idx/k_idx are fp32 [N,H,d] / [N,d].  We convert to bf16 in SMEM for HMMA.
// (the dot is over d=128; tiny rounding, well within the bf16 rms budget since
// the inputs were already produced from a bf16 projection.)
// ===========================================================================
template <int BLK_M, int BLK_N, int DHEAD>
__global__ void __launch_bounds__(128) block_score_hmma(
    const __nv_bfloat16* __restrict__ q_idx,  // [N, H, d]  (bf16)
    const __nv_bfloat16* __restrict__ k_idx,  // [N, d]     (bf16)
    const int64_t* __restrict__ positions,  // [N]
    float* __restrict__ max_score,          // [H, nblk, N]
    int N, int H, int block_size, int nblk, float scale, bool causal) {
  static_assert(BLK_M == 64 && BLK_N == 64 && DHEAD == 128, "tile shape");
  constexpr int NUM_WARPS = 4;
  constexpr int WARP_M = BLK_M / NUM_WARPS;  // 16
  // block_size == BLK_N == 64? No: indexer block_size=128. We process key tiles
  // of BLK_N=64 and there are 2 key-tiles per 128-token sparse block; the max
  // over the two is taken in the running max.

  const int h = blockIdx.x;
  const int qtile = blockIdx.y;
  const int q0 = qtile * BLK_M;

  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const int grp = lid / 4;
  const int sub = lid % 4;
  const int wm = wid * WARP_M;

  extern __shared__ __nv_bfloat16 smem_sc[];
  // sQ: [BLK_M x DHEAD] bf16, sK: single [BLK_N x DHEAD] bf16 buffer.
  constexpr int NUM_STAGES = 1;
  __nv_bfloat16* sQ = smem_sc;
  __nv_bfloat16* sK = sQ + BLK_M * DHEAD;

  // positions for this query tile (fp registers / smem). Load once.
  __shared__ long qpos[BLK_M];
  if (tid < BLK_M) qpos[tid] = (q0 + tid < N) ? positions[q0 + tid] : (long)2e18;
  __syncthreads();

  // Load Q tile [BLK_M x DHEAD] (bf16, direct copy).
  for (int e = tid; e < BLK_M * DHEAD; e += 128) {
    int r = e / DHEAD, c = e % DHEAD;
    __nv_bfloat16 v = __float2bfloat16(0.f);
    if (q0 + r < N) v = q_idx[((int64_t)(q0 + r) * H + h) * DHEAD + c];
    sQ[r * DHEAD + c] = v;
  }

  // Running max per query row owned by this thread: rows {wm+grp, wm+grp+8}.
  // We need one running max per (this query row, sparse-block b). Each thread,
  // after the HMMA, holds S for its 2 query-rows over the 64 keys of the tile.
  // The 16 n-tiles (key cols) it covers: cols {nt*8 + sub*2, +1}. We reduce the
  // key axis into the running per-block max held across threads -> store to a
  // shared [BLK_M x nblk] running max, updated each key tile.
  // Simpler & robust: keep running max in shared mem [BLK_M * nblk] in registers
  // is impossible (nblk dynamic). Use a global running max via atomicMax? Avoid.
  // Instead: loop key tiles, and maintain shared smax[BLK_M][nblk] in SMEM.
  // nblk can be large; but only blocks overlapping [0, q0+BLK_M) matter and are
  // <= qtile+1 (causal). Allocate dynamic smax of size BLK_M*nblk_local.

  // Number of sparse blocks that can be non -inf for this query tile.
  const int max_b = causal ? min(nblk, (q0 + BLK_M - 1) / block_size + 1) : nblk;

  // smax lives after sQ/sK in dynamic smem (fp32). Size BLK_M * max_b.
  float* smax = reinterpret_cast<float*>(sK + NUM_STAGES * BLK_N * DHEAD);
  for (int e = tid; e < BLK_M * max_b; e += 128) smax[e] = -INFINITY;

  // Only key tiles with any j <= max query pos matter; for causal, key tiles
  // beyond the query tile are fully masked. Last relevant key row = q0+BLK_M-1.
  const int last_key = causal ? min(N, q0 + BLK_M) : N;
  const int n_ktiles = (last_key + BLK_N - 1) / BLK_N;

  __syncthreads();

  // bf16 Q SMEM ready.
  __nv_bfloat16* sQa = sQ;

  // Simple loader for K (bf16, direct copy) per tile.
  auto load_k = [&](__nv_bfloat16* dst, int k0) {
    for (int e = tid; e < BLK_N * DHEAD; e += 128) {
      int r = e / DHEAD, c = e % DHEAD;
      __nv_bfloat16 v = __float2bfloat16(0.f);
      if (k0 + r < N) v = k_idx[(int64_t)(k0 + r) * DHEAD + c];
      dst[r * DHEAD + c] = v;
    }
  };

  constexpr int NT = BLK_N / 8;  // 8 key n-tiles of 8
  constexpr int KK = DHEAD / 16; // 8 k-steps of 16

  for (int kt = 0; kt < n_ktiles; kt++) {
    int k0 = kt * BLK_N;
    __nv_bfloat16* sKc = sK;  // single buffer; loop-end sync protects reuse.
    load_k(sKc, k0);
    __syncthreads();

    // S[16q x 64k] for this warp: 8 n-tiles, acc 4 each.
    float Sr[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++)
      Sr[i][0] = Sr[i][1] = Sr[i][2] = Sr[i][3] = 0.f;

#pragma unroll
    for (int kk = 0; kk < KK; kk++) {
      const int koff = kk * 16;
      uint32_t a0 = lds_u32(&sQa[(wm + grp) * DHEAD + koff + sub * 2]);
      uint32_t a1 = lds_u32(&sQa[(wm + grp + 8) * DHEAD + koff + sub * 2]);
      uint32_t a2 = lds_u32(&sQa[(wm + grp) * DHEAD + koff + sub * 2 + 8]);
      uint32_t a3 = lds_u32(&sQa[(wm + grp + 8) * DHEAD + koff + sub * 2 + 8]);
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        const int krow = nt * 8 + grp;  // key index within tile (n dim)
        uint32_t b0 = lds_u32(&sKc[krow * DHEAD + koff + sub * 2]);
        uint32_t b1 = lds_u32(&sKc[krow * DHEAD + koff + sub * 2 + 8]);
        hmma_bf16_m16n8k16(Sr[nt][0], Sr[nt][1], Sr[nt][2], Sr[nt][3], a0, a1,
                           a2, a3, b0, b1, Sr[nt][0], Sr[nt][1], Sr[nt][2],
                           Sr[nt][3]);
      }
    }

    // Each thread holds S for query rows {wm+grp, wm+grp+8} and
    // key cols {nt*8 + sub*2, +1} (nt 0..7) within this key tile.
    // Apply scale + causal mask, fold into smax[query_row][block].
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
      int kc0 = nt * 8 + sub * 2;  // key col within tile
      int kc1 = kc0 + 1;
      int jg0 = k0 + kc0;          // global key index
      int jg1 = k0 + kc1;
      // query rows
      int qr0 = wm + grp;
      int qr1 = wm + grp + 8;
      long pi0 = qpos[qr0];
      long pi1 = qpos[qr1];
      long pj0 = (jg0 < N) ? positions[jg0] : (long)-2e18;
      long pj1 = (jg1 < N) ? positions[jg1] : (long)-2e18;

      float v00 = Sr[nt][0] * scale;  // (qr0, kc0)
      float v01 = Sr[nt][1] * scale;  // (qr0, kc1)
      float v10 = Sr[nt][2] * scale;  // (qr1, kc0)
      float v11 = Sr[nt][3] * scale;  // (qr1, kc1)

      bool valid_q0 = (q0 + qr0 < N);
      bool valid_q1 = (q0 + qr1 < N);
      bool keep00 = valid_q0 && jg0 < N && (!causal || pj0 <= pi0);
      bool keep01 = valid_q0 && jg1 < N && (!causal || pj1 <= pi0);
      bool keep10 = valid_q1 && jg0 < N && (!causal || pj0 <= pi1);
      bool keep11 = valid_q1 && jg1 < N && (!causal || pj1 <= pi1);

      int b0 = jg0 / block_size;
      int b1 = jg1 / block_size;
      if (keep00 && b0 < max_b) atomicMaxFloat(&smax[qr0 * max_b + b0], v00);
      if (keep01 && b1 < max_b) atomicMaxFloat(&smax[qr0 * max_b + b1], v01);
      if (keep10 && b0 < max_b) atomicMaxFloat(&smax[qr1 * max_b + b0], v10);
      if (keep11 && b1 < max_b) atomicMaxFloat(&smax[qr1 * max_b + b1], v11);
    }
    __syncthreads();
  }

  // Write smax -> max_score[h, b, q].  max_score initialized to -inf already.
  for (int e = tid; e < BLK_M * max_b; e += 128) {
    int r = e / max_b, b = e % max_b;
    if (q0 + r < N) {
      float v = smax[r * max_b + b];
      max_score[((int64_t)h * nblk + b) * N + (q0 + r)] = v;
    }
  }
}

// ---------------------------------------------------------------------------
// Scalar fallback (fp32 inputs / generic): one block per (row,head).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void project_norm_rope_scalar(
    const scalar_t* __restrict__ X, const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ norm, const int64_t* __restrict__ positions,
    float* __restrict__ out, int N, int hidden, int H, int d, int rotary_dim,
    float theta, float eps, bool apply_rope) {
  const int row_head = blockIdx.x;
  const int n = row_head / H;
  const int h = row_head % H;
  const int c = threadIdx.x;
  if (n >= N || c >= d) return;
  const scalar_t* xrow = X + (int64_t)n * hidden;
  const scalar_t* wrow = W + ((int64_t)h * d + c) * hidden;
  float acc = 0.f;
  for (int k = 0; k < hidden; ++k)
    acc += (float)xrow[k] * (float)wrow[k];
  extern __shared__ float smem[];
  smem[c] = acc * acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (c < stride) smem[c] += smem[c + stride];
    __syncthreads();
  }
  float ssum = smem[0];
  __syncthreads();
  float inv_rms = rsqrtf(ssum / (float)d + eps);
  float normed = acc * inv_rms * (1.0f + (float)norm[c]);
  smem[c] = normed;
  __syncthreads();
  float result = normed;
  if (apply_rope && c < rotary_dim) {
    const int half = rotary_dim / 2;
    long pos = positions[n];
    int fi = (c < half) ? c : (c - half);
    float ang = (float)pos * (1.0f / powf(theta, (float)fi / (float)half));
    float cosv = cosf(ang), sinv = sinf(ang);
    float x1 = smem[fi], x2 = smem[fi + half];
    result = (c < half) ? (x1 * cosv - x2 * sinv) : (x2 * cosv + x1 * sinv);
  }
  out[((int64_t)n * H + h) * d + c] = result;
}

__global__ void block_score_scalar(const float* __restrict__ q_idx,
                                    const float* __restrict__ k_idx,
                                    const int64_t* __restrict__ positions,
                                    float* __restrict__ max_score, int N, int H,
                                    int d, int block_size, int nblk, float scale,
                                    bool causal) {
  const int h = blockIdx.x;
  const int i = blockIdx.y;
  if (h >= H || i >= N) return;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int lane = tid % kWarpSize;
  const int warp = tid / kWarpSize;
  const int nwarps = nthreads / kWarpSize;
  extern __shared__ float qsh[];
  for (int c = tid; c < d; c += nthreads)
    qsh[c] = q_idx[((int64_t)i * H + h) * d + c];
  __syncthreads();
  long pos_i = causal ? positions[i] : 0;
  float* wred = qsh + d;
  for (int b = 0; b < nblk; ++b) {
    int jstart = b * block_size;
    int jend = min(jstart + block_size, N);
    float local_max = -INFINITY;
    for (int j = jstart + tid; j < jend; j += nthreads) {
      bool keep = !causal || (positions[j] <= pos_i);
      if (keep) {
        const float* krow = k_idx + (int64_t)j * d;
        float dot = 0.f;
#pragma unroll 4
        for (int c = 0; c < d; ++c) dot += qsh[c] * krow[c];
        local_max = fmaxf(local_max, dot * scale);
      }
    }
    float wmax = warp_reduce_max(local_max);
    if (lane == 0) wred[warp] = wmax;
    __syncthreads();
    if (tid == 0) {
      float bmax = -INFINITY;
      for (int w = 0; w < nwarps; ++w) bmax = fmaxf(bmax, wred[w]);
      max_score[((int64_t)h * nblk + b) * N + i] = bmax;
    }
    __syncthreads();
  }
}

}  // namespace

// Full pipeline: hidden states -> max_score[H, nblk, N].
torch::Tensor sm120_indexer_block_scores(
    torch::Tensor q, torch::Tensor k, torch::Tensor q_proj, torch::Tensor k_proj,
    torch::Tensor q_norm, torch::Tensor k_norm, torch::Tensor positions,
    int64_t block_size, int64_t n_heads, int64_t head_dim, double scale,
    int64_t rotary_dim, double rope_theta, double eps, bool causal,
    bool apply_rope) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(q.dim() == 2, "q must be [N, hidden]");
  const int N = q.size(0);
  const int hidden = q.size(1);
  const int H = n_heads;
  const int d = head_dim;
  TORCH_CHECK(q_proj.size(0) == H * d && q_proj.size(1) == hidden, "q_proj shape");
  TORCH_CHECK(k_proj.size(0) == d && k_proj.size(1) == hidden, "k_proj shape");
  TORCH_CHECK(positions.dtype() == torch::kInt64, "positions must be int64");
  TORCH_CHECK(d == 128, "this kernel requires head_dim == 128");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());

  q = q.contiguous();
  k = k.contiguous();
  q_proj = q_proj.contiguous();
  k_proj = k_proj.contiguous();
  q_norm = q_norm.contiguous();
  k_norm = k_norm.contiguous();
  positions = positions.contiguous();

  const int nblk = (N + block_size - 1) / block_size;
  auto max_score = torch::full({H, nblk, N},
                               -std::numeric_limits<float>::infinity(), fopt);

  const bool is_bf16 = (q.scalar_type() == torch::kBFloat16);

  if (is_bf16) {
    // Index buffers kept in bf16 (consumed directly by the HMMA score kernel).
    auto bopt = torch::TensorOptions().dtype(torch::kBFloat16).device(q.device());
    auto q_idx = torch::empty({N, H, d}, bopt);
    auto k_idx = torch::empty({N, 1, d}, bopt);
    auto qidx_ptr = reinterpret_cast<__nv_bfloat16*>(q_idx.data_ptr());
    auto kidx_ptr = reinterpret_cast<__nv_bfloat16*>(k_idx.data_ptr());

    // ---- HMMA projection ----
    constexpr int BLK_M = 64, DHEAD = 128, K_TILE = 32, NUM_STAGES = 2;
    size_t tile_bytes =
        (size_t)(NUM_STAGES * BLK_M * K_TILE + NUM_STAGES * DHEAD * K_TILE) *
        sizeof(__nv_bfloat16);
    size_t acc_bytes = (size_t)BLK_M * DHEAD * sizeof(float);
    size_t shmem_proj = tile_bytes > acc_bytes ? tile_bytes : acc_bytes;
    int m_blocks = (N + BLK_M - 1) / BLK_M;
    cudaFuncSetAttribute(project_norm_rope_hmma<BLK_M, DHEAD, __nv_bfloat16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)shmem_proj);
    {
      dim3 grid(m_blocks, H);
      project_norm_rope_hmma<BLK_M, DHEAD, __nv_bfloat16>
          <<<grid, 128, shmem_proj, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(q_proj.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(q_norm.data_ptr()),
              positions.data_ptr<int64_t>(), qidx_ptr, N, hidden, H, rotary_dim,
              (float)rope_theta, (float)eps, apply_rope);
    }
    {
      dim3 grid(m_blocks, 1);
      project_norm_rope_hmma<BLK_M, DHEAD, __nv_bfloat16>
          <<<grid, 128, shmem_proj, stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(k_proj.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(k_norm.data_ptr()),
              positions.data_ptr<int64_t>(), kidx_ptr, N, hidden, 1, rotary_dim,
              (float)rope_theta, (float)eps, apply_rope);
    }

    // ---- HMMA scoring ----
    constexpr int SBLK_M = 64, SBLK_N = 64;
    int q_tiles = (N + SBLK_M - 1) / SBLK_M;
    // smem: Q tile + 1*K tile + smax(BLK_M * max_b). max_b <= nblk.
    size_t smem_sc = SBLK_M * DHEAD * sizeof(__nv_bfloat16) +
                     1 * SBLK_N * DHEAD * sizeof(__nv_bfloat16) +
                     SBLK_M * nblk * sizeof(float);
    cudaFuncSetAttribute(block_score_hmma<SBLK_M, SBLK_N, DHEAD>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_sc);
    dim3 sgrid(H, q_tiles);
    block_score_hmma<SBLK_M, SBLK_N, DHEAD><<<sgrid, 128, smem_sc, stream>>>(
        qidx_ptr, kidx_ptr, positions.data_ptr<int64_t>(),
        max_score.data_ptr<float>(), N, H, block_size, nblk, (float)scale,
        causal);
  } else {
    // ---- scalar fallback (fp32 / fp16) ----
    auto q_idx = torch::empty({N, H, d}, fopt);
    auto k_idx = torch::empty({N, 1, d}, fopt);
    AT_DISPATCH_SWITCH(
        q.scalar_type(), "sm120_indexer_project_scalar",
        AT_DISPATCH_CASE(torch::kFloat, [&] {
          size_t shmem = d * sizeof(float);
          project_norm_rope_scalar<scalar_t><<<dim3(N * H), dim3(d), shmem,
                                                stream>>>(
              q.data_ptr<scalar_t>(), q_proj.data_ptr<scalar_t>(),
              q_norm.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
              q_idx.data_ptr<float>(), N, hidden, H, d, rotary_dim,
              (float)rope_theta, (float)eps, apply_rope);
          project_norm_rope_scalar<scalar_t><<<dim3(N), dim3(d), shmem,
                                               stream>>>(
              k.data_ptr<scalar_t>(), k_proj.data_ptr<scalar_t>(),
              k_norm.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
              k_idx.data_ptr<float>(), N, hidden, 1, d, rotary_dim,
              (float)rope_theta, (float)eps, apply_rope);
        })
        AT_DISPATCH_CASE(torch::kHalf, [&] {
          size_t shmem = d * sizeof(float);
          project_norm_rope_scalar<scalar_t><<<dim3(N * H), dim3(d), shmem,
                                                stream>>>(
              q.data_ptr<scalar_t>(), q_proj.data_ptr<scalar_t>(),
              q_norm.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
              q_idx.data_ptr<float>(), N, hidden, H, d, rotary_dim,
              (float)rope_theta, (float)eps, apply_rope);
          project_norm_rope_scalar<scalar_t><<<dim3(N), dim3(d), shmem,
                                               stream>>>(
              k.data_ptr<scalar_t>(), k_proj.data_ptr<scalar_t>(),
              k_norm.data_ptr<scalar_t>(), positions.data_ptr<int64_t>(),
              k_idx.data_ptr<float>(), N, hidden, 1, d, rotary_dim,
              (float)rope_theta, (float)eps, apply_rope);
        }));

    const int score_threads = 128;
    const int nwarps = score_threads / kWarpSize;
    dim3 sgrid(H, N);
    size_t sshmem = (d + nwarps) * sizeof(float);
    block_score_scalar<<<sgrid, dim3(score_threads), sshmem, stream>>>(
        q_idx.data_ptr<float>(), k_idx.data_ptr<float>(),
        positions.data_ptr<int64_t>(), max_score.data_ptr<float>(), N, H, d,
        block_size, nblk, (float)scale, causal);
  }

  C10_CUDA_CHECK(cudaGetLastError());
  return max_score;  // [H, nblk, N]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_scores", &sm120_indexer_block_scores,
        "M3 indexer block scores (SM120), full pipeline from hidden states");
}
