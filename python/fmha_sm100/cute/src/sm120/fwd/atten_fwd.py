# SPDX-FileCopyrightText: Copyright (c) 2026 turbollama contributors
# SPDX-License-Identifier: MIT

"""SM120 dense FlashAttention forward kernel.

Architecture: per-warp HMMA BF16 (m16n8k16) with cp.async pipeline.
This is the SM120 equivalent of SM100's tcgen05-based sparse attention.

Design decisions:
  - Accumulator in registers (not TMEM — unavailable on SM120)
  - cp.async (LDGSTS) for GMEM→SMEM (not TMA — mbarrier TX broken on SM120)
  - mma.sync.aligned (HMMA) per-warp (not tcgen05.mma)
  - 2-stage async pipeline (load next tile while computing current)
  - Online softmax in registers (standard FA2 approach)
  - Tile: M=64, N=64, D=128 (fits in register file)
  - 4 warps per CTA, each warp processes M/4=16 Q rows

Verified SM120 HW capabilities (2026-06-12):
  - HMMA BF16 m16n8k16: standard SM80+ path
  - cp.async.ca.shared.global: ✅ works
  - cp.async.commit_group / wait_group: ✅ works
  - mbarrier basic (init/arrive/try_wait): ✅ works
  - QMMA.SF block-scaled: ✅ works (Phase 2)
  - LDSM packed FP6/FP4: ✅ works (Phase 2)
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

from src.common.cute_dsl_utils import assume_tensor_aligned


# ===========================================================================
# SM120 FlashAttention Configuration
# ===========================================================================

class SM120AttentionConfig:
    """Tile sizes and pipeline config for SM120 FA kernel."""

    head_dim: int = 128
    blk_m: int = 64          # Q tile rows
    blk_n: int = 64          # KV tile cols (seqlen K dimension)
    blk_k: int = 16          # MMA K dimension (BF16 m16n8k16)

    num_warps: int = 4       # warps per CTA
    num_stages: int = 2      # async pipeline depth

    # Per-warp tile: each warp handles blk_m/num_warps = 16 Q rows
    warp_m: int = 16         # matches HMMA M=16

    # Register budget per warp for accumulator:
    # M=16, D=128 in F32 = 16*128/32 = 64 registers for O accumulator
    # Plus LSE (16 floats = 16 regs), rowmax (16 regs)
    # Total ~96 regs for state, leaves ~160 regs for operands+temporaries

    # SMEM layout:
    # Q tile: blk_m × head_dim × 2B = 64×128×2 = 16 KB
    # K tile: blk_n × head_dim × 2B = 64×128×2 = 16 KB (×2 stages = 32 KB)
    # V tile: blk_n × head_dim × 2B = 64×128×2 = 16 KB (×2 stages = 32 KB)
    # Total: ~80 KB SMEM (within SM120's 228 KB limit)

    smem_q_bytes: int = 64 * 128 * 2         # 16 KB
    smem_k_bytes: int = 64 * 128 * 2         # 16 KB per stage
    smem_v_bytes: int = 64 * 128 * 2         # 16 KB per stage


# ===========================================================================
# Inline PTX helpers for SM120
# ===========================================================================

@dsl_user_op
def cp_async_16b(smem_ptr, gmem_ptr, *, loc=None, ip=None):
    """cp.async.ca.shared.global 16 bytes."""
    llvm.inline_asm(
        T.i32(),
        [smem_ptr.toint().ir_value(loc=loc, ip=ip),
         gmem_ptr.toint().ir_value(loc=loc, ip=ip)],
        "{\n"
        ".reg .u32 sa;\n"
        "cvt.u32.u64 sa, $1;\n"
        "cp.async.ca.shared.global [sa], [$2], 16;\n"
        "mov.u32 $0, 0;\n"
        "}\n",
        "=r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    """cp.async.commit_group."""
    llvm.inline_asm(
        None, [],
        "cp.async.commit_group;\n", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def cp_async_wait(n: int = 0, *, loc=None, ip=None):
    """cp.async.wait_group N."""
    llvm.inline_asm(
        None, [],
        f"cp.async.wait_group {n};\n", "",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


# ===========================================================================
# SM120 Dense FlashAttention Forward
# ===========================================================================

class DenseAttentionForwardSm120:
    """SM120 dense FlashAttention forward (BF16, per-warp HMMA).

    Implements standard FlashAttention-2 algorithm:
      for each K/V block:
        S = Q @ K^T                    (QK GEMM)
        m_new = max(m_old, rowmax(S))  (online max)
        P = exp2(S - m_new)            (softmax numerator)
        l_new = exp2(m_old - m_new) * l_old + rowsum(P)  (running sum)
        O = exp2(m_old - m_new) * O + P @ V    (PV GEMM + rescale)
      O = O / l_new                    (normalize)
    """

    def __init__(
        self,
        head_dim: int = 128,
        num_heads_q: int = 32,
        num_heads_kv: int = 8,
        causal: bool = False,
        dtype=cutlass.BFloat16,
    ):
        self.cfg = SM120AttentionConfig()
        if head_dim != 128:
            raise NotImplementedError("SM120 FA currently supports only D=128")
        self.head_dim = head_dim
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.qheadperkv = num_heads_q // num_heads_kv
        self.causal = causal
        self.dtype = dtype

        # MMA operation: BF16 m16n8k16 with F32 accumulator
        self.mma_m = 16
        self.mma_n = 8
        self.mma_k = 16

    def get_smem_bytes(self) -> int:
        """Total shared memory needed."""
        cfg = self.cfg
        return (cfg.smem_q_bytes +
                cfg.smem_k_bytes * cfg.num_stages +
                cfg.smem_v_bytes * cfg.num_stages)

    def get_grid_dim(self, seq_len_q: int, batch_size: int) -> tuple:
        """Compute grid dimensions."""
        num_m_blocks = (seq_len_q + self.cfg.blk_m - 1) // self.cfg.blk_m
        return (num_m_blocks, batch_size * self.num_heads_q, 1)

    def get_block_dim(self) -> tuple:
        """Threads per CTA."""
        return (self.cfg.num_warps * 32, 1, 1)

    def compile_kernel(self):
        """Compile the SM120 FA kernel via CuTe-DSL.

        The kernel follows standard FA2 structure:
        1. Load Q tile to SMEM (once)
        2. For each K/V block:
           a. cp.async load K to SMEM (pipelined)
           b. GEMM: S = Q @ K^T via mma.sync BF16
           c. Online softmax: m, l, P update
           d. cp.async load V to SMEM (pipelined)
           e. GEMM: O += P @ V via mma.sync BF16
           f. Rescale O by exp2(m_old - m_new)
        3. Normalize: O = O / l
        4. Store O to GMEM
        """
        cfg = self.cfg

        @cute.kernel
        def sm120_fmha_fwd(
            Q: cute.Tensor,        # [total_q, Hq, D] BF16
            K: cute.Tensor,        # [total_kv, Hkv, D] BF16
            V: cute.Tensor,        # [total_kv, Hkv, D] BF16
            O: cute.Tensor,        # [total_q, Hq, D] BF16 output
            LSE: cute.Tensor,      # [total_q, Hq] F32 log-sum-exp
            seq_len_q: Int32,
            seq_len_k: Int32,
            scale: Float32,        # 1/sqrt(D)
        ):
            # Block/thread indexing
            block_m_idx = cute.arch.block_idx_x()
            head_idx = cute.arch.block_idx_y()
            warp_idx = cute.arch.warp_idx()
            lane_idx = cute.arch.lane_idx()

            kv_head_idx = head_idx // Int32(self.qheadperkv)

            # Shared memory allocation
            smem = cute.arch.dynamic_smem_as(cutlass.BFloat16, self.get_smem_bytes() // 2)

            # Q tile in SMEM: [blk_m, D]
            q_smem_offset = 0
            # K tile in SMEM: [blk_n, D] × num_stages
            k_smem_offset = cfg.smem_q_bytes // 2
            # V tile in SMEM: [blk_n, D] × num_stages
            v_smem_offset = k_smem_offset + (cfg.smem_k_bytes * cfg.num_stages) // 2

            # Initialize accumulator registers
            # Each warp handles warp_m=16 rows of O: [16, D=128] in F32
            # That's 16*128 = 2048 floats / 32 lanes = 64 floats per lane
            # Plus m (16 values), l (16 values) per warp

            # Compute Q row range for this warp
            q_start = block_m_idx * Int32(cfg.blk_m) + warp_idx * Int32(cfg.warp_m)

            # Number of KV blocks to iterate
            num_kv_blocks = (seq_len_k + Int32(cfg.blk_n - 1)) // Int32(cfg.blk_n)

            # === Phase 1: Load Q to SMEM (cooperative, all warps) ===
            # Each thread loads 16B chunks via cp.async
            # ... (cooperative Q load across all threads)

            # === Phase 2: Main loop over K/V blocks ===
            # for kv_block in range(num_kv_blocks):
            #   1. cp.async K[kv_block] → SMEM
            #   2. wait K, compute S = Q @ K^T (HMMA BF16)
            #   3. online softmax on S
            #   4. cp.async V[kv_block] → SMEM
            #   5. wait V, compute O += P @ V (HMMA BF16)
            #   6. rescale O

            # === Phase 3: Normalize O, store to GMEM ===
            # O = O / l
            # convert F32 → BF16 and store

            # (Full implementation in next iteration — this is the skeleton)
            pass

        return sm120_fmha_fwd
