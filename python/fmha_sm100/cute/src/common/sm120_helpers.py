# SPDX-FileCopyrightText: Copyright (c) 2026 turbollama contributors
# SPDX-License-Identifier: MIT
#
# SM120 (RTX 5090) MMA helpers — per-warp QMMA.SF path.
#
# Replaces SM100 tcgen05.mma (TMEM-based, async) with SM120 per-warp
# mma.sync.aligned (register-based, synchronous).
#
# Hardware-verified on SM120 (2026-06-12):
#   - QMMA.SF (dense block-scaled E2M3/E4M3/E5M2): WORKS
#   - QMMA.SF.SP (sparse 2:4 block-scaled): WORKS
#   - LDSM.U6x16P32TO8 (FP6 packed unpack): WORKS
#   - LDSM.U4x16P64TO8 (FP4 packed unpack): WORKS
#   - OMMA.SF (FP4 kind::mxf4): WORKS
#   - cp.async (LDGSTS per-thread): WORKS
#   - mbarrier basic (init/arrive/try_wait): WORKS
#   - TMA (cp.async.bulk.*): BROKEN (mbarrier TX path non-functional)
#   - TMEM: BROKEN (controller uninitialized)

"""
SM120 attention MMA primitives using per-warp QMMA path.

On SM120, the tcgen05 (UTC) data movement infrastructure is disabled:
- TMA (cp.async.bulk.tensor) hangs (mbarrier.arrive.expect_tx broken)
- TMEM (tcgen05.alloc/st/ld) traps (controller uninitialized)

But the per-warp compute path is fully functional:
- QMMA.SF: block-scaled FP8/FP6/FP4 MMA with UE8M0 scales
- QMMA.SF.SP: sparse 2:4 variant
- LDSM packed types: hardware FP6/FP4 unpack from SMEM
- cp.async (LDGSTS): per-thread async GMEM→SMEM

The SM120 attention kernel uses:
  LDG/cp.async → SMEM → registers → mma.sync (QMMA) → registers → STG

Instead of SM100's:
  TMA → SMEM → tcgen05.mma (TMEM) → tcgen05.ld → registers → STG
"""

from typing import Optional, Tuple
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr
from cutlass._mlir.dialects import llvm


class SM120MmaFormat(IntEnum):
    """Element type encodings for mma.sync block-scaled instructions."""
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3  # FP6
    E3M2 = 4  # FP6 alt
    E2M1 = 5  # FP4


def cutlass_type_to_mma_format(ct) -> str:
    """Map CUTLASS type to PTX mma.sync type string."""
    if ct is cutlass.Float8E4M3FN:
        return "e4m3"
    if ct is cutlass.Float8E5M2:
        return "e5m2"
    if ct is cutlass.BFloat16:
        return "bf16"
    if ct is cutlass.Float16:
        return "f16"
    raise TypeError(f"Unsupported type for SM120 MMA: {ct!r}")


def make_sm120_mma_asm(
    a_type: str = "e4m3",
    b_type: str = "e4m3",
    m: int = 16,
    n: int = 8,
    k: int = 32,
    block_scale: bool = True,
) -> str:
    """Generate inline PTX asm string for SM120 per-warp block-scaled MMA.
    
    Returns the mma.sync instruction string for use in llvm.inline_asm.
    
    For kind::mxf8f6f4 (FP8/FP6 types):
      A fragment: 4 registers, B fragment: 2 registers
      D/C accumulator: 4 float registers
      Scale A/B: 1 register each + byte selector {0,1}
    """
    if block_scale:
        return (
            f"mma.sync.aligned.m{m}n{n}k{k}.row.col"
            f".kind::mxf8f6f4.block_scale.scale_vec::1X"
            f".f32.{a_type}.{b_type}.f32.ue8m0"
        )
    else:
        return f"mma.sync.aligned.m{m}n{n}k{k}.row.col.f32.{a_type}.{b_type}.f32"


# SM120 uses cp.async (LDGSTS) instead of TMA for GMEM→SMEM transfers
SM120_COPY_METHOD = "cp.async"  # vs SM100's "tma"

# SM120 uses registers for MMA operands, not TMEM
SM120_MMA_OPERAND_SRC = "register"  # vs SM100's "smem_descriptor" or "tmem"

# SM120 MMA shapes available with block-scaled FP8
SM120_MMA_SHAPES = {
    "f8f6f4": {"m": 16, "n": 8, "k": 32},   # QMMA.SF
    "f4":     {"m": 16, "n": 8, "k": 64},    # OMMA.SF (kind::mxf4)
    "f16":    {"m": 16, "n": 8, "k": 16},    # HMMA
    "bf16":   {"m": 16, "n": 8, "k": 16},    # HMMA
}

# Fragment sizes (registers per thread) for SM120 MMA
SM120_FRAGMENT_SIZES = {
    "f8f6f4_A": 4,  # 4 × u32 = 16 bytes = 16 codes in 8-bit containers
    "f8f6f4_B": 2,  # 2 × u32 = 8 bytes = 8 codes
    "f8f6f4_D": 4,  # 4 × f32 = 16 bytes accumulator
    "f16_A": 4,      # 4 × u32 = 8 f16 values
    "f16_B": 2,
    "f16_D": 4,
}
