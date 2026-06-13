# SPDX-FileCopyrightText: Copyright (c) 2026 turbollama contributors
# SPDX-License-Identifier: MIT
#
# SM120 copy utilities — replaces TMA with cp.async (LDGSTS).
#
# SM120 cannot use cp.async.bulk.tensor (TMA) because mbarrier TX
# completion path is non-functional. Instead we use per-thread
# cp.async.ca.shared.global (LDGSTS) which is proven to work.

"""
SM120 copy primitives for attention kernel data loading.

Replaces SM100's TMA (cp.async.bulk.tensor.2d) with:
  - cp.async.ca.shared.global for GMEM→SMEM async copy (per-thread)
  - cp.async.commit_group / cp.async.wait_group for synchronization

The per-thread copy is less efficient than TMA's bulk engine, but
it's the only working async GMEM→SMEM path on SM120.
"""

from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def cp_async_load_16b(smem_ptr, gmem_ptr, *, loc=None, ip=None):
    """cp.async.ca.shared.global 16 bytes (128 bits) from GMEM to SMEM.
    
    This is the SM120 replacement for TMA tile loads. Each thread
    copies 16 bytes independently. For a 128-byte cache line,
    8 threads cooperate (threadIdx.x % 8).
    """
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa;\n"
        "cvt.u32.u64 sa, $0;\n"
        "cp.async.ca.shared.global [sa], [$1], 16;\n"
        "}\n",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_load_4b(smem_ptr, gmem_ptr, *, loc=None, ip=None):
    """cp.async.ca.shared.global 4 bytes from GMEM to SMEM."""
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .u32 sa;\n"
        "cvt.u32.u64 sa, $0;\n"
        "cp.async.ca.shared.global [sa], [$1], 4;\n"
        "}\n",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    """cp.async.commit_group — commit pending async copies."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.commit_group;\n",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_wait_group(n: int = 0, *, loc=None, ip=None):
    """cp.async.wait_group N — wait until at most N groups pending."""
    llvm.inline_asm(
        None,
        [],
        f"cp.async.wait_group {n};\n",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldsm_fp6_unpack(dst_reg, smem_addr, *, loc=None, ip=None):
    """ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32
    
    Hardware FP6 unpack: 12 bytes (16 × 6-bit codes) from SMEM →
    16 bytes (16 × 8-bit containers) in register.
    Verified working on SM120 (2026-06-12).
    """
    llvm.inline_asm(
        None,
        [
            dst_reg.ir_value(loc=loc, ip=ip),
            smem_addr.ir_value(loc=loc, ip=ip),
        ],
        "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {$0}, [$1];\n",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op  
def ldsm_fp4_unpack(dst_reg, smem_addr, *, loc=None, ip=None):
    """ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64
    
    Hardware FP4 unpack: 8 bytes (16 × 4-bit codes) from SMEM →
    16 bytes (16 × 8-bit containers) in register.
    Verified working on SM120 (2026-06-12).
    """
    llvm.inline_asm(
        None,
        [
            dst_reg.ir_value(loc=loc, ip=ip),
            smem_addr.ir_value(loc=loc, ip=ip),
        ],
        "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 {$0}, [$1];\n",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
