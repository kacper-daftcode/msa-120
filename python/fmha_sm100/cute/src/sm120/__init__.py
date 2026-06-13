# SPDX-FileCopyrightText: Copyright (c) 2026 turbollama contributors
# SPDX-License-Identifier: MIT

"""SM120 (RTX 5090 / RTX PRO 6000) attention kernels.

Per-warp MMA path using HMMA (BF16) and QMMA.SF (FP8 block-scaled).
Replaces SM100's tcgen05/TMEM/TMA architecture with:
  - cp.async (LDGSTS) for GMEM→SMEM
  - mma.sync per-warp for compute
  - Register-based accumulators
"""
