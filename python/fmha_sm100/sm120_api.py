# SPDX-FileCopyrightText: Copyright (c) 2026 turbollama contributors
# SPDX-License-Identifier: MIT

"""SM120 FlashAttention API — drop-in compatible with MSA fmha_sm100 interface.

Usage:
    from fmha_sm100.sm120_api import fmha_sm120, is_sm120

    if is_sm120():
        out, lse = fmha_sm120(q, k, v, softmax_scale=1/sqrt(D))
"""

import math
import os
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

_ext = None
_ext_lock = __import__("threading").Lock()
_CSRC = Path(__file__).parent / "csrc"


def _build_extension():
    global _ext
    if _ext is not None:
        return _ext
    with _ext_lock:
        if _ext is not None:
            return _ext
        logger.info("JIT-compiling SM120 FMHA kernels (first import)...")
        _ext = load(
            name="sm120_fmha_ext",
            sources=[
                str(_CSRC / "sm120_launch.cu"),
                str(_CSRC / "sm120_fmha_fwd.cu"),
                str(_CSRC / "sm120_fmha_fwd_fp8.cu"),
            ],
            extra_cuda_cflags=[
                "-gencode=arch=compute_120f,code=sm_120f",
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
            ],
            extra_ldflags=[
                "-L/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib",
                "-lcudart",
            ],
            verbose=False,
        )
        logger.info("SM120 FMHA extension compiled.")
        return _ext


def is_sm120() -> bool:
    """Check if current GPU is SM120 (Blackwell consumer/workstation)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.major == 12 and props.minor == 0


def fmha_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SM120 FlashAttention forward pass.

    Automatically dispatches to BF16 or FP8 kernel based on input dtype.
    Compatible with MSA's fmha_sm100() interface pattern.

    Args:
        q: [total_q, num_heads_q, D] — BF16 or FP8_E4M3
        k: [total_kv, num_heads_kv, D] — same dtype as q
        v: [total_kv, num_heads_kv, D] — same dtype as q (or k)
        softmax_scale: 1/sqrt(head_dim) if None
        causal: causal mask (not yet supported)

    Returns:
        (output, lse):
            output: [total_q, num_heads_q, D] — BF16
            lse: [total_q, num_heads_q] — FP32
    """
    if causal:
        raise NotImplementedError("SM120 FA: causal mask not yet implemented")

    assert q.dim() == 3, f"q must be [total_q, Hq, D], got shape {q.shape}"
    head_dim = q.size(2)
    assert head_dim == 128, f"SM120 FA supports D=128 only, got {head_dim}"

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Handle batched inputs [B, S, H, D] → flatten to [B*S, H, D]
    orig_shape = None
    if q.dim() == 4:
        B, S = q.shape[:2]
        orig_shape = (B, S)
        q = q.reshape(B * S, q.shape[2], q.shape[3])
        k = k.reshape(-1, k.shape[-2], k.shape[-1])
        v = v.reshape(-1, v.shape[-2], v.shape[-1])

    ext = _build_extension()
    o, lse = ext.forward(q.contiguous(), k.contiguous(), v.contiguous(), softmax_scale)

    if orig_shape is not None:
        o = o.reshape(orig_shape[0], orig_shape[1], o.shape[1], o.shape[2])

    return o, lse


def fmha_sm120_plan(*args, **kwargs):
    """Placeholder for MSA plan API compatibility.

    SM120 kernels don't need a separate planning step (no TMA descriptors).
    Returns a dummy plan object for API compatibility.
    """
    return {"sm120": True}
