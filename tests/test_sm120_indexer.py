# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""GPU validation of the SM120 M3 learned indexer block-score kernel vs the
verified torch reference python/fmha_sm100/indexer_ref.py.

Run directly:
    CUDA_VISIBLE_DEVICES=0 python tests/test_sm120_indexer.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from fmha_sm100.indexer_ref import IndexerWeights, m3_indexer_block_scores  # noqa: E402

_CSRC = _ROOT / "python" / "fmha_sm100" / "csrc" / "sm120_indexer.cu"


def _build():
    from torch.utils.cpp_extension import load

    return load(
        name="sm120_indexer",
        sources=[str(_CSRC)],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120f,code=sm_120f",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ],
        verbose=True,
    )


def _run_case(ext, N, *, dtype, causal, apply_rope, seed):
    torch.manual_seed(seed)
    hidden = 6144
    n_heads, head_dim = 4, 128
    block_size = 128
    scale = head_dim ** -0.5
    rotary_dim = 64
    rope_theta = 5_000_000.0
    eps = 1e-6

    x = torch.randn(N, hidden, dtype=dtype, device="cuda")
    w = IndexerWeights(
        q_proj=torch.randn(n_heads * head_dim, hidden, dtype=dtype, device="cuda") * 0.02,
        k_proj=torch.randn(head_dim, hidden, dtype=dtype, device="cuda") * 0.02,
        q_norm=torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1,
        k_norm=torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1,
    )
    positions = torch.arange(N, device="cuda", dtype=torch.int64)

    ref = m3_indexer_block_scores(
        x, x, w, block_size=block_size, n_heads=n_heads, head_dim=head_dim,
        scale=scale, rotary_dim=rotary_dim, rope_theta=rope_theta, eps=eps,
        positions=positions, causal=causal, apply_rope=apply_rope, project=True,
    )  # [H, nblk, N] fp32

    out = ext.block_scores(
        x, x, w.q_proj, w.k_proj, w.q_norm, w.k_norm, positions,
        block_size, n_heads, head_dim, scale, rotary_dim, rope_theta, eps,
        causal, apply_rope,
    )

    assert out.shape == ref.shape, (out.shape, ref.shape)

    finite = torch.isfinite(ref)
    # both must agree on which entries are -inf (causal-masked / empty)
    mask_match = bool((torch.isfinite(out) == finite).all().item())

    rdiff = (out[finite] - ref[finite]).float()
    maxabs = rdiff.abs().max().item() if rdiff.numel() else 0.0
    rms = rdiff.pow(2).mean().sqrt().item() if rdiff.numel() else 0.0
    return rms, maxabs, mask_match, ref.shape


def main():
    if not torch.cuda.is_available():
        print("NO CUDA DEVICE")
        sys.exit(1)
    print(f"device: {torch.cuda.get_device_name(0)}")
    ext = _build()

    cases = [
        # (N, dtype, causal, apply_rope, label)
        (128, torch.bfloat16, True, True, "N=128 exact-1-block causal+rope bf16"),
        (384, torch.bfloat16, True, True, "N=384 3-block causal+rope bf16"),
        (300, torch.bfloat16, True, True, "N=300 partial-last-block causal+rope bf16"),
        (512, torch.bfloat16, True, True, "N=512 4-block causal+rope bf16"),
        (200, torch.bfloat16, False, True, "N=200 non-causal+rope bf16"),
        (256, torch.bfloat16, True, False, "N=256 causal no-rope bf16"),
        (384, torch.float32, True, True, "N=384 causal+rope fp32"),
        (300, torch.float32, True, True, "N=300 partial causal+rope fp32"),
    ]

    print("\n%-44s %-12s %-12s %-10s %s" % ("case", "rms", "maxabs", "mask", "result"))
    print("-" * 100)
    all_pass = True
    for N, dtype, causal, apply_rope, label in cases:
        rms, maxabs, mask_match, shape = _run_case(
            ext, N, dtype=dtype, causal=causal, apply_rope=apply_rope, seed=N + int(causal)
        )
        tol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        ok = (rms < tol) and mask_match
        all_pass = all_pass and ok
        print("%-44s %-12.3e %-12.3e %-10s %s" % (
            label, rms, maxabs, str(mask_match), "PASS" if ok else "FAIL"))

    print("-" * 100)
    print("ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
