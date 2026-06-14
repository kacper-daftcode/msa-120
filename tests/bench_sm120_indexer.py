# SPDX-License-Identifier: MIT
"""Wall-clock benchmark for the SM120 M3 indexer block-score kernel."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "python"))

from fmha_sm100.indexer_ref import IndexerWeights  # noqa: E402

_CSRC = _ROOT / "python" / "fmha_sm100" / "csrc" / "sm120_indexer.cu"


def _build():
    from torch.utils.cpp_extension import load

    return load(
        name="sm120_indexer",
        sources=[str(_CSRC)],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120f,code=sm_120f",
            "-O3", "-std=c++17", "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )


def bench(ext, N, dtype=torch.bfloat16, iters=50, warmup=10):
    hidden = 6144
    n_heads, head_dim = 4, 128
    block_size = 128
    scale = head_dim ** -0.5
    rotary_dim = 64
    rope_theta = 5_000_000.0
    eps = 1e-6

    torch.manual_seed(0)
    x = torch.randn(N, hidden, dtype=dtype, device="cuda")
    w = IndexerWeights(
        q_proj=torch.randn(n_heads * head_dim, hidden, dtype=dtype, device="cuda") * 0.02,
        k_proj=torch.randn(head_dim, hidden, dtype=dtype, device="cuda") * 0.02,
        q_norm=torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1,
        k_norm=torch.randn(head_dim, dtype=dtype, device="cuda") * 0.1,
    )
    positions = torch.arange(N, device="cuda", dtype=torch.int64)

    args = (x, x, w.q_proj, w.k_proj, w.q_norm, w.k_norm, positions,
            block_size, n_heads, head_dim, scale, rotary_dim, rope_theta, eps,
            True, True)

    for _ in range(warmup):
        ext.block_scores(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ext.block_scores(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    # FLOPs: projection GEMM q (N*512*6144*2) + k (N*128*6144*2)
    proj_flops = N * (n_heads * head_dim) * hidden * 2 + N * head_dim * hidden * 2
    # scoring: H * N(queries) * N(keys, ~causal half) * d * 2
    nblk = (N + block_size - 1) // block_size
    score_flops = n_heads * N * N * head_dim * 2 * 0.5  # ~causal
    total_flops = proj_flops + score_flops
    tflops = total_flops / (ms * 1e-3) / 1e12
    return ms, tflops, proj_flops, score_flops


def main():
    ext = _build()
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"{'N':>6} {'dtype':>10} {'ms/call':>10} {'TFLOPS':>10}")
    for N in [512, 1024, 2048, 4096]:
        for dtype in [torch.bfloat16]:
            ms, tf, pf, sf = bench(ext, N, dtype=dtype)
            print(f"{N:>6} {str(dtype).split('.')[-1]:>10} {ms:>10.4f} {tf:>10.2f}")


if __name__ == "__main__":
    main()
