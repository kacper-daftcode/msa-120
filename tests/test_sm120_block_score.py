"""SM120 MSA max-pool block-score correctness test via torch JIT extension.

Validates block_max_score against a tight torch reference (exact GEMM + max,
no quantization) across several shapes: non-GQA, GQA, block_size 64/128, and a
partial last block.
"""
import math
import os

import torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")

print("Building SM120 block-score extension (JIT)...")
ext = load(
    name="sm120_block_score",
    sources=[os.path.join(_CSRC, "sm120_block_score.cu")],
    extra_cuda_cflags=[
        "-gencode=arch=compute_120f,code=sm_120f",
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
    ],
    verbose=False,
)
print("Extension built successfully.\n")


def reference_block_max(q, k, scale, block_size):
    """q [Sq, Hq, D] bf16, k [Sk, Hkv, D] bf16 -> [Hq, nblk, Sq] fp32."""
    seq_q, Hq, _ = q.shape
    seq_k, Hkv, _ = k.shape
    group = Hq // Hkv
    nblk = (seq_k + block_size - 1) // block_size
    out = torch.empty(Hq, nblk, seq_q, dtype=torch.float32, device=q.device)
    for h in range(Hq):
        hkv = h // group
        sc = (q[:, h, :].float() @ k[:, hkv, :].float().T) * scale  # [Sq, Sk]
        for blk in range(nblk):
            lo = blk * block_size
            hi = min((blk + 1) * block_size, seq_k)
            out[h, blk, :] = sc[:, lo:hi].max(dim=1).values
    return out


def run_case(name, seq_q, seq_k, Hq, Hkv, block_size, seed=0):
    torch.manual_seed(seed)
    D = 128
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(seq_q, Hq, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(seq_k, Hkv, D, dtype=torch.bfloat16, device="cuda")

    ref = reference_block_max(q, k, scale, block_size)
    got = ext.block_max_score(q, k, float(scale), int(block_size))

    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    diff = (got.float() - ref.float()).abs()
    maxabs = diff.max().item()
    rms = diff.pow(2).mean().sqrt().item()
    ok = rms < 1e-3
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name:42s} "
        f"shape={tuple(got.shape)} rms={rms:.3e} maxabs={maxabs:.3e}"
    )
    return ok


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (sm_{props.major}{props.minor})\n")

    cases = [
        # name, seq_q, seq_k, Hq, Hkv, block_size
        ("non-GQA bs128",                 64,  256, 8,  8,  128),
        ("non-GQA bs64",                  64,  256, 8,  8,  64),
        ("GQA Hq16/Hkv4 bs128",           96,  512, 16, 4,  128),
        ("GQA Hq16/Hkv4 bs64",            96,  512, 16, 4,  64),
        ("partial last block bs128",      48,  200, 8,  2,  128),
        ("partial last block bs64",       48,  200, 8,  2,  64),
        ("seq_k < block_size",            33,  100, 4,  4,  128),
        ("GQA partial bs64 odd",          17,  333, 12, 3,  64),
        ("single query",                  1,   257, 8,  8,  128),
    ]

    all_ok = True
    for c in cases:
        all_ok &= run_case(*c)

    print()
    if all_ok:
        print("ALL PASS")
    else:
        print("SOME FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
