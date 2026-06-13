"""SM120 FlashAttention correctness test via torch extension."""
import os
import sys
import torch

# Build the extension inline (JIT)
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")

print("Building SM120 FMHA extension (JIT)...")
sm120_fmha = load(
    name="sm120_fmha",
    sources=[
        os.path.join(_CSRC, "sm120_launch.cu"),
        os.path.join(_CSRC, "sm120_fmha_fwd.cu"),
    ],
    extra_cuda_cflags=[
        "-gencode=arch=compute_120f,code=sm_120f",
        "-O3", "-std=c++17",
        "--expt-relaxed-constexpr",
    ],
    extra_ldflags=[
        f"-L/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib",
        "-lcudart",
    ],
    verbose=True,
)
print("Extension built successfully.\n")


def reference_attention(q, k, v, scale):
    """PyTorch reference: standard attention."""
    # q: [S_q, Hq, D], k: [S_k, Hkv, D], v: [S_k, Hkv, D]
    S = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * scale
    P = torch.softmax(S, dim=-1)
    O = torch.einsum("qhk,khd->qhd", P, v.float())
    return O.to(q.dtype)


def test_sm120_fmha():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return

    props = torch.cuda.get_device_properties(0)
    if props.major != 12:
        print(f"SKIP: GPU is sm_{props.major}{props.minor}, need sm_120")
        return

    print(f"GPU: {props.name} (sm_{props.major}{props.minor})")

    # Test config
    seq_q, seq_k = 64, 64
    num_heads_q, num_heads_kv = 8, 8
    head_dim = 128
    scale = 1.0 / (head_dim ** 0.5)

    torch.manual_seed(42)
    q = torch.randn(seq_q, num_heads_q, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(seq_k, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(seq_k, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1

    # Reference
    ref_o = reference_attention(q, k, v, scale)

    # SM120 kernel
    o, lse = sm120_fmha.forward(q.contiguous(), k.contiguous(), v.contiguous(), scale)

    # Compare
    max_err = (o.float() - ref_o.float()).abs().max().item()
    mean_err = (o.float() - ref_o.float()).abs().mean().item()

    print(f"\nResults (seq_q={seq_q}, seq_k={seq_k}, heads={num_heads_q}, D={head_dim}):")
    print(f"  Max abs error:  {max_err:.6f}")
    print(f"  Mean abs error: {mean_err:.6f}")

    assert max_err < 0.01, f"max error {max_err} too large!"
    print("\n✓ SM120 FlashAttention PASS")

    # Larger test
    seq_q2, seq_k2 = 256, 512
    q2 = torch.randn(seq_q2, num_heads_q, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    k2 = torch.randn(seq_k2, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    v2 = torch.randn(seq_k2, num_heads_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1

    ref_o2 = reference_attention(q2, k2, v2, scale)
    o2, _ = sm120_fmha.forward(q2.contiguous(), k2.contiguous(), v2.contiguous(), scale)

    max_err2 = (o2.float() - ref_o2.float()).abs().max().item()
    mean_err2 = (o2.float() - ref_o2.float()).abs().mean().item()

    print(f"\nLarger test (seq_q={seq_q2}, seq_k={seq_k2}):")
    print(f"  Max abs error:  {max_err2:.6f}")
    print(f"  Mean abs error: {mean_err2:.6f}")

    assert max_err2 < 0.02, f"max error {max_err2} too large for larger test!"
    print("✓ SM120 FlashAttention larger test PASS")


if __name__ == "__main__":
    test_sm120_fmha()
