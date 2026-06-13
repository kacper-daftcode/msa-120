"""SM120 FlashAttention unified test — BF16 + FP8 paths."""
import os
import math
import torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")

print("Building SM120 FMHA unified extension (BF16 + FP8)...")
sm120_fmha = load(
    name="sm120_fmha_unified",
    sources=[
        os.path.join(_CSRC, "sm120_launch.cu"),
        os.path.join(_CSRC, "sm120_fmha_fwd.cu"),
        os.path.join(_CSRC, "sm120_fmha_fwd_fp8.cu"),
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
print("Extension built.\n")


def ref_attn(q, k, v, scale):
    """Reference attention with GQA support."""
    hq, hkv = q.shape[1], k.shape[1]
    if hq != hkv:
        # Expand KV heads for GQA
        repeat = hq // hkv
        k = k[:, :, None, :].expand(-1, hkv, repeat, -1).reshape(k.shape[0], hq, -1)
        v = v[:, :, None, :].expand(-1, hkv, repeat, -1).reshape(v.shape[0], hq, -1)
    S = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * scale
    P = torch.softmax(S, dim=-1)
    return torch.einsum("qhk,khd->qhd", P, v.float())


def test_bf16():
    print("=" * 60)
    print("SM120 BF16 FlashAttention")
    print("=" * 60)
    scale = 1.0 / math.sqrt(128)
    configs = [
        (64, 64, 8, 8, "basic"),
        (256, 512, 8, 8, "medium"),
        (1024, 1024, 4, 4, "1K×1K"),
        (128, 256, 32, 8, "GQA 32/8"),
        (63, 65, 1, 1, "non-aligned"),
    ]
    passed = 0
    for sq, sk, hq, hkv, name in configs:
        torch.manual_seed(42)
        q = torch.randn(sq, hq, 128, dtype=torch.bfloat16, device="cuda") * 0.1
        k = torch.randn(sk, hkv, 128, dtype=torch.bfloat16, device="cuda") * 0.1
        v = torch.randn(sk, hkv, 128, dtype=torch.bfloat16, device="cuda") * 0.1
        ref = ref_attn(q, k, v, scale).to(torch.bfloat16)
        got, _ = sm120_fmha.forward_bf16(q, k, v, scale)
        err = (got.float() - ref.float()).abs().max().item()
        ok = err < 0.005
        print(f"  {'✓' if ok else '✗'} {name:15s} ({sq}×{sk} h={hq}/{hkv}): err={err:.6f}")
        if ok: passed += 1
    print(f"  BF16: {passed}/{len(configs)} passed\n")
    return passed == len(configs)


def test_fp8():
    print("=" * 60)
    print("SM120 FP8 (QMMA.SF E4M3) FlashAttention")
    print("=" * 60)
    scale = 1.0 / math.sqrt(128)
    configs = [
        (64, 64, 1, 1, "basic 1h"),
        (128, 128, 1, 1, "2 blocks"),
        (64, 64, 4, 4, "4 heads"),
        (256, 256, 1, 1, "large"),
        (64, 64, 8, 4, "GQA 8/4"),
        (63, 65, 1, 1, "non-aligned"),
    ]
    passed = 0
    for sq, sk, hq, hkv, name in configs:
        torch.manual_seed(42)
        # Generate in float, quantize to FP8
        q_f = torch.randn(sq, hq, 128, device="cuda") * 0.1
        k_f = torch.randn(sk, hkv, 128, device="cuda") * 0.1
        v_f = torch.randn(sk, hkv, 128, device="cuda") * 0.1
        q = q_f.to(torch.float8_e4m3fn)
        k = k_f.to(torch.float8_e4m3fn)
        v = v_f.to(torch.float8_e4m3fn)
        # Reference using the actual FP8 quantized values
        ref = ref_attn(q.float(), k.float(), v.float(), scale)
        got, _ = sm120_fmha.forward_fp8(q, k, v, scale)
        err = (got.float() - ref.float()).abs().max().item()
        ok = err < 0.02  # FP8 has wider tolerance
        print(f"  {'✓' if ok else '✗'} {name:15s} ({sq}×{sk} h={hq}/{hkv}): err={err:.6f}")
        if ok: passed += 1
    print(f"  FP8: {passed}/{len(configs)} passed\n")
    return passed == len(configs)


def test_auto_dispatch():
    print("=" * 60)
    print("Auto-dispatch (forward() picks BF16 or FP8)")
    print("=" * 60)
    scale = 1.0 / math.sqrt(128)
    torch.manual_seed(42)

    # BF16 auto
    q = torch.randn(64, 4, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(64, 4, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(64, 4, 128, dtype=torch.bfloat16, device="cuda") * 0.1
    got, _ = sm120_fmha.forward(q, k, v, scale)
    assert got.dtype == torch.bfloat16
    print("  ✓ BF16 auto-dispatch works")

    # FP8 auto
    q8 = q.to(torch.float8_e4m3fn)
    k8 = k.to(torch.float8_e4m3fn)
    v8 = v.to(torch.float8_e4m3fn)
    got8, _ = sm120_fmha.forward(q8, k8, v8, scale)
    assert got8.dtype == torch.bfloat16  # FP8 kernel outputs BF16
    print("  ✓ FP8 auto-dispatch works")
    print()
    return True


if __name__ == "__main__":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (sm_{props.major}{props.minor})\n")

    r1 = test_bf16()
    r2 = test_fp8()
    r3 = test_auto_dispatch()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'✓' if r1 else '✗'} BF16 tests")
    print(f"  {'✓' if r2 else '✗'} FP8 tests")
    print(f"  {'✓' if r3 else '✗'} Auto-dispatch")
    print()
    if r1 and r2 and r3:
        print(">>> ALL PASSED <<<")
    else:
        print(">>> SOME FAILED <<<")
