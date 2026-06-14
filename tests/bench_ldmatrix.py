"""Bench harness for SM120 dense BF16 forward at N=4096 prefill."""
import os, math, time, torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")
EXT = os.environ.get("BENCH_EXT", "sm120_fmha_bench")

print(f"Building {EXT}...")
sm120 = load(
    name=EXT,
    sources=[
        os.path.join(_CSRC, "sm120_launch.cu"),
        os.path.join(_CSRC, "sm120_fmha_fwd.cu"),
        os.path.join(_CSRC, "sm120_fmha_fwd_fp8.cu"),
    ],
    extra_cuda_cflags=[
        "-gencode=arch=compute_120f,code=sm_120f",
        "-O3", "-std=c++17", "--expt-relaxed-constexpr",
    ],
    extra_ldflags=[
        "-L/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib", "-lcudart",
    ],
    verbose=False,
)
print("built.\n")

def bench(N=4096, H=16, D=128, iters=50, warmup=10):
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    q = torch.randn(N, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(N, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(N, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
    for _ in range(warmup):
        o, _ = sm120.forward_bf16(q, k, v, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o, _ = sm120.forward_bf16(q, k, v, scale)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1e3
    # FLOPs: 2 GEMMs (QK + PV), each 2*N*N*D, times H heads (full, non-causal)
    flops = 2 * (2.0 * N * N * D) * H
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"N={N} H={H} D={D}: {ms:.3f} ms/iter  {tflops:.2f} TFLOPS")
    return ms, tflops

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    bench()
    bench()
