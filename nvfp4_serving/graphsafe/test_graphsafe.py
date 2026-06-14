"""Numerical equivalence: graphsafe_swigluoai_nvfp4_moe vs the validated
unfused_swigluoai_nvfp4_moe on random NVFP4-shaped inputs.

Both implementations compute the SAME thing (single-group group_gemm per expert,
swigluoai, weighted combine); the graphsafe one just dispatches with fixed
shapes.  So they must agree to ~fp noise (rel RMS < 1e-3).  We DO NOT need real
checkpoint weights for equivalence -- random NVFP4-shaped packed weights /
e4m3 scales exercise the exact same kernels.

Run (inside the image, only when GPUs are free -- see CAPTURE_NOTES.md):

  sudo docker run --rm --gpus all \
    -v /home/kacper/msa-120/nvfp4_serving:/work \
    --entrypoint python3 vllm/vllm-openai:minimax-m3 \
    /work/graphsafe/test_graphsafe.py

It imports the validated reference from ../unfused_moe/unfused_moe.py.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "unfused_moe"))

from graphsafe_moe import (  # noqa: E402
    graphsafe_swigluoai_nvfp4_moe,
    build_expert_weight_scales_mma,
)
from unfused_moe import unfused_swigluoai_nvfp4_moe  # noqa: E402  (validated ref)

DEV = "cuda"
FP4_MAX = 6.0
E4M3_MAX = 448.0


def rel_rms(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).pow(2).mean().sqrt() / (b.pow(2).mean().sqrt() + 1e-20)).item()


def _rand_nvfp4_weight(N, K, dev=DEV, seed=0):
    """Random NVFP4-shaped weight for one expert.

    Returns (packed [N, K//2] uint8, block_scale [N, K//16] e4m3, global fp32).
    Values are arbitrary but valid -- we only need both impls to run the same
    kernels on the same data, not a meaningful GEMM.
    """
    g = torch.Generator(device=dev).manual_seed(seed)
    packed = torch.randint(0, 256, (N, K // 2), device=dev, dtype=torch.uint8, generator=g)
    # block scales: small positive e4m3 values
    bs = (torch.rand(N, K // 16, device=dev, generator=g) * 0.5 + 0.25).to(torch.float8_e4m3fn)
    gscale = torch.tensor(1.0 / 6.0, device=dev, dtype=torch.float32)
    return packed, bs, gscale


def build_case(E, I, H, T, k, seed=1234):
    torch.manual_seed(seed)
    w13_packed, w13_scale, w13_scale_2 = [], [], []
    w2_packed, w2_scale, w2_scale_2 = [], [], []
    for e in range(E):
        gp, gs, g2 = _rand_nvfp4_weight(2 * I, H, seed=seed + e)
        dp, ds, d2 = _rand_nvfp4_weight(H, I, seed=seed + 1000 + e)
        w13_packed.append(gp)
        w13_scale.append(gs)
        w13_scale_2.append(g2)
        w2_packed.append(dp)
        w2_scale.append(ds)
        w2_scale_2.append(d2)
    w13_packed = torch.stack(w13_packed)
    w2_packed = torch.stack(w2_packed)
    w13_scale_2 = torch.stack(w13_scale_2).reshape(-1)
    w2_scale_2 = torch.stack(w2_scale_2).reshape(-1)

    x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    router = torch.randn(T, E, device=DEV).float()
    probs = torch.softmax(router, -1)
    topk_w, topk_ids = torch.topk(probs, k, -1)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)) * 2.0  # routed_scaling_factor

    w13_sf_mma = build_expert_weight_scales_mma(w13_scale, 2 * I, H)
    w2_sf_mma = build_expert_weight_scales_mma(w2_scale, H, I)
    return dict(
        x=x, w13_packed=w13_packed, w13_scale=w13_scale, w13_scale_2=w13_scale_2,
        w2_packed=w2_packed, w2_scale=w2_scale, w2_scale_2=w2_scale_2,
        topk_ids=topk_ids, topk_weights=topk_w,
        w13_sf_mma=w13_sf_mma, w2_sf_mma=w2_sf_mma,
    )


def run_pair(c, expert_capacity=None):
    ref = unfused_swigluoai_nvfp4_moe(
        c["x"], c["w13_packed"], c["w13_scale"], c["w13_scale_2"],
        c["w2_packed"], c["w2_scale"], c["w2_scale_2"],
        c["topk_ids"], c["topk_weights"],
        activation="swigluoai",
        w13_sf_mma=c["w13_sf_mma"], w2_sf_mma=c["w2_sf_mma"],
    )
    gs = graphsafe_swigluoai_nvfp4_moe(
        c["x"], c["w13_packed"], c["w13_scale"], c["w13_scale_2"],
        c["w2_packed"], c["w2_scale"], c["w2_scale_2"],
        c["topk_ids"], c["topk_weights"],
        activation="swigluoai",
        w13_sf_mma=c["w13_sf_mma"], w2_sf_mma=c["w2_sf_mma"],
        expert_capacity=expert_capacity,
    )
    return ref, gs


def main():
    assert torch.cuda.is_available(), "needs a GPU + flashinfer (run in image)"
    H, I = 6144, 1536  # per-rank TP4 shapes from the task (2I=1536 -> I=768? see note)
    # NOTE: task says per-rank 2I=1536 => I=768. Use that; also a generic case.
    cases = [
        # (E, I, H, T, k, expert_capacity)
        (8, 768, 6144, 1, 4, None),     # decode batch=1, worst-case C
        (8, 768, 6144, 1, 4, 4),        # decode batch=1, tight static C
        (16, 768, 6144, 17, 4, None),   # prefill-ish, worst-case C
        (32, 512, 4096, 64, 4, None),   # larger, generic
        (8, 768, 6144, 8, 2, 8),        # tight C that may force overflow-drop
    ]
    worst = 0.0
    all_ok = True
    for (E, Ii, Hh, T, k, cap) in cases:
        c = build_case(E, Ii, Hh, T, k, seed=1234 + E + T)
        ref, gsout = run_pair(c, expert_capacity=cap)
        r = rel_rms(gsout, ref)
        # For tight cap that can overflow, only worst-case-cap is guaranteed exact.
        tol = 1e-3 if cap is None or cap >= T * k else 5e-2
        ok = r < tol
        all_ok = all_ok and ok
        worst = max(worst, r if (cap is None or cap >= T * k) else 0.0)
        print(
            f"E={E:3d} I={Ii:4d} H={Hh:4d} T={T:3d} k={k} cap={str(cap):>5}: "
            f"rel RMS = {r:.2e}  tol={tol:.0e}  {'OK' if ok else 'FAIL'}"
        )
    print("-" * 60)
    print(f"worst (exact-cap) rel RMS = {worst:.2e}  (threshold 1e-3)")
    print("RESULT:", "PASS" if (all_ok and worst < 1e-3) else "FAIL")
    sys.exit(0 if (all_ok and worst < 1e-3) else 1)


if __name__ == "__main__":
    main()
