# SPDX-License-Identifier: MIT
"""Reproduce + fix the multi-group SFA-offset bug in flashinfer 0.6.12's SM120
NVFP4 grouped GEMM (group_gemm_nvfp4_nt_groupwise).

Builds a tiny 4-group NVFP4 grouped GEMM and compares, per group, three methods
against a bf16-dequant reference:

  [REF ]  per-group single-group call  (the validated method; noise floor)
  [BUG ]  naive multi-group call: a_scale = concat of per-group swizzled blocks
          (the obvious/intended layout) -> groups >=2 read wrong SF rows
  [FIX ]  multi-group call with a_scale SCATTERED to the kernel's per-group
          offsets (batched_nvfp4_moe.batched_nvfp4_gemm)

Expected: [REF] and [FIX] at the NVFP4 noise floor for ALL groups; [BUG] blows
up for groups i>=2.

Run (few GB; OK alongside nothing else on the GPU):
    python3 repro_multigroup_sfa.py
"""
import os
import sys
import torch
import flashinfer
from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batched_nvfp4_moe import (
    _quant_gs, _pad4, _pad128, _sf_k_of, kernel_sf_m_offset,
    weight_scale_to_mma, build_batched_weight_scale_mma, batched_nvfp4_gemm,
    FP4_MAX, E4M3_MAX,
)

torch.manual_seed(0)
DEV = "cuda"


def rel_rms(a, b):
    a = a.float(); b = b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) /
            torch.sqrt(torch.mean(b ** 2)).clamp_min(1e-12)).item()


def make_nvfp4_weight(N, K):
    """Random bf16 weight -> NVFP4 packed + on-disk LINEAR e4m3 block scale +
    global scale_2, plus a bf16 dequant for the reference."""
    w = (torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.2)
    w_gs = _quant_gs(w)                      # quant global scale
    w_q, w_sf_sw = flashinfer.fp4_quantize(  # swizzled SF
        w, global_scale=w_gs, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    # also get LINEAR scale for the per-expert weight_scale_to_mma path:
    # dequant back to bf16 for the reference using the swizzled scales directly.
    w_scale_2 = (1.0 / w_gs).float()         # checkpoint dequant global scale
    return w, w_q, w_sf_sw, w_gs, w_scale_2


def linear_sf_from_weight(w, w_gs):
    """Produce on-disk LINEAR [N, K//16] e4m3 block scale (is_sf_swizzled=False)
    so we can drive weight_scale_to_mma / build_batched_weight_scale_mma exactly
    like the production path."""
    _, sf_lin = flashinfer.fp4_quantize(
        w, global_scale=w_gs, sf_vec_size=16, is_sf_swizzled_layout=False
    )
    N, K = w.shape
    return sf_lin.view(torch.uint8).reshape(N, K // 16)


def dequant_ref(x, w_bf16):
    """bf16 matmul reference (x already bf16)."""
    return (x.float() @ w_bf16.float().t())


def per_group_ref(x_chunks, w_packed, w_scale_list, w_scale_2, N, K):
    """The validated single-group-per-call method."""
    outs = []
    for i, xe in enumerate(x_chunks):
        n = xe.shape[0]
        mp = _pad4(n)
        xx = xe if mp == n else torch.cat([xe, xe.new_zeros(mp - n, K)], 0)
        gs = _quant_gs(xx)
        a_q, a_sf = flashinfer.fp4_quantize(
            xx, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        w_sf_mma = weight_scale_to_mma(w_scale_list[i], N, K)
        alpha = ((1.0 / gs) * w_scale_2[i].float()).reshape(1).float()
        m_indptr = torch.tensor([0, mp], dtype=torch.int32, device=DEV)
        out = group_gemm_nvfp4_nt_groupwise(
            a_q, w_packed[i:i + 1], a_sf.reshape(-1), w_sf_mma, m_indptr,
            alpha=alpha, out_dtype=torch.bfloat16,
        )
        outs.append(out[:n])
    return outs


def naive_multigroup_buggy(x_chunks, w_packed, w_sf_full, w_scale_2, N, K):
    """The INTENDED-but-broken layout: concat per-group swizzled SF blocks with
    NO scatter; pass straight to the multi-group kernel."""
    G = len(x_chunks)
    sf_k = _sf_k_of(K)
    n_list = [c.shape[0] for c in x_chunks]
    mpad4 = [_pad4(n) for n in n_list]
    m_indptr_list = [0]
    for mp in mpad4:
        m_indptr_list.append(m_indptr_list[-1] + mp)
    cum_m = m_indptr_list[-1]

    a_q_buf = torch.zeros(cum_m, K // 2, dtype=torch.uint8, device=DEV)
    sf_blocks = []
    a_gs = torch.empty(G, dtype=torch.float32, device=DEV)
    for i in range(G):
        n = n_list[i]; mp = mpad4[i]
        xe = x_chunks[i]
        if mp > n:
            xe = torch.cat([xe, xe.new_zeros(mp - n, K)], 0)
        gs = _quant_gs(xe); a_gs[i] = gs
        a_q, a_sf = flashinfer.fp4_quantize(
            xe, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        a_q_buf[m_indptr_list[i]:m_indptr_list[i] + mp] = a_q.reshape(mp, K // 2)
        sf_blocks.append(a_sf.reshape(-1))   # 128-padded block, simply concatenated
    a_sf_buf = torch.cat(sf_blocks, 0).contiguous()
    m_indptr = torch.tensor(m_indptr_list, dtype=torch.int32, device=DEV)
    alpha = ((1.0 / a_gs) * w_scale_2.float()).reshape(G).float().contiguous()
    out = group_gemm_nvfp4_nt_groupwise(
        a_q_buf, w_packed, a_sf_buf, w_sf_full, m_indptr,
        alpha=alpha, out_dtype=torch.bfloat16,
    )
    return [out[m_indptr_list[i]:m_indptr_list[i] + n_list[i]] for i in range(G)]


def main():
    print("flashinfer", flashinfer.__version__,
          "| device", torch.cuda.get_device_name(0))
    N, K = 256, 256                 # out_n, in_k (multiples of 128/64)
    group_sizes = [100, 50, 200, 30]   # ragged, all < and > 128 -> exercises bug
    G = len(group_sizes)
    print("group_sizes =", group_sizes, " N =", N, " K =", K)
    print("kernel SF offsets per group:",
          [kernel_sf_m_offset(sum(_pad4(g) for g in group_sizes[:i]), i)
           for i in range(G)])
    print("correct 128-padded bases     :",
          [sum(_pad128(g) for g in group_sizes[:i]) for i in range(G)])

    # build per-group weights + activation chunks
    x_chunks, w_bf16_list = [], []
    w_packed = torch.zeros(G, N, K // 2, dtype=torch.uint8, device=DEV)
    w_scale_list, w_scale_2 = [], torch.empty(G, device=DEV)
    for i, g in enumerate(group_sizes):
        x_chunks.append(torch.randn(g, K, device=DEV, dtype=torch.bfloat16) * 0.3)
        w, w_q, _, w_gs, ws2 = make_nvfp4_weight(N, K)
        w_bf16_list.append(w)
        w_packed[i] = w_q.reshape(N, K // 2)
        w_scale_list.append(linear_sf_from_weight(w, w_gs))
        w_scale_2[i] = ws2

    # references
    ref = [dequant_ref(x_chunks[i], w_bf16_list[i]) for i in range(G)]
    per_grp = per_group_ref(x_chunks, w_packed, w_scale_list, w_scale_2, N, K)

    # full weight SF for the multi-group calls
    w_sf_full = build_batched_weight_scale_mma(w_scale_list, N, K)
    bug = naive_multigroup_buggy(x_chunks, w_packed, w_sf_full, w_scale_2, N, K)
    fix = batched_nvfp4_gemm(x_chunks, w_packed, w_sf_full, w_scale_2, N)

    print("\nper-group rel RMS vs bf16-dequant reference:")
    print(f"{'grp':>4} {'n':>5} | {'REF(loop)':>10} {'BUG(concat)':>12} {'FIX(scatter)':>13}")
    worst_fix = 0.0
    worst_ref = 0.0
    bug_blows = False
    for i in range(G):
        r_ref = rel_rms(per_grp[i], ref[i])
        r_bug = rel_rms(bug[i], ref[i])
        r_fix = rel_rms(fix[i], ref[i])
        worst_fix = max(worst_fix, r_fix)
        worst_ref = max(worst_ref, r_ref)
        if r_bug > 0.5:
            bug_blows = True
        print(f"{i:>4} {group_sizes[i]:>5} | {r_ref:>10.4f} {r_bug:>12.4f} {r_fix:>13.4f}")

    # cross-check FIX vs REF directly (should be ~identical scales)
    print("\nFIX vs REF(loop) per-group rel RMS (should be ~0):")
    for i in range(G):
        print(f"  grp {i}: {rel_rms(fix[i], per_grp[i]):.2e}")

    print("\nSUMMARY")
    print(f"  BUG demonstrates failure (some group rel RMS > 0.5): {bug_blows}")
    print(f"  FIX worst-group rel RMS = {worst_fix:.4f}  (REF worst = {worst_ref:.4f})")
    ok = bug_blows and worst_fix < worst_ref * 1.05 + 0.02
    print("  RESULT:", "PASS - bug reproduced AND fixed" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
