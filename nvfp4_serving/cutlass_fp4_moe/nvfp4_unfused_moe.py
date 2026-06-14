# SPDX-License-Identifier: MIT
"""Standalone un-fused swigluoai NVFP4 MoE for SM120 (RTX PRO 6000).

Correctness-first NVFP4 MoE for MiniMax-M3 on SM120. Replaces the fused
flashinfer b12x MoE (which applies plain SiLU -> wrong for M3) with an un-fused
path: NVFP4 gate_up GEMM -> swigluoai (torch) -> NVFP4 down GEMM, using
`flashinfer.gemm.group_gemm_nvfp4_nt_groupwise` ONE single-group call per routed
expert (the doc-endorsed B2a "per-expert loop").

Verified contract (see probe_real_w.py / debug17.py): a single-group
group_gemm_nvfp4_nt_groupwise matches a dequant->bf16 reference at the NVFP4
noise floor (~0.10), and the per-expert-loop full MoE matches the bf16 swigluoai
reference at ~0.133 rel RMS.

  * activation quantized with `flashinfer.fp4_quantize(is_sf_swizzled_layout
    =True)` per expert; the swizzled scale factors are passed flattened as
    a_scale (single group, m padded to a multiple of 4).
  * weight block-scales (on-disk LINEAR E4M3) -> `nvfp4_block_scale_interleave`
    (swizzle, m padded to 128) -> `convert_sf_to_mma_layout(num_groups=1)`.
  * alpha = 1/(a_gs) * w_scale_2  (a_gs = fp4_quantize quant global scale,
    w_scale_2 = checkpoint dequant global scale).

NOTE: the multi-group form of group_gemm_nvfp4_nt_groupwise was found to
miscompute SFA for groups >= 2 in this flashinfer build (0.6.12); the
per-expert single-group loop sidesteps that and is numerically exact.  A future
fast path can batch this once that kernel's grouped-SFA offset is fixed.
"""
from __future__ import annotations
import torch
import flashinfer
from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

FP4_MAX = 6.0
E4M3_MAX = 448.0
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0


# --------------------------------------------------------------------------- #
# activation (contiguous halves: gate=first half, up=second half)
# --------------------------------------------------------------------------- #
def swigluoai(gate_up, *, alpha=SWIGLU_ALPHA, limit=SWIGLU_LIMIT):
    """out = (clamp(up,-limit,+limit)+1)*clamp(gate,max=limit)*sigmoid(alpha*gate)."""
    d = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :d], gate_up[..., d:]
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    return (up_c + 1.0) * (gate_c * torch.sigmoid(alpha * gate_c))


def silu_and_mul(gate_up):
    d = gate_up.shape[-1] // 2
    return torch.nn.functional.silu(gate_up[..., :d]) * gate_up[..., d:]


# --------------------------------------------------------------------------- #
# quant helpers
# --------------------------------------------------------------------------- #
def _quant_gs(t):
    """fp4_quantize *quantization* global scale = (FP4_MAX*E4M3_MAX)/amax."""
    return (FP4_MAX * E4M3_MAX) / t.abs().amax().float().clamp_min(1e-12)


def _pad4(x):
    return (x + 3) // 4 * 4


def weight_scale_to_mma(w_scale_e4m3, N, K):
    """On-disk LINEAR [N, K//16] E4M3 block scale -> swizzle (pad m to 128) ->
    mma 6D layout for one expert (num_groups=1)."""
    sf = w_scale_e4m3.view(torch.uint8)
    Npad = ((N + 127) // 128) * 128
    if N < Npad:
        sf = torch.cat(
            [sf, torch.zeros(Npad - N, sf.shape[1], dtype=torch.uint8, device=sf.device)], 0
        )
    sf_sw = flashinfer.nvfp4_block_scale_interleave(sf)
    return convert_sf_to_mma_layout(sf_sw, m=Npad, k=K, num_groups=1)


def build_expert_weight_scales_mma(w_scale_list, N, K):
    """Per-expert mma weight scales (list of [N, K//16] E4M3) -> list of 6D mma
    views, one per expert.  Mirrors FlashInferB12xExperts.w*_sf_mma but kept
    per-expert for the single-group loop."""
    return [weight_scale_to_mma(s, N, K) for s in w_scale_list]


# --------------------------------------------------------------------------- #
# one single-group NVFP4 GEMM (out = x @ w^T, NT) for one expert
# --------------------------------------------------------------------------- #
def _nvfp4_gemm_one(xe, w_packed, w_sf_mma, w_scale_2, out_n):
    """xe [m, K] bf16 -> [m, out_n] bf16 via NVFP4 group_gemm (single group).
    w_packed [out_n, K//2] uint8, w_sf_mma 6D mma scale, w_scale_2 scalar global.
    """
    m, K = xe.shape
    mpad = _pad4(m)
    if mpad > m:
        xe = torch.cat(
            [xe, xe.new_zeros(mpad - m, K)], 0
        )
    gs = _quant_gs(xe)
    a_q, a_sf = flashinfer.fp4_quantize(
        xe, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    alpha = ((1.0 / gs) * w_scale_2.float()).reshape(1).float()
    m_indptr = torch.tensor([0, mpad], dtype=torch.int32, device=xe.device)
    out = group_gemm_nvfp4_nt_groupwise(
        a_q, w_packed.unsqueeze(0), a_sf.reshape(-1), w_sf_mma, m_indptr,
        alpha=alpha, out_dtype=torch.bfloat16,
    )
    return out[:m]


# --------------------------------------------------------------------------- #
# the un-fused MoE (per-expert loop)
# --------------------------------------------------------------------------- #
def unfused_swigluoai_nvfp4_moe(
    x,                      # [T, H] bf16 hidden states
    w13_packed,             # [E, 2I, H//2] uint8  (gate;up stacked, contiguous)
    w13_scale,              # list E x [2I, H//16] E4M3 (on-disk linear)
    w13_scale_2,            # [E] fp32 dequant global scale (gate_up)
    w2_packed,              # [E, H, I//2] uint8   (down)
    w2_scale,               # list E x [H, I//16] E4M3
    w2_scale_2,             # [E] fp32 dequant global scale (down)
    topk_ids,               # [T, k] int
    topk_weights,           # [T, k] float (already renorm * routed_scaling_factor)
    *,
    activation="swigluoai",
    alpha=SWIGLU_ALPHA,
    limit=SWIGLU_LIMIT,
    w13_sf_mma=None,        # optional precomputed list of E mma views
    w2_sf_mma=None,
):
    T, H = x.shape
    E = w13_packed.shape[0]
    I = w13_packed.shape[1] // 2
    dev = x.device

    if w13_sf_mma is None:
        w13_sf_mma = build_expert_weight_scales_mma(list(w13_scale), 2 * I, H)
    if w2_sf_mma is None:
        w2_sf_mma = build_expert_weight_scales_mma(list(w2_scale), H, I)

    act_fn = (lambda gu: swigluoai(gu, alpha=alpha, limit=limit)) if activation == "swigluoai" \
        else silu_and_mul

    out = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for e in range(E):
        sel = (topk_ids == e)
        if not sel.any():
            continue
        tok, slot = sel.nonzero(as_tuple=True)
        xe = x[tok]                                              # [n, H]
        gate_up = _nvfp4_gemm_one(xe, w13_packed[e], w13_sf_mma[e], w13_scale_2[e], 2 * I)
        inter = act_fn(gate_up).bfloat16()                       # [n, I]
        ye = _nvfp4_gemm_one(inter, w2_packed[e], w2_sf_mma[e], w2_scale_2[e], H)
        w = topk_weights[tok, slot].float().unsqueeze(-1)
        out.index_add_(0, tok, ye.float() * w)
    return out.to(x.dtype)
