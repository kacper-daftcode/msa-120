# SPDX-License-Identifier: MIT
"""Batched (single grouped-GEMM) swigluoai NVFP4 MoE for SM120 (RTX PRO 6000).

Drop-in replacement for ``unfused_moe.unfused_swigluoai_nvfp4_moe`` that issues
ONE ``flashinfer.gemm.group_gemm_nvfp4_nt_groupwise`` call over all routed
experts instead of a per-expert Python loop.

It works around the multi-group activation-scale-factor (SFA) offset bug in
flashinfer 0.6.12's SM120 NVFP4 grouped GEMM (see ROOTCAUSE.md). The kernel
computes each group's activation-SF base as the *deterministic but wrong*

    kernel_sf_m_offset(i) = ((m_indptr[i] + i*127) // 128) * 128   (swizzled rows)

For i>=2 this over-advances by floor(127*i/128) extra 128-row blocks. Because
those gaps are deterministic and always >= each group's real 128-padded SF
block, we *pre-scatter* the per-group swizzled scale-factor blocks into a buffer
at exactly those offsets. The kernel then reads each group's SF from the correct
rows, giving bit-identical scales to the validated per-expert loop.

The activation FP4 data (`a`), output (`D`) and *weight* scale use the kernel's
correct per-group strides (m_offset for A/D, i*sf_n*sf_k for weight SF), so only
the activation SF needs the scatter workaround.
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

_ALIGN_MN = 128  # alignment_swizzled_mn in the kernel


# --------------------------------------------------------------------------- #
# activation
# --------------------------------------------------------------------------- #
def swigluoai(gate_up, *, alpha=SWIGLU_ALPHA, limit=SWIGLU_LIMIT):
    d = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :d], gate_up[..., d:]
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    return (up_c + 1.0) * (gate_c * torch.sigmoid(alpha * gate_c))


def silu_and_mul(gate_up):
    d = gate_up.shape[-1] // 2
    return torch.nn.functional.silu(gate_up[..., :d]) * gate_up[..., d:]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _quant_gs(t):
    return (FP4_MAX * E4M3_MAX) / t.abs().amax().float().clamp_min(1e-12)


def _pad4(x):
    return (x + 3) // 4 * 4


def _pad128(x):
    return (x + 127) // 128 * 128


def _sf_k_of(K):
    """Kernel sf_k = round_up(K, 64) / 16 (swizzled_k / sf_vec_size)."""
    return ((K + 63) // 64 * 64) // 16


def kernel_sf_m_offset(m_offset: int, i: int) -> int:
    """Exact replica of the kernel's (buggy) per-group activation-SF base row.

    group_gemm_nvfp4_groupwise_sm120.cuh, lines 74-76:
        sf_m_offset = (m_offset + i*(128-1)) // 128 * 128
    """
    return (m_offset + i * (_ALIGN_MN - 1)) // _ALIGN_MN * _ALIGN_MN


def weight_scale_to_mma(w_scale_e4m3, N, K):
    """On-disk LINEAR [N, K//16] E4M3 block scale -> swizzle (pad m to 128) ->
    mma 6D layout for one expert (num_groups=1). Bit-identical to the per-expert
    path."""
    sf = w_scale_e4m3.view(torch.uint8)
    Npad = _pad128(N)
    if N < Npad:
        sf = torch.cat(
            [sf, torch.zeros(Npad - N, sf.shape[1], dtype=torch.uint8, device=sf.device)], 0
        )
    sf_sw = flashinfer.nvfp4_block_scale_interleave(sf)
    return convert_sf_to_mma_layout(sf_sw, m=Npad, k=K, num_groups=1)


def build_batched_weight_scale_mma(w_scale_list, N, K):
    """Stack per-expert swizzled weight scales into one contiguous (E, m_tiles,
    k_tiles, 32,4,4) buffer and view as the 6D mma layout for num_groups=E. The
    kernel reads group i at i*sf_n*sf_k (correct stride), so this is exact."""
    Npad = _pad128(N)
    blocks = []
    for s in w_scale_list:
        sf = s.view(torch.uint8)
        if sf.shape[0] < Npad:
            sf = torch.cat(
                [sf, torch.zeros(Npad - sf.shape[0], sf.shape[1],
                                 dtype=torch.uint8, device=sf.device)], 0
            )
        blocks.append(flashinfer.nvfp4_block_scale_interleave(sf).reshape(-1))
    flat = torch.cat(blocks, 0).contiguous()
    return convert_sf_to_mma_layout(flat, m=Npad, k=K, num_groups=len(w_scale_list))


# --------------------------------------------------------------------------- #
# core: one batched NVFP4 grouped GEMM with SFA scatter workaround
# --------------------------------------------------------------------------- #
def batched_nvfp4_gemm(
    x_chunks,        # list[Tensor [n_g, K] bf16]  (one per group, n_g may be 0-skipped upstream)
    w_packed,        # [G, out_n, K//2] uint8 (E2M1)
    w_sf_mma,        # 6D mma weight scales for G groups (build_batched_weight_scale_mma)
    w_scale_2,       # [G] fp32 dequant global scale
    out_n,
):
    """Compute, for each group g, out_g = swigluoai-free NVFP4 GEMM
    x_chunks[g] @ w_packed[g]^T (NT), batched in a SINGLE grouped-GEMM call.

    Returns list[Tensor [n_g, out_n] bf16], one per group (unpadded)."""
    assert len(x_chunks) == w_packed.shape[0]
    G = len(x_chunks)
    dev = x_chunks[0].device
    K = x_chunks[0].shape[1]
    sf_k = _sf_k_of(K)

    n_list = [c.shape[0] for c in x_chunks]
    mpad4 = [_pad4(n) for n in n_list]
    m_indptr_list = [0]
    for mp in mpad4:
        m_indptr_list.append(m_indptr_list[-1] + mp)
    cum_m = m_indptr_list[-1]

    sf_off = [kernel_sf_m_offset(m_indptr_list[i], i) for i in range(G)]
    total_sf_rows = sf_off[-1] + _pad128(n_list[-1])

    a_q_buf = torch.zeros(cum_m, K // 2, dtype=torch.uint8, device=dev)
    a_sf_buf = torch.zeros(total_sf_rows * sf_k, dtype=torch.uint8, device=dev)
    a_gs = torch.empty(G, dtype=torch.float32, device=dev)

    for i in range(G):
        n = n_list[i]
        xe = x_chunks[i]
        mp4 = mpad4[i]
        if mp4 > n:
            xe = torch.cat([xe, xe.new_zeros(mp4 - n, K)], 0)
        gs = _quant_gs(xe)
        a_gs[i] = gs
        a_q, a_sf = flashinfer.fp4_quantize(
            xe, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        # FP4 packed activation: kernel uses m_offset stride -> place at m_indptr.
        x_off = m_indptr_list[i]
        a_q_buf[x_off:x_off + mp4] = a_q.reshape(mp4, K // 2)
        # Activation SF: scatter the 128-padded swizzled block to the kernel's
        # (buggy-but-deterministic) per-group offset.
        blk_rows = _pad128(n)
        sf_flat = a_sf.reshape(-1)
        dst = sf_off[i] * sf_k
        a_sf_buf[dst:dst + sf_flat.numel()] = sf_flat

    m_indptr = torch.tensor(m_indptr_list, dtype=torch.int32, device=dev)
    alpha = ((1.0 / a_gs) * w_scale_2.float()).reshape(G).float().contiguous()

    out = group_gemm_nvfp4_nt_groupwise(
        a_q_buf, w_packed, a_sf_buf, w_sf_mma, m_indptr,
        alpha=alpha, out_dtype=torch.bfloat16,
    )  # [cum_m, out_n]

    res = []
    for i in range(G):
        x_off = m_indptr_list[i]
        res.append(out[x_off:x_off + n_list[i]])
    return res


# --------------------------------------------------------------------------- #
# the batched MoE (single grouped GEMM per projection over ACTIVE experts)
# --------------------------------------------------------------------------- #
def batched_swigluoai_nvfp4_moe(
    x,                      # [T, H] bf16 hidden states
    w13_packed,             # [E, 2I, H//2] uint8
    w13_scale,              # list E x [2I, H//16] E4M3 (on-disk linear)
    w13_scale_2,            # [E] fp32
    w2_packed,              # [E, H, I//2] uint8
    w2_scale,               # list E x [H, I//16] E4M3
    w2_scale_2,             # [E] fp32
    topk_ids,               # [T, k] int
    topk_weights,           # [T, k] float (already renorm * routed_scaling_factor)
    *,
    activation="swigluoai",
    alpha=SWIGLU_ALPHA,
    limit=SWIGLU_LIMIT,
    w13_sf_mma_full=None,   # optional precomputed 6D mma (num_groups=E) for w13
    w2_sf_mma_full=None,    # optional precomputed 6D mma (num_groups=E) for w2
):
    """Same signature/semantics as unfused_swigluoai_nvfp4_moe, one batched call
    per projection. Only experts that are actually routed-to participate (active
    subset), keeping the grouped GEMM dense."""
    T, H = x.shape
    E = w13_packed.shape[0]
    I = w13_packed.shape[1] // 2
    dev = x.device

    act_fn = (lambda gu: swigluoai(gu, alpha=alpha, limit=limit)) \
        if activation == "swigluoai" else silu_and_mul

    # token routing per expert
    tok_lists, slot_lists, active = [], [], []
    for e in range(E):
        sel = (topk_ids == e)
        if not sel.any():
            continue
        tok, slot = sel.nonzero(as_tuple=True)
        tok_lists.append(tok)
        slot_lists.append(slot)
        active.append(e)

    out = torch.zeros(T, H, dtype=torch.float32, device=dev)
    if not active:
        return out.to(x.dtype)

    # weight scales for active experts (6D mma stacked along groups)
    w13_sf = build_batched_weight_scale_mma([w13_scale[e].view(torch.uint8) for e in active],
                                            2 * I, H)
    w2_sf = build_batched_weight_scale_mma([w2_scale[e].view(torch.uint8) for e in active],
                                           H, I)

    # ---- batched gate_up GEMM ----
    x_chunks = [x[tok] for tok in tok_lists]
    gate_up_chunks = batched_nvfp4_gemm(
        x_chunks,
        w13_packed[active],
        w13_sf,
        w13_scale_2[active],
        2 * I,
    )

    # ---- swigluoai + re-quant + batched down GEMM ----
    inter_chunks = [act_fn(gu).bfloat16() for gu in gate_up_chunks]
    y_chunks = batched_nvfp4_gemm(
        inter_chunks,
        w2_packed[active],
        w2_sf,
        w2_scale_2[active],
        H,
    )

    for i, e in enumerate(active):
        tok = tok_lists[i]
        slot = slot_lists[i]
        w = topk_weights[tok, slot].float().unsqueeze(-1)
        out.index_add_(0, tok, y_chunks[i].float() * w)

    return out.to(x.dtype)
