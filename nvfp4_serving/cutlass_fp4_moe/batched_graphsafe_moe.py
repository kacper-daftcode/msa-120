# SPDX-License-Identifier: MIT
"""W4A4 NVFP4 swigluoai MoE for SM120 — batched grouped-GEMM + graph-safe.

This is the production fast path that fuses the two validated building blocks:

  * graphsafe/graphsafe_moe.py — fixed-capacity static ``[E, C]`` routing table
    built with arange/cumsum/scatter (NO ``nonzero`` / ``.item()`` / host
    ``torch.tensor`` in the hot path) so the whole MoE is CUDA-graph capturable.

  * batched_fix/batched_nvfp4_moe.py — ONE multi-group
    ``flashinfer.gemm.group_gemm_nvfp4_nt_groupwise`` call over ALL experts with
    the activation-scale-factor (SFA) SCATTER workaround for the flashinfer
    0.6.12 SM120 multi-group offset bug.

Unlike graphsafe (which loops E single-group GEMMs -> 2*E kernel launches),
this issues exactly TWO grouped-GEMM launches per MoE layer (gate_up + down),
each spanning all E experts. Every expert ALWAYS processes exactly ``C`` rows
(static), so the grouped GEMM, the swigluoai epilogue, the re-quant, and the
down GEMM all see compile-time-constant shapes -> graph-capture safe.

REAL W4A4 FP4 COMPUTE: activations are quantized to FP4 (E2M1 + E4M3 block-16
scale) and fed to the cutlass ``group_gemm_nvfp4`` kernel, which does the matmul
in FP4 (vs marlin, which dequantizes weights to bf16 and matmuls in bf16).

Because every group has the SAME static row count ``C`` (a multiple of 4), the
kernel's per-group activation-SF base row simplifies to a closed form:

    m_indptr[i]   = i * C                      (C already %4 == 0, no pad needed)
    sf_m_off(i)   = ((i*C) + i*127) // 128 * 128   (kernel's buggy-but-fixed offset)

We pre-scatter each group's 128-padded swizzled activation-SF block to
``sf_m_off(i)`` so the kernel reads the correct scales -> bit-identical to the
per-expert single-group loop (proven by repro_multigroup_sfa.py).
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
# activation (contiguous halves: gate=first half, up=second half) — IDENTICAL
# to unfused_moe.swigluoai (the semantic reference).
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
    """fp4_quantize *quantization* global scale = (FP4_MAX*E4M3_MAX)/amax."""
    return (FP4_MAX * E4M3_MAX) / t.abs().amax().float().clamp_min(1e-12)


def _pad4(x: int) -> int:
    return (x + 3) // 4 * 4


def _pad128(x: int) -> int:
    return (x + 127) // 128 * 128


def _sf_k_of(K: int) -> int:
    """Kernel sf_k = round_up(K, 64) / 16 (swizzled_k / sf_vec_size)."""
    return ((K + 63) // 64 * 64) // 16


def kernel_sf_m_offset(m_offset: int, i: int) -> int:
    """Exact replica of the kernel's (buggy) per-group activation-SF base row.

    group_gemm_nvfp4_groupwise_sm120.cuh:
        sf_m_offset = (m_offset + i*(128-1)) // 128 * 128
    """
    return (m_offset + i * (_ALIGN_MN - 1)) // _ALIGN_MN * _ALIGN_MN


def weight_scale_to_mma(w_scale_e4m3, N, K):
    """On-disk LINEAR [N, K//16] E4M3 block scale -> swizzle (pad m to 128) ->
    mma 6D layout for one expert (num_groups=1). Build-time only."""
    sf = w_scale_e4m3.view(torch.uint8)
    Npad = _pad128(N)
    if N < Npad:
        sf = torch.cat(
            [sf, torch.zeros(Npad - N, sf.shape[1], dtype=torch.uint8, device=sf.device)], 0
        )
    sf_sw = flashinfer.nvfp4_block_scale_interleave(sf)
    return convert_sf_to_mma_layout(sf_sw, m=Npad, k=K, num_groups=1)


def build_batched_weight_scale_mma(w_scale_list, N, K):
    """Stack per-expert swizzled weight scales into one contiguous buffer and
    view as the 6D mma layout for num_groups=E. The kernel reads group i at
    i*sf_n*sf_k (correct stride), so this is exact. Build-time only."""
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
# static multi-group plan: precomputed once per (E, C, K) — NOT in the hot path.
# All the python-int offset math (m_indptr, sf scatter offsets) is constant for
# a fixed capacity C, so we compute it once and reuse it every forward.
# --------------------------------------------------------------------------- #
class _GroupPlan:
    __slots__ = ("E", "C", "K", "sf_k", "cum_m", "total_sf_rows",
                 "m_indptr", "sf_off", "a_pack_off",
                 "a_q_buf", "a_sf_buf", "a_gs")

    def __init__(self, E: int, C: int, K: int, device):
        assert C % 4 == 0, "capacity C must be a multiple of 4"
        self.E = E
        self.C = C
        self.K = K
        self.sf_k = _sf_k_of(K)
        # Every group has exactly C rows (already %4). m_indptr = [0, C, 2C, ...].
        m_indptr_list = [i * C for i in range(E + 1)]
        self.cum_m = m_indptr_list[-1]
        self.a_pack_off = m_indptr_list[:E]                 # packed-A row offset
        self.sf_off = [kernel_sf_m_offset(m_indptr_list[i], i) for i in range(E)]
        self.total_sf_rows = self.sf_off[-1] + _pad128(C)
        self.m_indptr = torch.tensor(m_indptr_list, dtype=torch.int32, device=device)
        # PERSISTENT activation buffers (reused every forward, zeroed in place).
        # Because the capacity C is fixed across ALL captured cudagraph sizes,
        # one allocation per plan is shared by every graph -> the per-graph
        # private pools stay tiny (just the GEMM outputs), avoiding the
        # capture-time OOM from re-allocating [E*C, K] buffers per graph size.
        self.a_q_buf = torch.zeros(self.cum_m, K // 2, dtype=torch.uint8, device=device)
        self.a_sf_buf = torch.zeros(self.total_sf_rows * self.sf_k,
                                    dtype=torch.uint8, device=device)
        self.a_gs = torch.empty(E, dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# capture-safe fixed-capacity routing table (verbatim from graphsafe_moe.py)
# --------------------------------------------------------------------------- #
def _build_routing_table(topk_ids, topk_weights, E, C, T):
    dev = topk_ids.device
    k = topk_ids.shape[1]
    N = T * k

    flat_e = topk_ids.reshape(N).to(torch.int64)
    flat_w = topk_weights.reshape(N).to(torch.float32)
    flat_tok = (
        torch.arange(T, device=dev, dtype=torch.int64)
        .unsqueeze(1).expand(T, k).reshape(N)
    )

    one_hot = (flat_e.unsqueeze(1) == torch.arange(E, device=dev).unsqueeze(0)).to(
        torch.int32
    )
    rank_in_expert = (one_hot.cumsum(0) - one_hot).gather(1, flat_e.unsqueeze(1)).squeeze(1)

    valid = rank_in_expert < C
    dest = flat_e * C + rank_in_expert
    sink = E * C
    dest = torch.where(valid, dest, torch.full_like(dest, sink))

    gather_flat = torch.full((E * C + 1,), T, device=dev, dtype=torch.int64)
    slot_w_flat = torch.zeros((E * C + 1,), device=dev, dtype=torch.float32)
    gather_flat.scatter_(0, dest, flat_tok)
    slot_w_flat.scatter_(0, dest, flat_w)

    gather_idx = gather_flat[:sink].view(E, C)
    slot_w = slot_w_flat[:sink].view(E, C)
    return gather_idx, slot_w


# --------------------------------------------------------------------------- #
# core batched grouped GEMM on the static [E, C, K] buffer (SFA scatter fix)
# --------------------------------------------------------------------------- #
def _grouped_nvfp4_static(x_buf, plan: _GroupPlan, w_packed, w_sf_mma, w_scale_2, out_n):
    """x_buf [E, C, K] bf16 -> out [E*C, out_n] bf16 via ONE multi-group GEMM.

    Per-group activation quant global scale a_gs[i] is computed from x_buf[i];
    the FP4 packed activation is placed at the kernel's per-group m_offset and
    the swizzled activation SF is SCATTERED to the kernel's (buggy-but-fixed)
    per-group SF base row.  alpha[i] = (1/a_gs[i]) * w_scale_2[i].
    """
    E, C, K = plan.E, plan.C, plan.K
    sf_k = plan.sf_k

    # Reuse the plan's PERSISTENT buffers (shared across all cudagraph sizes).
    # a_q_buf rows for active groups are fully overwritten below; the SF buffer
    # has gaps (the kernel's per-group offsets are non-contiguous) so it must be
    # zeroed each call (capture-safe in-place op).
    a_q_buf = plan.a_q_buf
    a_sf_buf = plan.a_sf_buf
    a_gs = plan.a_gs
    a_q_buf.zero_()
    a_sf_buf.zero_()

    for i in range(E):
        xe = x_buf[i]                                       # [C, K], static
        gs = _quant_gs(xe)
        a_gs[i] = gs
        a_q, a_sf = flashinfer.fp4_quantize(
            xe, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        x_off = plan.a_pack_off[i]
        a_q_buf[x_off:x_off + C] = a_q.reshape(C, K // 2)
        dst = plan.sf_off[i] * sf_k
        sf_flat = a_sf.reshape(-1)
        a_sf_buf[dst:dst + sf_flat.numel()] = sf_flat

    alpha = ((1.0 / a_gs) * w_scale_2.float()).reshape(E).float().contiguous()
    out = group_gemm_nvfp4_nt_groupwise(
        a_q_buf, w_packed, a_sf_buf, w_sf_mma, plan.m_indptr,
        alpha=alpha, out_dtype=torch.bfloat16,
    )                                                       # [E*C, out_n]
    return out


# --------------------------------------------------------------------------- #
# the batched + graph-safe MoE
# --------------------------------------------------------------------------- #
def batched_graphsafe_swigluoai_nvfp4_moe(
    x,                      # [T, H] bf16 hidden states
    w13_packed,             # [E, 2I, H//2] uint8
    w13_scale,              # (unused if w13_sf_mma given) list E x [2I, H//16]
    w13_scale_2,            # [E] fp32 dequant global scale (gate_up)
    w2_packed,              # [E, H, I//2] uint8
    w2_scale,               # (unused if w2_sf_mma given) list E x [H, I//16]
    w2_scale_2,             # [E] fp32 dequant global scale (down)
    topk_ids,               # [T, k] int
    topk_weights,           # [T, k] float (already renorm)
    *,
    activation="swigluoai",
    alpha=SWIGLU_ALPHA,
    limit=SWIGLU_LIMIT,
    w13_sf_mma=None,        # precomputed 6D mma (num_groups=E) for w13
    w2_sf_mma=None,         # precomputed 6D mma (num_groups=E) for w2
    plan13: _GroupPlan = None,
    plan2: _GroupPlan = None,
    expert_capacity=None,
):
    """Single batched grouped-GEMM per projection, static shapes (graph-safe).

    Numerically identical to unfused_swigluoai_nvfp4_moe / graphsafe twin (the
    only difference is processing extra all-zero rows whose router weight is 0).
    """
    T, H = x.shape
    E = w13_packed.shape[0]
    I = w13_packed.shape[1] // 2
    dev = x.device
    k = topk_ids.shape[1]

    act_fn = (lambda gu: swigluoai(gu, alpha=alpha, limit=limit)) \
        if activation == "swigluoai" else silu_and_mul

    if expert_capacity is None:
        C = _pad4(T * k)
    else:
        C = _pad4(int(expert_capacity))

    if w13_sf_mma is None:
        w13_sf_mma = build_batched_weight_scale_mma(list(w13_scale), 2 * I, H)
    if w2_sf_mma is None:
        w2_sf_mma = build_batched_weight_scale_mma(list(w2_scale), H, I)
    if plan13 is None:
        plan13 = _GroupPlan(E, C, H, dev)
    if plan2 is None:
        plan2 = _GroupPlan(E, C, I, dev)

    # ---- fixed-shape routing table (no nonzero / item / host tensor) --------
    gather_idx, slot_w = _build_routing_table(topk_ids, topk_weights, E, C, T)

    # Pad row T (zeros) so empty/overflow slots gather a zero token.
    x_pad = torch.cat([x, x.new_zeros(1, H)], 0)            # [T+1, H]

    # Static [E, C, H] activation buffer (per-expert capacity rows).
    x_buf = x_pad.index_select(0, gather_idx.reshape(-1)).view(E, C, H)

    # ---- batched gate_up GEMM (ONE grouped call over all E experts) ----
    gate_up = _grouped_nvfp4_static(
        x_buf, plan13, w13_packed, w13_sf_mma, w13_scale_2, 2 * I
    )                                                       # [E*C, 2I]

    # ---- swigluoai epilogue + re-quant + batched down GEMM ----
    inter = act_fn(gate_up).bfloat16().view(E, C, I)        # [E, C, I]
    y = _grouped_nvfp4_static(
        inter, plan2, w2_packed, w2_sf_mma, w2_scale_2, H
    )                                                       # [E*C, H]

    # ---- weighted scatter back to tokens (static-length index_add) ----
    out_pad = torch.zeros(T + 1, H, dtype=torch.float32, device=dev)
    contrib = y.float() * slot_w.reshape(-1).unsqueeze(-1)  # [E*C, H], pad=0
    out_pad.index_add_(0, gather_idx.reshape(-1), contrib)
    return out_pad[:T].to(x.dtype)
