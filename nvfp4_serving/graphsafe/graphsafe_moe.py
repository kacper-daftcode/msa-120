# SPDX-License-Identifier: MIT
"""Capture-safe (CUDA-graph-able) un-fused swigluoai NVFP4 MoE for SM120.

Drop-in numeric twin of `unfused_swigluoai_nvfp4_moe` (see
``nvfp4_serving/unfused_moe/unfused_moe.py``) that uses ONLY fixed-shape,
non-host-syncing operations so the model can run WITHOUT ``--enforce-eager``
(i.e. with vLLM piecewise compilation / cudagraph capture enabled).

WHY THE ORIGINAL BREAKS CAPTURE
-------------------------------
The validated per-expert loop dispatches with DATA-DEPENDENT shapes:

  * ``sel.any()``                      -> .item()-style host sync + python branch
  * ``sel.nonzero(as_tuple=True)``     -> output shape depends on tensor *values*
  * ``xe = x[tok]``                    -> variable-length gather (dynamic m)
  * ``mpad = _pad4(m)`` (python int)   -> dynamic per-expert m
  * ``if mpad > m: torch.cat(...)``    -> value-dependent branch + dynamic concat
  * ``torch.tensor([0, mpad], ...)``   -> host tensor build in the hot path
  * ``out[:m]`` / ``index_add_(tok)``  -> dynamic slice / scatter length

All of these invalidate a CUDA-graph capture
(``cudaErrorStreamCaptureInvalidated``) and force eager.

HOW THIS VERSION IS CAPTURE-SAFE
--------------------------------
We replace the nonzero loop with a FIXED-CAPACITY routing table built only with
arithmetic / scatter / cumsum (no nonzero, no .item(), no host tensor in the hot
path):

  * Flatten routing to ``T*k`` (token, expert) slots.  Sort/bucket them into a
    static ``[E, C]`` slot table where ``C`` (expert_capacity) is a COMPILE-TIME
    constant.  Empty cells hold a sentinel that points at a zero pad row.
  * Gather ``x`` into a static ``[E, C, H]`` buffer (pad token = an appended zero
    row at index T).  Every expert ALWAYS processes exactly ``C`` rows, so the
    NVFP4 GEMM, swigluoai, and 2nd GEMM all see static shapes.
  * Per-slot router weights are gathered into ``[E, C]`` with 0.0 for pad slots,
    so padded rows contribute nothing.
  * Scatter the weighted expert outputs back with a single fixed-length
    ``index_add_`` (pad slots target the throw-away row T).

``C`` defaults to the worst case ``pad4(T*k)`` (all tokens to one expert) which
is always correct; pass a tighter static ``expert_capacity`` (e.g. the
decode-time per-expert cap) for speed.  Overflow beyond ``C`` is dropped (cannot
happen at the default), matching the "drop on overflow" semantics of standard
capture-safe MoE; the integrator sizes ``C`` to the captured batch.

The single-group ``group_gemm_nvfp4_nt_groupwise`` call, the activation, and the
alpha math are byte-for-byte the same as the validated path, so numerics match
(the only difference is processing extra all-zero rows, which produce zero
contribution because their router weight is 0).
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
# activation (contiguous halves: gate=first half, up=second half) -- IDENTICAL
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
# quant helpers -- IDENTICAL to the validated path
# --------------------------------------------------------------------------- #
def _quant_gs(t):
    """fp4_quantize *quantization* global scale = (FP4_MAX*E4M3_MAX)/amax."""
    return (FP4_MAX * E4M3_MAX) / t.abs().amax().float().clamp_min(1e-12)


def _pad4(x: int) -> int:
    return (x + 3) // 4 * 4


def weight_scale_to_mma(w_scale_e4m3, N, K):
    """On-disk LINEAR [N, K//16] E4M3 block scale -> swizzle (pad m to 128) ->
    mma 6D layout for one expert (num_groups=1).  Build-time only (not hot)."""
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
    views, one per expert.  Build-time (process_weights_after_loading), not hot."""
    return [weight_scale_to_mma(s, N, K) for s in w_scale_list]


# --------------------------------------------------------------------------- #
# one single-group NVFP4 GEMM on a STATIC-shape buffer -- capture safe
# --------------------------------------------------------------------------- #
def _nvfp4_gemm_one_static(xe, w_packed, w_sf_mma, w_scale_2, m_indptr_full):
    """xe [C, K] bf16 (C is a COMPILE-TIME multiple of 4) -> [C, out_n] bf16 via
    single-group NVFP4 group_gemm.  No padding branch, no torch.tensor build:
    ``m_indptr_full`` ([0, C] int32, device) is precomputed once outside.

    This is exactly ``_nvfp4_gemm_one`` from the validated path with the dynamic
    m-padding and per-call ``torch.tensor`` hoisted out; numerics are identical
    because C is already a multiple of 4 (the only thing the original padding
    guaranteed).
    """
    gs = _quant_gs(xe)
    a_q, a_sf = flashinfer.fp4_quantize(
        xe, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    alpha = ((1.0 / gs) * w_scale_2.float()).reshape(1).float()
    out = group_gemm_nvfp4_nt_groupwise(
        a_q, w_packed.unsqueeze(0), a_sf.reshape(-1), w_sf_mma, m_indptr_full,
        alpha=alpha, out_dtype=torch.bfloat16,
    )
    return out


# --------------------------------------------------------------------------- #
# capture-safe fixed-capacity routing table (no nonzero / item / host tensor)
# --------------------------------------------------------------------------- #
def _build_routing_table(topk_ids, topk_weights, E, C, T):
    """Fixed-shape bucketing of the (token, slot) -> expert routing.

    Returns
    -------
    gather_idx : int64 [E, C]
        For expert e, the source token row for each of its C capacity slots.
        Empty/overflow slots hold ``T`` (a pad row appended to x).
    slot_w     : float32 [E, C]
        Router weight for each slot; 0.0 for empty/overflow slots so they add
        nothing downstream.

    Every op here is fixed-shape: arange, comparison, cumsum, scatter.  No
    ``nonzero``, no ``.item()``, no host ``torch.tensor``.  Shapes depend only on
    E, C, T, k -- all compile-time constants for a captured batch.
    """
    dev = topk_ids.device
    k = topk_ids.shape[1]
    N = T * k  # total (token, slot) routing entries -- static

    flat_e = topk_ids.reshape(N).to(torch.int64)            # [N] expert id per entry
    flat_w = topk_weights.reshape(N).to(torch.float32)      # [N] weight per entry
    flat_tok = (
        torch.arange(T, device=dev, dtype=torch.int64)
        .unsqueeze(1)
        .expand(T, k)
        .reshape(N)
    )                                                       # [N] source token row

    # Per-entry intra-expert position via a stable cumulative count.
    # one_hot[n, e] = 1 if entry n routes to expert e.  [N, E] static.
    one_hot = (flat_e.unsqueeze(1) == torch.arange(E, device=dev).unsqueeze(0)).to(
        torch.int32
    )
    # exclusive prefix sum down the N axis -> rank of this entry within its expert
    # (0-based).  cumsum is capture-safe.
    rank_in_expert = (one_hot.cumsum(0) - one_hot).gather(1, flat_e.unsqueeze(1)).squeeze(1)

    # Linear destination index into the flat [E*C] table; clamp overflow (>=C) to
    # a sink slot E*C (dropped) so the scatter stays in-bounds & fixed-shape.
    valid = rank_in_expert < C
    dest = flat_e * C + rank_in_expert
    sink = E * C
    dest = torch.where(valid, dest, torch.full_like(dest, sink))

    # Scatter token rows & weights into the flat table (+1 sink cell, discarded).
    gather_flat = torch.full((E * C + 1,), T, device=dev, dtype=torch.int64)
    slot_w_flat = torch.zeros((E * C + 1,), device=dev, dtype=torch.float32)
    gather_flat.scatter_(0, dest, flat_tok)
    slot_w_flat.scatter_(0, dest, flat_w)

    gather_idx = gather_flat[:sink].view(E, C)
    slot_w = slot_w_flat[:sink].view(E, C)
    return gather_idx, slot_w


# --------------------------------------------------------------------------- #
# the capture-safe un-fused MoE
# --------------------------------------------------------------------------- #
def graphsafe_swigluoai_nvfp4_moe(
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
    expert_capacity=None,   # static per-expert capacity C (multiple of 4).
):
    """Capture-safe twin of ``unfused_swigluoai_nvfp4_moe``.

    Same signature, same numerics.  ``expert_capacity`` (C) MUST be a constant
    across captured decode steps; if None we use the always-correct worst case
    ``pad4(T*k)``.  For decode-latency capture pass the static per-expert cap.
    """
    T, H = x.shape
    E = w13_packed.shape[0]
    I = w13_packed.shape[1] // 2
    dev = x.device
    k = topk_ids.shape[1]

    if w13_sf_mma is None:
        w13_sf_mma = build_expert_weight_scales_mma(list(w13_scale), 2 * I, H)
    if w2_sf_mma is None:
        w2_sf_mma = build_expert_weight_scales_mma(list(w2_scale), H, I)

    act_fn = (lambda gu: swigluoai(gu, alpha=alpha, limit=limit)) if activation == "swigluoai" \
        else silu_and_mul

    # Static per-expert capacity (compile-time constant). Default = worst case.
    if expert_capacity is None:
        C = _pad4(T * k)
    else:
        C = _pad4(int(expert_capacity))

    # ---- fixed-shape routing table (no nonzero / item / host tensor) --------
    gather_idx, slot_w = _build_routing_table(topk_ids, topk_weights, E, C, T)

    # Pad row T (all zeros) so empty/overflow slots gather a zero token.
    x_pad = torch.cat([x, x.new_zeros(1, H)], 0)            # [T+1, H]

    # Static device m_indptr [0, C].  Built with arange (NO host->device data
    # copy: ``torch.tensor([0, C])`` would memcpy a host list and invalidate a
    # CUDA-graph capture).  C is a python-int constant so ``arange(2) * C`` is a
    # pure-device op yielding [0, C].
    m_indptr_full = torch.arange(2, dtype=torch.int32, device=dev) * C

    # Flat scatter target: pad slots (token T) are routed to a throw-away row.
    out_pad = torch.zeros(T + 1, H, dtype=torch.float32, device=dev)

    for e in range(E):  # E is a compile-time constant -> fully unrolled in capture
        idx_e = gather_idx[e]                               # [C] int64, static
        xe = x_pad.index_select(0, idx_e)                  # [C, H] bf16, static
        gate_up = _nvfp4_gemm_one_static(
            xe, w13_packed[e], w13_sf_mma[e], w13_scale_2[e], m_indptr_full
        )                                                  # [C, 2I]
        inter = act_fn(gate_up).bfloat16()                 # [C, I]
        ye = _nvfp4_gemm_one_static(
            inter, w2_packed[e], w2_sf_mma[e], w2_scale_2[e], m_indptr_full
        )                                                  # [C, H]
        contrib = ye.float() * slot_w[e].unsqueeze(-1)     # [C, H], pad slots = 0
        out_pad.index_add_(0, idx_e, contrib)              # static-length scatter

    return out_pad[:T].to(x.dtype)
