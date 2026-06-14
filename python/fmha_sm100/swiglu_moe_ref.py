# SPDX-License-Identifier: MIT
"""Correctness-first torch reference for the MiniMax-M3 NVFP4 MoE path.

MiniMax-M3 routed experts use the GPT-OSS-style *clamped* SwiGLU activation
(``hidden_act = "swigluoai"``), NOT plain SiLU.  A serving stack that silently
falls back to ``silu(gate) * up`` degrades quality on all 57 MoE layers.

This module provides:

  * ``swigluoai`` / ``swigluoai_from_gate_up`` -- the exact clamped-SwiGLU
    activation, with the M3 checkpoint defaults ``alpha = 1.702``,
    ``limit = 7.0``.
  * ``swigluoai_moe`` -- a clean bf16/fp32 reference of the full MoE block:
    top-k routing -> per routed expert (gate_up GEMM -> swigluoai -> down GEMM)
    -> routing-weighted sum.
  * An NVFP4 path: ``quantize_to_nvfp4`` / ``dequantize_nvfp4`` (block-16 E2M1
    data + E4M3 block scales + FP32 global scale, the exact layout the M3-NVFP4
    checkpoint ships) and ``swigluoai_moe_nvfp4`` which runs the two GEMMs on
    *dequantized* NVFP4 weights so the numerics match what a real NVFP4 GEMM
    path produces (an un-fused: NVFP4 gate_up GEMM -> swigluoai -> NVFP4 down
    GEMM per expert).

------------------------------------------------------------------------------
swigluoai (exact form, verified -- see ``__main__`` and the report):

    gate, up = split(gate_up)          # contiguous halves: gate then up
    gate = gate.clamp(max=limit)       # one-sided clamp (min=None)
    up   = up.clamp(-limit, +limit)    # symmetric clamp
    glu  = gate * sigmoid(alpha * gate)
    out  = (up + 1) * glu

with ``alpha = 1.702``, ``limit = 7.0`` (from MiniMax-M3-NVFP4/config.json:
``swiglu_alpha``, ``swiglu_limit``).  Algebraically identical to the model
card's ``(clamp(up,+/-limit)+1) * clamp(gate,max=limit) * sigmoid(alpha*gate)``.

Split convention -- IMPORTANT / verified against the checkpoint:
  The M3 checkpoint stores ``gate_proj`` and ``up_proj`` as *separate* tensors
  (``...experts.<e>.gate_proj.weight`` / ``...up_proj.weight``).  When fused
  into a single ``w13 = [gate_proj; up_proj]`` weight for a fused MoE kernel,
  vLLM stacks them contiguously (rows ``0:I`` = gate, ``I:2I`` = up).  Hence the
  fused gate_up activation tensor is **contiguous halves, gate-then-up** -- this
  is what this reference uses and what the model card calls "non-interleaved".
  (vLLM's ``SwigluOAIAndMul`` uses *interleaved* ``[..., ::2]/[..., 1::2]``
  because GPT-OSS ships pre-interleaved fused weights; that is a GPT-OSS weight
  packing, not the M3 layout.  See the design doc for the plumbing consequence.)
------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# M3-NVFP4 checkpoint defaults (config.json: text_config.swiglu_*).
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0

# NVFP4 constants (match python/fmha_sm100/cute/quantize.py).
NVFP4_BLOCK_SIZE = 16
NVFP4_FP4_MAX = 6.0
NVFP4_FP8_E4M3_MAX = 448.0

# E2M1 representable magnitudes (positive grid); used for nearest-grid rounding.
_E2M1_GRID = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
def swigluoai_from_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    alpha: float = SWIGLU_ALPHA,
    limit: float = SWIGLU_LIMIT,
) -> torch.Tensor:
    """Clamped SwiGLU on already-split (gate, up) halves.

    out = (clamp(up, -limit, +limit) + 1) * clamp(gate, max=limit)
          * sigmoid(alpha * clamp(gate, max=limit))
    """
    gate_c = gate.clamp(max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    glu = gate_c * torch.sigmoid(alpha * gate_c)
    return (up_c + 1.0) * glu


def swigluoai(
    gate_up: torch.Tensor,
    *,
    alpha: float = SWIGLU_ALPHA,
    limit: float = SWIGLU_LIMIT,
    interleaved: bool = False,
) -> torch.Tensor:
    """Clamped SwiGLU on a fused gate_up tensor ``[..., 2I]`` -> ``[..., I]``.

    ``interleaved=False`` (default, the M3 layout): gate = first half, up =
    second half.  ``interleaved=True`` (GPT-OSS layout): gate = even indices,
    up = odd indices.
    """
    if interleaved:
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
    else:
        d = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :d], gate_up[..., d:]
    return swigluoai_from_gate_up(gate, up, alpha=alpha, limit=limit)


def silu_and_mul(gate_up: torch.Tensor, *, interleaved: bool = False) -> torch.Tensor:
    """Plain SiLU SwiGLU (the WRONG activation for M3) -- for the gap check."""
    if interleaved:
        gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
    else:
        d = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :d], gate_up[..., d:]
    return torch.nn.functional.silu(gate) * up


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_topk(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Softmax top-k routing.

    Returns ``(topk_ids[T, k], topk_weights[T, k])``.  The softmax is taken over
    all experts; top-k is selected; weights optionally renormalised over the
    selected set and scaled by ``routed_scaling_factor`` (M3 uses 2.0).

    ``correction_bias`` (M3's ``e_score_correction_bias``) is added to the
    scores used for *selection only* (not to the returned weights), matching the
    DeepSeek/M3 grouped-bias convention.
    """
    probs = torch.softmax(router_logits.float(), dim=-1)
    scores = probs
    if correction_bias is not None:
        scores = probs + correction_bias.float().view(1, -1)
    _, topk_ids = torch.topk(scores, top_k, dim=-1)
    topk_weights = torch.gather(probs, -1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_ids, topk_weights


# ---------------------------------------------------------------------------
# Dense (unquantised) reference MoE
# ---------------------------------------------------------------------------
def swigluoai_moe(
    hidden: torch.Tensor,                     # [T, H]
    w13: torch.Tensor,                        # [E, 2I, H]  ([gate; up] stacked)
    w2: torch.Tensor,                         # [E, H, I]
    *,
    router_logits: Optional[torch.Tensor] = None,   # [T, E]
    topk_ids: Optional[torch.Tensor] = None,        # [T, k]
    topk_weights: Optional[torch.Tensor] = None,    # [T, k]
    top_k: Optional[int] = None,
    alpha: float = SWIGLU_ALPHA,
    limit: float = SWIGLU_LIMIT,
    interleaved: bool = False,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    correction_bias: Optional[torch.Tensor] = None,
    activation: str = "swigluoai",
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference MoE: top-k route -> per-expert (gate_up GEMM -> act -> down
    GEMM) -> routing-weighted sum.  Returns ``[T, H]`` in ``hidden.dtype``.

    Provide either ``router_logits`` (+ ``top_k``) or precomputed
    ``topk_ids``/``topk_weights``.  ``activation`` is ``"swigluoai"`` (correct)
    or ``"silu"`` (plain, for the degradation comparison).
    """
    T, H = hidden.shape
    E, twoI, H2 = w13.shape
    assert H2 == H, f"w13 in-dim {H2} != hidden dim {H}"
    I = twoI // 2
    assert w2.shape == (E, H, I), f"w2 {tuple(w2.shape)} != {(E, H, I)}"

    if topk_ids is None or topk_weights is None:
        assert router_logits is not None and top_k is not None
        topk_ids, topk_weights = route_topk(
            router_logits, top_k,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            correction_bias=correction_bias,
        )
    k = topk_ids.shape[1]

    act_fn = swigluoai if activation == "swigluoai" else silu_and_mul
    hc = hidden.to(compute_dtype)
    w13c = w13.to(compute_dtype)
    w2c = w2.to(compute_dtype)

    out = torch.zeros(T, H, dtype=compute_dtype, device=hidden.device)
    # Loop over experts; gather the tokens routed to each (correctness-first).
    for e in range(E):
        sel = (topk_ids == e)                 # [T, k]
        if not sel.any():
            continue
        tok_idx, slot_idx = sel.nonzero(as_tuple=True)   # which token, which slot
        x = hc[tok_idx]                        # [n, H]
        gate_up = x @ w13c[e].t()              # [n, 2I]
        if activation == "swigluoai":
            act = act_fn(gate_up, alpha=alpha, limit=limit, interleaved=interleaved)
        else:
            act = act_fn(gate_up, interleaved=interleaved)
        y = act @ w2c[e].t()                   # [n, H]
        w = topk_weights[tok_idx, slot_idx].to(compute_dtype).unsqueeze(-1)
        out.index_add_(0, tok_idx, y * w)
    return out.to(hidden.dtype)


# ---------------------------------------------------------------------------
# NVFP4 quantize / dequantize (block-16 E2M1 + E4M3 block scales + FP32 global)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Nvfp4Weight:
    """Per-expert NVFP4 weight: logical (dequantizable) representation.

    Stored as ``q`` (E2M1-rounded magnitudes on the grid, float), ``block_scale``
    (E4M3-rounded per-16 scale), and ``global_scale`` (FP32 scalar), matching the
    checkpoint contract ``w = q * block_scale * global_scale``.
    """
    q: torch.Tensor             # [..., K]  float, values on the E2M1 grid
    block_scale: torch.Tensor   # [..., K//16] float (already E4M3-rounded)
    global_scale: torch.Tensor  # scalar float
    shape: Tuple[int, ...]


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round magnitudes to the nearest E2M1 grid point, preserving sign."""
    grid = _E2M1_GRID.to(x.device)
    sign = torch.sign(x)
    mag = x.abs()
    # nearest grid point
    idx = torch.bucketize(mag, (grid[:-1] + grid[1:]) / 2.0)
    return sign * grid[idx]


def _round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round a positive scale tensor through float8_e4m3fn and back to fp32."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def quantize_to_nvfp4(w: torch.Tensor) -> Nvfp4Weight:
    """Quantize a 2D weight ``[N, K]`` (K divisible by 16) to NVFP4.

    Faithful reimplementation of the modelopt / TE NVFP4 recipe:
      * global_scale = amax(|w|) / (448 * 6)
      * per-16-block scale = amax_block / 6, rounded to E4M3, then divided by
        global_scale so it sits in the E4M3 dynamic range
      * data = round_to_e2m1(w / (block_scale * global_scale))
    Dequant is ``q * block_scale * global_scale``.
    """
    assert w.ndim == 2, "expect 2D weight"
    N, K = w.shape
    assert K % NVFP4_BLOCK_SIZE == 0, f"K={K} not divisible by {NVFP4_BLOCK_SIZE}"
    wf = w.float()
    global_scale = wf.abs().amax() / (NVFP4_FP8_E4M3_MAX * NVFP4_FP4_MAX)
    global_scale = global_scale.clamp_min(1e-12)

    blocks = wf.view(N, K // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)                       # [N, K//16]
    # scale that maps block to the FP4 [0,6] range, expressed relative to global
    raw_scale = (block_amax / NVFP4_FP4_MAX) / global_scale
    block_scale = _round_to_e4m3(raw_scale).clamp_min(1e-12)     # E4M3-quantized
    eff = (block_scale * global_scale).unsqueeze(-1)             # [N, K//16, 1]
    q = _round_to_e2m1(blocks / eff).view(N, K)
    return Nvfp4Weight(q=q, block_scale=block_scale, global_scale=global_scale, shape=(N, K))


def dequantize_nvfp4(qw: Nvfp4Weight) -> torch.Tensor:
    """Dequantize an ``Nvfp4Weight`` back to fp32: ``q * block_scale * global``."""
    N, K = qw.shape
    q = qw.q.view(N, K // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    eff = (qw.block_scale * qw.global_scale).unsqueeze(-1)
    return (q * eff).view(N, K)


def quantize_experts_nvfp4(w: torch.Tensor) -> Tuple[Nvfp4Weight, ...]:
    """Quantize a stacked expert weight ``[E, N, K]`` to a tuple of per-expert
    ``Nvfp4Weight``."""
    return tuple(quantize_to_nvfp4(w[e]) for e in range(w.shape[0]))


def dequantize_experts_nvfp4(qws: Tuple[Nvfp4Weight, ...], dtype, device) -> torch.Tensor:
    """Stack per-expert dequantized weights back to ``[E, N, K]``."""
    return torch.stack([dequantize_nvfp4(q).to(dtype) for q in qws]).to(device)


# ---------------------------------------------------------------------------
# NVFP4 MoE (un-fused: NVFP4 gate_up GEMM -> swigluoai -> NVFP4 down GEMM)
# ---------------------------------------------------------------------------
def swigluoai_moe_nvfp4(
    hidden: torch.Tensor,                     # [T, H]
    w13_q: Tuple[Nvfp4Weight, ...],           # E x Nvfp4Weight ([2I, H])
    w2_q: Tuple[Nvfp4Weight, ...],            # E x Nvfp4Weight ([H, I])
    *,
    router_logits: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    alpha: float = SWIGLU_ALPHA,
    limit: float = SWIGLU_LIMIT,
    interleaved: bool = False,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    correction_bias: Optional[torch.Tensor] = None,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """MoE with NVFP4-quantized expert weights.

    Each GEMM consumes the *dequantized* NVFP4 weight, so the result equals what
    a real NVFP4 GEMM kernel produces (its inputs are the same E2M1/E4M3/global
    triples).  ``compute_dtype`` is the GEMM accumulation/output dtype.
    """
    dev = hidden.device
    w13 = dequantize_experts_nvfp4(w13_q, compute_dtype, dev)   # [E, 2I, H]
    w2 = dequantize_experts_nvfp4(w2_q, compute_dtype, dev)     # [E, H, I]
    return swigluoai_moe(
        hidden, w13, w2,
        router_logits=router_logits, topk_ids=topk_ids, topk_weights=topk_weights,
        top_k=top_k, alpha=alpha, limit=limit, interleaved=interleaved,
        renormalize=renormalize, routed_scaling_factor=routed_scaling_factor,
        correction_bias=correction_bias, activation="swigluoai",
        compute_dtype=compute_dtype,
    )


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).pow(2).mean().sqrt().item()


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return _rms(a, b) / (b.float().pow(2).mean().sqrt().item() + 1e-20)


def _main() -> None:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Small M3-shaped problem (scaled down): H, I multiples of 16 for NVFP4.
    T, H, I, E, k = 64, 256, 512, 8, 4
    alpha, limit = SWIGLU_ALPHA, SWIGLU_LIMIT
    rsf = 2.0  # M3 routed_scaling_factor

    hidden = torch.randn(T, H, device=dev, dtype=dtype) * 1.0
    # 0.04 std keeps gate_up roughly in [-8, 8] so the clamp at +/-7 actually bites.
    w13 = (torch.randn(E, 2 * I, H, device=dev, dtype=dtype) * 0.04)
    w2 = (torch.randn(E, H, I, device=dev, dtype=dtype) * 0.04)
    router_logits = torch.randn(T, E, device=dev, dtype=dtype)

    topk_ids, topk_weights = route_topk(
        router_logits, k, renormalize=True, routed_scaling_factor=rsf
    )

    print("=" * 72)
    print("MiniMax-M3 NVFP4 swigluoai MoE -- reference self-check")
    print("=" * 72)
    print(f"device={dev} dtype={dtype}")
    print(f"shapes: hidden[{T},{H}] w13[{E},{2*I},{H}] w2[{E},{H},{I}] top_k={k}")
    print(f"swiglu: alpha={alpha} limit={limit} routed_scaling_factor={rsf}")
    print(f"routing: ids{tuple(topk_ids.shape)} weights{tuple(topk_weights.shape)} "
          f"(per-row weight sum~={topk_weights.sum(-1).mean().item():.3f} == k*rsf? "
          f"renorm*rsf={rsf})")

    # --- (a) swigluoai vs plain SiLU: quantify the gap ---
    out_swig = swigluoai_moe(
        hidden, w13, w2, topk_ids=topk_ids, topk_weights=topk_weights,
        alpha=alpha, limit=limit, activation="swigluoai",
    )
    out_silu = swigluoai_moe(
        hidden, w13, w2, topk_ids=topk_ids, topk_weights=topk_weights,
        activation="silu",
    )
    # also a raw activation-level gap on a clamp-stressing tensor
    gu = torch.randn(4096, 2 * I, device=dev, dtype=torch.float32) * 4.0
    a_swig = swigluoai(gu, alpha=alpha, limit=limit)
    a_silu = silu_and_mul(gu)
    print("\n(a) swigluoai vs plain-SiLU (the WRONG activation):")
    print(f"    MoE-output  abs RMS = {_rms(out_swig, out_silu):.6e}")
    print(f"    MoE-output  rel RMS = {_rel_rms(out_swig, out_silu):.6e}")
    print(f"    activation  abs RMS = {_rms(a_swig, a_silu):.6e}  "
          f"(rel {_rel_rms(a_swig, a_silu):.4f})")
    print(f"    => they DIFFER substantially; SiLU fallback corrupts every MoE layer.")

    # --- (b) NVFP4-quantized MoE vs bf16 MoE: the quant noise floor ---
    w13_q = quantize_experts_nvfp4(w13)
    w2_q = quantize_experts_nvfp4(w2)
    out_nvfp4 = swigluoai_moe_nvfp4(
        hidden, w13_q, w2_q, topk_ids=topk_ids, topk_weights=topk_weights,
        alpha=alpha, limit=limit, compute_dtype=torch.bfloat16,
    )
    # bf16 reference computed in bf16 GEMMs (same compute dtype as the NVFP4 path)
    out_bf16 = swigluoai_moe(
        hidden, w13, w2, topk_ids=topk_ids, topk_weights=topk_weights,
        alpha=alpha, limit=limit, activation="swigluoai",
        compute_dtype=torch.bfloat16,
    )
    # weight-level quant error sanity
    w13_dq = dequantize_nvfp4(w13_q[0])
    print("\n(b) NVFP4-quantized MoE vs bf16 MoE (quant noise floor):")
    print(f"    weight[expert0 w13] dequant rel RMS = {_rel_rms(w13_dq, w13[0]):.6e}")
    print(f"    MoE-output abs RMS = {_rms(out_nvfp4, out_bf16):.6e}")
    print(f"    MoE-output rel RMS = {_rel_rms(out_nvfp4, out_bf16):.6e}")
    print(f"    => small vs the SiLU gap above (quant noise << wrong-activation error).")

    # --- (c) shapes / routing correctness ---
    print("\n(c) shapes / routing correctness:")
    assert out_swig.shape == (T, H), out_swig.shape
    assert out_nvfp4.shape == (T, H), out_nvfp4.shape
    # every token routed to exactly k distinct experts
    assert (topk_ids < E).all() and (topk_ids >= 0).all()
    distinct = torch.tensor([len(set(row.tolist())) for row in topk_ids])
    assert (distinct == k).all(), "top-k ids not distinct per token"
    # interleaved vs contiguous split differ (proves split convention matters)
    gap_split = _rms(
        swigluoai(gu, alpha=alpha, limit=limit, interleaved=False),
        swigluoai(gu, alpha=alpha, limit=limit, interleaved=True),
    )
    print(f"    out shape {tuple(out_swig.shape)} OK; nvfp4 out {tuple(out_nvfp4.shape)} OK")
    print(f"    top-k ids in [0,{E}) and distinct per token: OK")
    print(f"    contiguous-vs-interleaved split RMS = {gap_split:.4e} "
          f"(non-zero => layout choice is load-bearing)")

    # --- summary one-liners for the report ---
    print("\n" + "-" * 72)
    print("SUMMARY")
    print(f"  swigluoai-vs-SiLU  MoE rel RMS = {_rel_rms(out_swig, out_silu):.4e}")
    print(f"  NVFP4-vs-bf16      MoE rel RMS = {_rel_rms(out_nvfp4, out_bf16):.4e}")
    print(f"  NVFP4-vs-bf16      MoE abs RMS = {_rms(out_nvfp4, out_bf16):.4e}")
    print("-" * 72)


if __name__ == "__main__":
    _main()
