# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Torch reference for the MiniMax-M3 *learned* lightning indexer block scorer.

This is the faithful reference for the small learned head M3 uses to pick the
top-k KV blocks for block-sparse attention. It is NOT the full-QK max-pool proxy
the dense FMHA "OnlyScore" path currently produces -- the real indexer has its
own projections, RMSNorm, and (partial) RoPE on a reduced index head dim.

Derivation / provenance (see docs/M3_INDEXER_SPEC.md for line-level citations):

* Checkpoint tensors (per sparse layer, BF16), from MiniMax-M3-NVFP4
  ``model.safetensors.index.json`` + safetensors headers:
    self_attn.indexer.q_proj.weight  [512, 6144]   (= index_n_heads*idx_dim, hidden)
    self_attn.indexer.k_proj.weight  [128, 6144]   (= 1*idx_dim, hidden)  -- single shared key head
    self_attn.indexer.q_norm.weight  [128]         (Gemma RMSNorm gain, per index head)
    self_attn.indexer.k_norm.weight  [128]         (Gemma RMSNorm gain)
* Config (text_config): index_n_heads=4, index_head_dim=128, index_block_size=128,
  index_topk_blocks=16, index_local_blocks=1; rope_theta=5e6,
  partial_rotary_factor=0.5 (rotary_dim=64), rms_norm_eps=1e-6.
* Scoring math (MSA paper arXiv:2606.13392 + vLLM reference
  models/minimax_m3/common/ops/index_topk.py):
    Q_idx = X W_q  -> [N, Hkv, d],   K_idx = X W_k -> [N, 1, d]   (single key head, broadcast)
    q_idx = GemmaRMSNorm(q_idx); k_idx = GemmaRMSNorm(k_idx); then partial-NeoX RoPE on both.
    S_{i,j}^(r) = (q_i^(r) . k_j) * scale,   scale = 1/sqrt(d) = head_dim**-0.5
    M_{i,b}^(r) = max_{j in block b, j<=i} S_{i,j}^(r)     (causal max-pool over 128-token block)
    TopK_b over M, k=16, per index head r (== per GQA group). Local block always kept.

Output layout matches what ``sparse_topk_select`` consumes:
    max_score[Hq_index, nblk, Q]   (here Hq_index == index_n_heads == 4)
which is exactly the cute kernel / golden ``max_score`` layout [heads, K_tiles, Q].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IndexerWeights:
    """Per-layer indexer parameters (load from the checkpoint, BF16/FP32)."""

    q_proj: torch.Tensor  # [n_heads * d, hidden]
    k_proj: torch.Tensor  # [d, hidden]
    q_norm: torch.Tensor  # [d]   Gemma RMSNorm gain
    k_norm: torch.Tensor  # [d]   Gemma RMSNorm gain


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gemma-style RMSNorm: normalize then scale by (1 + weight).

    Computed in fp32 (Gemma/FlashInfer convention) over the last dim.
    """
    dt = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    out = x * (1.0 + weight.float())
    return out.to(dt)


def _rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float, device, dtype):
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = positions.float()[:, None] * inv_freq[None, :]  # [N, half]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def apply_partial_neox_rope(
    x: torch.Tensor,  # [N, H, D]
    positions: torch.Tensor,  # [N]
    rotary_dim: int,
    theta: float,
) -> torch.Tensor:
    """Partial NeoX RoPE: rotate first ``rotary_dim`` channels (split-half), pass the rest.

    NeoX style pairs channel c with c+half within the rotary block.
    """
    N, H, D = x.shape
    half = rotary_dim // 2
    cos, sin = _rope_cos_sin(positions, rotary_dim, theta, x.device, x.dtype)  # [N, half]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    rot = x[..., :rotary_dim]
    x1 = rot[..., :half]
    x2 = rot[..., half:]
    rot_out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return torch.cat([rot_out, x[..., rotary_dim:]], dim=-1)


def m3_indexer_block_scores(
    q: torch.Tensor,            # [N, hidden]  hidden states feeding the indexer (or precomputed index_q if project=False)
    k: torch.Tensor,            # [N, hidden]  (same hidden states; single key head)
    indexer_weights: IndexerWeights,
    *,
    block_size: int = 128,
    n_heads: int = 4,
    head_dim: int = 128,
    scale: float | None = None,         # default 1/sqrt(head_dim)
    rotary_dim: int = 64,
    rope_theta: float = 5_000_000.0,
    eps: float = 1e-6,
    positions: torch.Tensor | None = None,   # [N] int; default arange(N)
    causal: bool = True,
    apply_rope: bool = True,
    project: bool = True,
) -> torch.Tensor:
    """Compute M3 indexer per-(index-head, kv-block, query) max scores.

    Returns ``max_score[n_heads, nblk, N]`` (fp32), matching the layout that
    ``sparse_topk_select`` consumes. ``nblk = ceil(N / block_size)``.

    With ``project=True`` (default) ``q``/``k`` are the layer-input hidden states
    ``[N, hidden]`` and the q_proj/k_proj GEMMs are applied here. With
    ``project=False`` they must already be ``[N, n_heads, head_dim]`` (q) and
    ``[N, head_dim]`` (k) and only norm/rope/score run.
    """
    device = q.device
    if scale is None:
        scale = head_dim ** -0.5
    N = q.shape[0]
    if positions is None:
        positions = torch.arange(N, device=device)

    if project:
        wq = indexer_weights.q_proj.to(q.dtype)
        wk = indexer_weights.k_proj.to(q.dtype)
        q_idx = (q @ wq.t()).view(N, n_heads, head_dim)
        k_idx = (k @ wk.t()).view(N, 1, head_dim)
    else:
        q_idx = q.view(N, n_heads, head_dim)
        k_idx = k.view(N, 1, head_dim)

    # Gemma RMSNorm per head.
    q_idx = gemma_rmsnorm(q_idx, indexer_weights.q_norm, eps)
    k_idx = gemma_rmsnorm(k_idx, indexer_weights.k_norm, eps)

    # Partial NeoX RoPE (same rope as the main attention branch).
    if apply_rope:
        q_idx = apply_partial_neox_rope(q_idx, positions, rotary_dim, rope_theta)
        k_idx = apply_partial_neox_rope(k_idx, positions, rotary_dim, rope_theta)

    # Token-level scaled dot products. Single key head broadcast across the
    # n_heads index-query heads. S[r, i, j] = scale * (q_idx[i, r] . k_idx[j, 0]).
    q_f = q_idx.float()                  # [N, H, D]
    k_f = k_idx.float()[:, 0, :]         # [N, D]
    # [H, N_q, N_k]
    s = torch.einsum("ihd,jd->hij", q_f, k_f) * scale

    if causal:
        i = positions[:, None]
        j = positions[None, :]
        causal_mask = (j <= i)  # [N_q, N_k]
        s = s.masked_fill(~causal_mask[None], float("-inf"))

    # Block max-pool over key blocks of `block_size`.
    nblk = (N + block_size - 1) // block_size
    pad = nblk * block_size - N
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    s = s.view(n_heads, N, nblk, block_size)
    max_score = s.max(dim=-1).values  # [H, N_q, nblk]
    return max_score.permute(0, 2, 1).contiguous()  # [H, nblk, N]


def full_qk_maxpool_reference(
    q_attn: torch.Tensor,  # [N, Hq, head_dim]  -- the REAL attention q (proxy path)
    k_attn: torch.Tensor,  # [N, Hk, head_dim]
    *,
    block_size: int = 128,
    scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """The current proxy: max-pool of the *full* attention QK (no learned indexer).

    Provided only so __main__ can show the learned indexer differs from it.
    Uses head 0 of each for a quick scalar comparison.
    """
    device = q_attn.device
    head_dim = q_attn.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5
    N = q_attn.shape[0]
    Hq = q_attn.shape[1]
    qf = q_attn.float()
    kf = k_attn.float()
    # broadcast kv heads to q heads (GQA) by simple repeat for the demo
    Hk = k_attn.shape[1]
    rep = Hq // Hk
    kf = kf.repeat_interleave(rep, dim=1)
    s = torch.einsum("ihd,jhd->hij", qf, kf) * scale
    if causal:
        idx = torch.arange(N, device=device)
        s = s.masked_fill(~(idx[None, :] <= idx[:, None])[None], float("-inf"))
    nblk = (N + block_size - 1) // block_size
    pad = nblk * block_size - N
    if pad:
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
    s = s.view(Hq, N, nblk, block_size)
    return s.max(dim=-1).values.permute(0, 2, 1).contiguous()


if __name__ == "__main__":
    torch.manual_seed(0)
    N, hidden = 384, 6144
    n_heads, head_dim = 4, 128
    block_size = 128
    dtype = torch.bfloat16

    x = torch.randn(N, hidden, dtype=dtype)
    w = IndexerWeights(
        q_proj=torch.randn(n_heads * head_dim, hidden, dtype=dtype) * 0.02,
        k_proj=torch.randn(head_dim, hidden, dtype=dtype) * 0.02,
        q_norm=torch.randn(head_dim, dtype=dtype) * 0.1,
        k_norm=torch.randn(head_dim, dtype=dtype) * 0.1,
    )

    ms = m3_indexer_block_scores(
        x, x, w, block_size=block_size, n_heads=n_heads, head_dim=head_dim
    )
    nblk = (N + block_size - 1) // block_size
    assert ms.shape == (n_heads, nblk, N), ms.shape
    finite = ms[torch.isfinite(ms)]
    print(f"learned indexer max_score shape = {tuple(ms.shape)}  (expected [{n_heads},{nblk},{N}])")
    print(f"  finite entries: {finite.numel()}/{ms.numel()}")
    print(f"  range: [{finite.min().item():.4f}, {finite.max().item():.4f}], "
          f"mean={finite.mean().item():.4f}")

    # Sanity: must differ from a full-QK max-pool on independent attention q/k.
    q_attn = torch.randn(N, n_heads, head_dim, dtype=dtype)
    k_attn = torch.randn(N, n_heads, head_dim, dtype=dtype)
    proxy = full_qk_maxpool_reference(q_attn, k_attn, block_size=block_size)
    print(f"proxy (full-QK max-pool) shape = {tuple(proxy.shape)}")
    # Same shape, but the learned-indexer scores are a different function entirely.
    both_finite = torch.isfinite(ms) & torch.isfinite(proxy)
    diff = (ms[both_finite] - proxy[both_finite]).abs().mean().item()
    print(f"  mean|indexer - proxy| over finite cells = {diff:.4f}  (expected clearly > 0)")
    assert diff > 1e-3, "learned indexer unexpectedly equals the full-QK proxy"

    # No-RoPE / no-project variants run too.
    q_pre = torch.randn(N, n_heads, head_dim, dtype=dtype)
    k_pre = torch.randn(N, head_dim, dtype=dtype)
    ms2 = m3_indexer_block_scores(
        q_pre, k_pre, w, block_size=block_size, n_heads=n_heads,
        head_dim=head_dim, project=False, apply_rope=False,
    )
    assert ms2.shape == (n_heads, nblk, N)
    print("self-check OK")
