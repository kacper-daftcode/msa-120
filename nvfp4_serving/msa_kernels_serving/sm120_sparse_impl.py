# SPDX-License-Identifier: MIT
"""SM120 block-sparse GQA attend impl for MiniMax-M3 (LIVE serving).

Routes the DECODE branch to our SM120 page-128 flash-decoding kernel
(``forward_sparse_decode_serving``); the PREFILL branch stays on vLLM's Triton
``minimax_m3_sparse_attn`` (Phase 1 -- prefill attend swap is a separate step).

Why decode-first: bs1 decode is the interactive hot path, and our decode kernel
(W4=3 ldmatrix, page-128, fused-cache, device seq_lens, per-kv-head block_ids)
is GRAPH-CAPTURE SAFE -- no host ``.item()``, no per-head Python loop, static
shapes -- so it captures into vLLM's FULL decode cudagraph (the thing the 90
tok/s baseline depends on). See SERVING_INTEGRATION.md.

Correctness vs the Triton reference + dense fp32 is proven in
``verify_decode_serving.py`` (rms < 3e-3 vs golden, < 3e-4 vs dense, incl.
per-kv-head-distinct selections, batched R, NHD+HND cache layouts, and 500
poison-stressed partial-selection cases NaN-free).
"""

from __future__ import annotations

import torch

from vllm.forward_context import get_forward_context  # type: ignore
from vllm.models.minimax_m3.common.ops.sparse_attn import (  # type: ignore
    minimax_m3_sparse_attn,
)
from vllm.models.minimax_m3.common.sparse_attention import (  # type: ignore
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseTritonImpl,
)

from ._loader import decode_serving_ext


class MiniMaxM3SparseSm120Impl(MiniMaxM3SparseImpl):
    """Decode -> SM120 kernel; prefill -> Triton. GQA group must be 16 (M3)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lazily JIT-built on first forward (kept off the import path so a failed
        # build degrades to an obvious runtime error, not a silent import skip).
        self._ext = None
        # Triton prefill fallback shares this impl's config; we call the op
        # directly rather than instantiating the Triton class.

    def _kernel(self):
        if self._ext is None:
            self._ext = decode_serving_ext()
        return self._ext

    def forward(
        self,
        layer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: "tuple[torch.Tensor | None, torch.Tensor | None]",
        output: torch.Tensor,
    ) -> torch.Tensor:
        # ---- mirror MiniMaxM3SparseTritonImpl.forward (sparse_attention.py:316) ----
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md: MiniMaxM3SparseMetadata = attn_metadata[layer.layer_name]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)
        decode_topk, prefill_topk = topk_idx

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        # M3 main attention is bf16; fp8 KV is gated to Triton by the selector,
        # so kv_cache is the bf16 fused cache [num_blocks, 2, 128, Hkv, hd] here.

        # ---- DECODE [:nd] -> our SM120 kernel ----
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            self._decode(q[:nd], kv_cache, decode_topk, d, out[:nd])

        # ---- PREFILL [nd:] -> Triton (Phase 1) ----
        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            minimax_m3_sparse_attn(
                q[nd:],
                kv_cache,
                prefill_topk,
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[nd:],
            )
        return output

    # ------------------------------------------------------------------
    def _decode(self, q, kv_cache, decode_topk, d, out) -> None:
        """Our page-128 flash-decoding kernel over the indexer-selected blocks.

        Graph-capture safe: only static-shape tensor ops + one allocation-free
        custom op. ``decode_topk`` is ``[Hkv, total_q, topk]`` (per-kv-head);
        for pure decode (decode_query_len == 1) ``total_q == num_decodes == R``,
        so we permute to ``[R, Hkv, topk]`` (a contiguous copy -- a capturable
        kernel, NOT a host sync).
        """
        qlen = d.decode_query_len
        num_heads = self.num_heads
        hkv = self.num_kv_heads
        gqa = num_heads // hkv

        if qlen != 1 or gqa != 16:
            # Spec-decode verify batches (qlen>1) or non-16 GQA: defer to Triton.
            from vllm.models.minimax_m3.common.ops.sparse_attn import (  # type: ignore
                minimax_m3_sparse_attn_decode,
            )
            minimax_m3_sparse_attn_decode(
                q, kv_cache, decode_topk, d.block_table, d.seq_lens,
                hkv, self.scale, out, qlen,
            )
            return

        R = q.shape[0]  # == num_decodes (qlen == 1)
        # decode_topk: [Hkv, R, topk] -> [R, Hkv, topk] int32 contiguous.
        block_ids = decode_topk.permute(1, 0, 2).contiguous().to(torch.int32)
        block_table = d.block_table.to(torch.int32)
        seq_lens = d.seq_lens.to(torch.int32)

        o, _lse = self._kernel().forward_sparse_decode_serving(
            q,                 # [R, Hq, 128] bf16
            kv_cache,          # [num_blocks, 2, 128, Hkv, 128] bf16 fused
            block_table,       # [R, max_logical_blocks] int32
            block_ids,         # [R, Hkv, topk] int32
            seq_lens,          # [R] int32 device
            float(self.scale),
            int(hkv),
            0,                 # split_chunks=0 -> 1 page/chunk (max split-K)
        )
        out.copy_(o)
