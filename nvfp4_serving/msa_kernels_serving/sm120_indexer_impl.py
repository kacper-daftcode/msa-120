# SPDX-License-Identifier: MIT
"""SM120 lightning-indexer impl for MiniMax-M3 (DRAFT).

Mirrors ``MiniMaxM3IndexerTritonImpl.forward``
(vllm/models/minimax_m3/common/indexer.py:375-433) but routes scoring + top-k
through our SM120 kernels:

  * score (prefill & decode) -> ``sm120_indexer`` block-score path
  * top-k                    -> ``sm120_sparse_topk.topk_select``

The Triton impl calls, for decode, the fused ``minimax_m3_index_decode``
(score+topk, index_topk.py:746) and, for prefill, ``minimax_m3_index_score``
(index_topk.py:641) then ``minimax_m3_index_topk`` (index_topk.py:702). We split
both phases into a score op + our topk_select.

KEY CONTRACT NOTE (M8 doc, index_topk.py:642-657):
``idx_q`` is ALREADY projected / normed / roped upstream. So we need only the
SCORE+MAXPOOL half of our ``sm120_indexer`` kernel (``block_score_hmma``,
sm120_indexer.cu:337), NOT its full projection pipeline. BUT: our only pybind
entrypoint, ``block_scores`` (sm120_indexer.cu:736), is the FULL pipeline taking
hidden states + projection weights — there is no score-only pybind today. So
this impl is blocked on exposing ``block_score_hmma`` as its own entrypoint
(e.g. ``index_block_scores(q_idx[N,H,128], k_idx[N,128], positions, ...)``).
The call below targets that NOT-YET-EXISTING entrypoint and is marked clearly.

!!! Code DRAFT — will not run until the score-only entrypoint exists and the
    paged/dense index-K layout is reconciled (see README.md). !!!
"""

from __future__ import annotations

import torch

from vllm.forward_context import get_forward_context  # type: ignore
from vllm.models.minimax_m3.common.indexer import (  # type: ignore
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
)

from ._loader import indexer_ext, topk_ext

# Our topk_select hard-codes topk == 16 (sm120_sparse_topk.cu:24). M3's
# topk_blocks is configurable; assert it matches at build/select time.
SM120_TOPK = 16


class MiniMaxM3IndexerSm120Impl(MiniMaxM3IndexerImpl):
    """Indexer score + top-k on SM120.

    __init__ inherited from the base (indexer.py:328): owns the side cache via
    ``self.index_cache``; ``indexer_backend_cls`` left as the base
    ``MiniMaxM3IndexerBackend`` (its cache shape [num_blocks,128,head_dim],
    indexer.py:95, is what our score path needs to read).
    """

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        # ---- mirror of MiniMaxM3IndexerTritonImpl.forward (indexer.py:380) ----
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run (indexer.py:381)
        index_md: MiniMaxM3IndexerMetadata = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)

        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        # idx_q: [total_q, num_index_heads, index_head_dim==128] (indexer.py:386)
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache  # [num_blocks, 128, head_dim] (indexer.py:95)

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            decode_topk = self._score_and_topk(
                iq[:nd],
                kv,
                block_table=d.block_table,
                seq_lens=d.seq_lens,
                # decode: each query token's KV length = its own position+1;
                # the score kernel reconstructs it from seq_lens/decode_query_len.
                cu_seqlens_q=None,
                prefix_lens=None,
                max_query_len=d.decode_query_len,
                max_seq_len=d.max_seq_len,
                decode_query_len=d.decode_query_len,
            )

        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            prefill_topk = self._score_and_topk(
                iq[nd:],
                kv,
                block_table=p.block_table,
                seq_lens=p.seq_lens,
                cu_seqlens_q=p.cu_seqlens_q,     # [num_prefills+1] int32, rebased
                prefix_lens=p.context_lens,      # [num_prefills] int32
                max_query_len=p.max_query_len,
                max_seq_len=p.max_seq_len,
                decode_query_len=None,
            )

        return decode_topk, prefill_topk

    # ------------------------------------------------------------------
    def _score_and_topk(
        self,
        idx_q: torch.Tensor,             # [total_q, num_idx_heads, 128]
        index_kv_cache: torch.Tensor,    # [num_blocks, 128, head_dim]
        *,
        block_table: torch.Tensor,       # [num_reqs, max_blocks] int32
        seq_lens: torch.Tensor,          # [num_reqs] int32
        cu_seqlens_q: "torch.Tensor | None",
        prefix_lens: "torch.Tensor | None",
        max_query_len: int,
        max_seq_len: int,
        decode_query_len: "int | None",
    ) -> torch.Tensor:
        """Score every visible 128-block, max-pool, then top-k.

        vLLM op contract (index_topk.py):
          score = minimax_m3_index_score(...) -> [num_idx_heads, total_q, max_block]
          topk  = minimax_m3_index_topk(score, ...) -> [num_idx_heads, total_q, topk]

        Our ops:
          max_score = sm120_indexer.<index_block_scores>(...) -> [H, nblk, total_q]
              (block_score_hmma output, sm120_indexer.cu:341,690 -> [H,nblk,N])
          out = sm120_sparse_topk.topk_select(max_score[H,K,Q], num_valid,
                    force_begin, force_end) -> [total_q, H, 16]
              (sm120_sparse_topk.cu:14-40)

        IMPORTANT LAYOUT MATCH: our topk_select wants max_score as [H, K, Q]
        which is EXACTLY our indexer's [H, nblk, total_q] output — they compose
        with no transpose. But vLLM's score op produces [H, total_q, max_block]
        (axes 1<->2 swapped) — so we deliberately keep OUR [H, nblk, total_q]
        layout end-to-end and never materialize vLLM's score layout.

        topk_select returns [total_q, H, 16]; vLLM consumers
        (sparse_attn.py:444 t_ptr) want [num_kv_heads, total_q, topk]. So we
        permute(1, 0, 2) -> [H, total_q, 16] before returning. (H == num_idx_heads
        == num_kv_heads for M3, index_topk.py:659.)
        """
        idx = indexer_ext()
        tk = topk_ext()

        # ---- SCORE ----
        # !!! BLOCKED: needs a score-only entrypoint exposing block_score_hmma.
        #     The existing `block_scores` (sm120_indexer.cu:736) is the FULL
        #     project+norm+rope+score pipeline from hidden states + weights, which
        #     is NOT what we want (idx_q is pre-projected). Pseudocall to the
        #     intended new entrypoint:
        #
        #         max_score = idx.index_block_scores(
        #             idx_q,                 # [total_q, H, 128] bf16 (pre-roped)
        #             index_kv_cache,        # paged index-K [num_blocks,128,128]
        #             block_table, seq_lens,
        #             cu_seqlens_q, prefix_lens,
        #             self.block_size,       # sparse block size == 128
        #             self.scale, causal=True,
        #         )  # -> [H, nblk, total_q] fp32
        #
        # Until that exists, raise so nobody silently runs the wrong (projection)
        # kernel. The full `block_scores` is NOT a drop-in: it also needs the
        # paged index-K layout (our standalone kernel is dense [N,128], vLLM's is
        # paged [num_blocks,128,128] via block_table) — flagged in README.
        if not hasattr(idx, "index_block_scores"):
            raise NotImplementedError(
                "sm120_indexer needs a score-only paged entrypoint "
                "(index_block_scores). The full `block_scores` pipeline does "
                "projection from hidden states and reads a dense (non-paged) "
                "index-K, neither of which matches the M3 indexer op contract. "
                "See vllm_integration/README.md."
            )
        max_score = idx.index_block_scores(  # pragma: no cover (entrypoint TBD)
            idx_q,
            index_kv_cache,
            block_table,
            seq_lens,
            cu_seqlens_q,
            prefix_lens,
            int(self.block_size),
            float(self.scale),
            True,  # causal
        )  # [H, nblk, total_q] fp32, contiguous

        # ---- TOP-K ----
        # num_valid: number of real KV blocks for the longest sequence; the
        # kernel clamps per-row by num_valid (block ids >= num_valid -> -1).
        # For mixed prefill, num_valid is per-query, but our topk_select takes a
        # single scalar; pass the global max-block count and rely on -inf score
        # rows being masked by the score kernel's causal fill (DRAFT: verify the
        # per-query causal masking actually lands in max_score, else this
        # over-selects future blocks).
        num_valid = max_score.shape[1]  # nblk

        # force_begin/force_end == init_blocks/local_blocks (M8 doc line 52).
        force_begin = int(self.init_blocks)
        force_end = int(self.local_blocks)

        assert self.topk_blocks == SM120_TOPK, (
            f"sm120 topk_select is fixed at {SM120_TOPK}; "
            f"model topk_blocks={self.topk_blocks}"
        )
        topk_qh16 = tk.topk_select(
            max_score.contiguous(),
            num_valid,
            force_begin,
            force_end,
        )  # [total_q, H, 16] int32, ascending block ids, -1 pad

        # -> [num_kv_heads(H), total_q, topk] to match the attend's t_ptr layout
        # (sparse_attn.py:444). decode_query_len unused here (per-token rows
        # already flattened into total_q).
        return topk_qh16.permute(1, 0, 2).contiguous()
