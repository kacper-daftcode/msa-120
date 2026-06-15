# SPDX-License-Identifier: MIT
"""SM120 block-sparse GQA attend impl for MiniMax-M3 (DRAFT).

Mirrors ``MiniMaxM3SparseTritonImpl.forward``
(vllm/models/minimax_m3/common/sparse_attention.py:316-373) but routes the
attend to our SM120 kernels:

  * prefill -> ``sm120_fmha_paged.forward_sparse_paged`` (per request)
  * decode  -> ``sm120_fmha_paged.forward_sparse_paged`` with short ``seq_q``

The Triton impl calls ``minimax_m3_sparse_attn`` (sparse_attn.py:441) /
``minimax_m3_sparse_attn_decode`` (sparse_attn.py:500). We replicate the same
metadata wiring and ``topk_idx`` consumption, but the underlying op contract
differs (see the big block of caveats below and vllm_integration/README.md).

!!! This is a code DRAFT. It will not run as-is. The paged kernel's KV-cache
    layout and page size DO NOT match vLLM's M3 cache yet; the mismatches are
    enumerated inline so the porting work is explicit. !!!
"""

from __future__ import annotations

import torch

# In the real overlay these import from vllm.models.minimax_m3.common.* — kept
# as a string-y reference here so this file is readable standalone in the repo.
from vllm.forward_context import get_forward_context  # type: ignore
from vllm.models.minimax_m3.common.sparse_attention import (  # type: ignore
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
)
from vllm.v1.attention.backend import AttentionLayer  # type: ignore

from ._loader import paged_ext


class MiniMaxM3SparseSm120Impl(MiniMaxM3SparseImpl):
    """Block-sparse attend over the indexer-selected blocks, on SM120.

    Subclass of the abstract base (sparse_attention.py:268). The ``__init__``
    signature is inherited unchanged (num_heads, head_size, scale, num_kv_heads,
    kv_cache_dtype, *, topk_blocks, sparse_block_size).
    """

    # ---- our paged kernel's K/V tile sizes ----
    # forward_sparse_paged hard-requires PAGE_SIZE == 64 in the cache layout
    # [num_pages, 64, Hkv, 128] (sm120_fmha_paged.cu:586-593). M3 pages are 128
    # (sparse_attention.py:80, SPARSE_BLOCK_SIZE=128). See PAGE_SIZE caveat below.
    KERNEL_PAGE_SIZE = 64

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_idx: "tuple[torch.Tensor | None, torch.Tensor | None]",
        output: torch.Tensor,
    ) -> torch.Tensor:
        # ---- mirror of MiniMaxM3SparseTritonImpl.forward (sparse_attention.py:324) ----
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound (line 326)
        main_md: MiniMaxM3SparseMetadata = attn_metadata[layer.layer_name]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)
        decode_topk, prefill_topk = topk_idx  # (line 329)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        hd = self.head_size
        # [total_q, num_heads, head_dim]  (our q layout: [Sq, Hq, 128]) — matches.
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)

        # NOTE(fp8): the Triton impl reinterprets the fp8 cache (line 336). Our
        # bf16 paged kernel is the serving path (M3 main attn is bf16); fp8 KV is
        # NOT supported by forward_sparse_paged yet, so select_main_impl_cls must
        # gate on `not is_quantized_kv_cache(kv_cache_dtype)` (see selector_patch).

        # ---- Split vLLM's fused KV cache into separate k_cache / v_cache ----
        # vLLM M3 cache: [num_blocks, 2, 128, num_kv_heads, head_dim] (K=[:,0]
        #   V=[:,1])  (sparse_attn.py:443).
        # Our kernel wants two tensors [num_pages, 64, Hkv, 128] (PAGE_SIZE=64).
        # MISMATCH: page axis 128 vs 64, and fused-2 vs split. This `_view_kv`
        # is a placeholder; a real port either (a) raises PAGE_SIZE to 128 in
        # the kernel, or (b) re-pages 128->2x64 and rebuilds block_table. We do
        # the cheap reshape for k/v split here and ASSUME a 128-page kernel
        # variant exists (KERNEL_PAGE_SIZE==128); flagged loudly.
        k_cache, v_cache = self._split_kv_cache(kv_cache)

        # Decode [:nd] — mirror of sparse_attention.py:341-354.
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None and decode_topk is not None
            self._run_paged(
                q[:nd],
                k_cache,
                v_cache,
                block_table=d.block_table,        # [num_decodes, max_blocks]
                topk_idx=decode_topk,             # [num_kv_heads, total_q, topk]
                seq_lens=d.seq_lens,              # [num_decodes] int32
                cu_seqlens_q=None,                # decode: uniform decode_query_len
                decode_query_len=d.decode_query_len,
                out=out[:nd],
            )

        # Prefill [nd:] — mirror of sparse_attention.py:357-372.
        if main_md.num_prefills > 0:
            p = main_md.prefill
            assert p is not None and prefill_topk is not None
            self._run_paged(
                q[nd:],
                k_cache,
                v_cache,
                block_table=p.block_table,        # [num_prefills, max_blocks]
                topk_idx=prefill_topk,            # [num_kv_heads, total_q, topk]
                seq_lens=p.seq_lens,              # [num_prefills] int32 (KV lens)
                cu_seqlens_q=p.cu_seqlens_q,      # [num_prefills+1] int32, rebased
                decode_query_len=None,
                out=out[nd:],
            )
        return output

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """vLLM fused cache [num_blocks, 2, 128, Hkv, D] -> (k, v) views.

        DRAFT: returns [num_blocks, 128, Hkv, D] views. Our kernel's TORCH_CHECK
        requires dim(1)==64; this WILL fail its shape check until either the
        kernel takes PAGE_SIZE=128 or the caller re-pages. Returned as-is so the
        layout mapping is visible; do not expect a passing TORCH_CHECK.
        """
        # kv_cache[:, 0] -> K, kv_cache[:, 1] -> V ; each [num_blocks,128,Hkv,D].
        k = kv_cache[:, 0]
        v = kv_cache[:, 1]
        return k.contiguous(), v.contiguous()

    def _run_paged(
        self,
        q: torch.Tensor,                 # [seq_q, Hq, 128]
        k_cache: torch.Tensor,           # [num_pages, PAGE, Hkv, 128]
        v_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,       # [num_reqs, max_blocks] int32
        topk_idx: torch.Tensor,          # [num_kv_heads, total_q, topk] int32
        seq_lens: torch.Tensor,          # [num_reqs] int32
        cu_seqlens_q: "torch.Tensor | None",
        decode_query_len: "int | None",
        out: torch.Tensor,
    ) -> None:
        """One forward_sparse_paged call per request.

        Our paged kernel (sm120_fmha_paged.cu:581) is per-(M-tile,head): its
        ``block_table`` is ``[num_m_blocks, max_logical_blocks]`` (per 64-query
        tile, sm120_fmha_paged.cu:597) and ``block_ids`` is 2D
        ``[num_m_blocks|seq_q, topk]`` HEAD-AGNOSTIC (line 599) — it has NO
        per-kv-head selection mode. vLLM gives a per-REQUEST block_table and a
        per-KV-HEAD topk. So we cannot pass the whole batch in one launch; we
        loop per request and, within a request, per kv-head (because the kernel
        applies one shared block_ids across all heads).

        This loop is the honest DRAFT shape of the call; a production port should
        add a kv-head-indexed block_ids mode + a per-request block_table arg to
        the kernel (one-launch), as the M8 doc recommends.
        """
        ext = paged_ext()
        scale = float(self.scale)
        num_kv_heads = self.num_kv_heads
        gqa = self.num_heads // num_kv_heads
        topk = topk_idx.shape[-1]

        num_reqs = seq_lens.shape[0]
        # Per-request query ranges.
        if cu_seqlens_q is not None:
            q_starts = cu_seqlens_q[:-1].tolist()
            q_ends = cu_seqlens_q[1:].tolist()
        else:
            assert decode_query_len is not None
            q_starts = [r * decode_query_len for r in range(num_reqs)]
            q_ends = [(r + 1) * decode_query_len for r in range(num_reqs)]

        for r in range(num_reqs):
            qs, qe = int(q_starts[r]), int(q_ends[r])
            if qe <= qs:
                continue
            q_r = q[qs:qe]                       # [sq_r, Hq, 128]
            seq_k = int(seq_lens[r].item())
            bt_r = block_table[r : r + 1]        # [1, max_blocks]
            # Our kernel wants block_table rows == num_m_blocks. With BLK_M=64
            # and decode/short prefill, num_m_blocks may be >1; replicate the
            # single request row to num_m_blocks (all M-tiles share the request's
            # page map). DRAFT shortcut.
            sq_r = qe - qs
            num_m_blocks = (sq_r + 64 - 1) // 64
            bt_tiles = bt_r.expand(num_m_blocks, bt_r.shape[1]).contiguous()

            for h in range(num_kv_heads):
                # topk_idx[h, qs:qe, :] -> [sq_r, topk] block ids (-1 pad).
                # This is the PER-KV-HEAD list; the kernel applies it to all gqa
                # q-heads of this kv-head, which matches GQA sharing.
                bids = topk_idx[h, qs:qe, :].contiguous()  # [sq_r, topk] int32
                # q for the gqa group of this kv-head.
                q_h = q_r[:, h * gqa : (h + 1) * gqa, :].contiguous()
                # NB: kernel's o is full [seq_q, Hq, 128]; here Hq==gqa for the
                # sliced group. seq_len_k passed for partial-last-block masking.
                o_h, _lse = ext.forward_sparse_paged(
                    q_h,
                    k_cache[:, :, h : h + 1, :].contiguous(),  # Hkv slice = 1
                    v_cache[:, :, h : h + 1, :].contiguous(),
                    bt_tiles,
                    bids,
                    scale,
                    True,            # causal
                    seq_k,
                )
                out[qs:qe, h * gqa : (h + 1) * gqa, :].copy_(o_h)
