# SPDX-License-Identifier: MIT
"""SM120 lightning-indexer impl for MiniMax-M3 (LIVE serving, Phase 2).

Routes the indexer TOP-K selection onto our SM120 ``topk_select_varlen`` kernel
(per-query num_valid -> SET-EXACT vs the Triton per-query topk, 2-3x faster),
while keeping the Triton SCORE kernels byte-identical. The score is the
projection-free, paged, causal block-max-pool that the fused
``minimax_m3_index_decode`` / ``minimax_m3_index_score`` already compute; we
reuse those EXACT Triton score launches and only swap the top-k op.

Why only topk (not the score kernel): vLLM's ``idx_q`` is pre-projected / normed
/ roped and index-K is paged ``[num_blocks,128,head_dim]`` via block_table. Our
standalone indexer score kernel does the full projection from hidden states and
reads a dense (non-paged) index-K, so it is NOT a drop-in for this contract
(blocker D/E). The top-k, by contrast, consumes the score tensor directly and is
where our kernel WINS, so Phase 2 swaps exactly that.

GRAPH-CAPTURE SAFE: the decode branch reuses the same split-K decode-score
launch (already captured into the FULL decode cudagraph by the Triton impl) and
adds only static-shape device ops (arange, sub, masked_fill) + one allocation-
free custom op. No host ``.item()`` / no host sync in the captured region.

CORRECTNESS: ``topk_select_varlen`` is proven SET-EXACT vs the fused Triton
decode topk across a mixed-seq stress grid (see test_topk_mixed_decode.py
``stress`` -> ALL SET-EXACT, 24 cases). The score reuse is verified identical
(test_topk_mixed_decode.py / test_score_decode_via_prefill.py).
"""

from __future__ import annotations

import os

import torch

from vllm.forward_context import get_forward_context  # type: ignore
from vllm.platforms import current_platform  # type: ignore
from vllm.models.minimax_m3.common.indexer import (  # type: ignore
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
)
from vllm.models.minimax_m3.common.ops.index_topk import (  # type: ignore
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_score,
    minimax_m3_index_decode,
    _decode_index_score_kernel,
)
from vllm.utils.math_utils import round_up  # type: ignore

from ._loader import topk_ext

# Our topk_select is fixed at topk == 16 (sm120_sparse_topk.cu).
SM120_TOPK = 16

# Which phases run our varlen topk. Default: prefill only (decode bs1 is launch-
# bound -- the fused Triton decode is already cheap there and our extra per-layer
# host ops cost more than the op saves; prefill is many query rows where our
# 2-3x faster topk dominates). Override via SM120_INDEXER_DECODE=ours to also
# swap decode (measured slower at bs1 -- see SERVING_INTEGRATION.md Phase 2).
_DECODE_OURS = os.environ.get("SM120_INDEXER_DECODE", "triton").lower() == "ours"
_PREFILL_OURS = os.environ.get("SM120_INDEXER_PREFILL", "ours").lower() == "ours"


class MiniMaxM3IndexerSm120Impl(MiniMaxM3IndexerImpl):
    """Indexer top-k on SM120 (our varlen kernel); score stays Triton.

    Mirrors ``MiniMaxM3IndexerTritonImpl.forward`` but replaces the top-k op:
      * decode  : reuse the split-K decode SCORE launch, then our varlen topk.
      * prefill : reuse ``minimax_m3_index_score``, then our varlen topk.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tk = None

    def _topk(self):
        if self._tk is None:
            self._tk = topk_ext()
        return self._tk

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> "tuple[torch.Tensor | None, torch.Tensor | None]":
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        index_md: MiniMaxM3IndexerMetadata = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)

        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache  # [num_blocks, 128, head_dim]

        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None

        assert self.topk_blocks == SM120_TOPK, (
            f"sm120 topk_select is fixed at {SM120_TOPK}; "
            f"model topk_blocks={self.topk_blocks}"
        )

        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            if _DECODE_OURS:
                decode_topk = self._decode_topk(
                    iq[:nd], kv, d.block_table, d.seq_lens,
                    d.max_seq_len, d.decode_query_len,
                )
            else:
                decode_topk = minimax_m3_index_decode(
                    iq[:nd], kv, d.block_table, d.seq_lens, d.max_seq_len,
                    self.topk_blocks, self.init_blocks, self.local_blocks,
                    self.num_kv_heads, self.scale, d.decode_query_len,
                )

        if index_md.num_prefills > 0:
            p = index_md.prefill
            assert p is not None
            if _PREFILL_OURS:
                prefill_topk = self._prefill_topk(
                    iq[nd:], kv, p.block_table, p.cu_seqlens_q,
                    p.seq_lens, p.context_lens, p.max_query_len, p.max_seq_len,
                )
            else:
                from vllm.models.minimax_m3.common.ops.index_topk import (  # type: ignore
                    minimax_m3_index_topk,
                )
                score = minimax_m3_index_score(
                    iq[nd:], kv, p.block_table, p.cu_seqlens_q, p.seq_lens,
                    p.context_lens, p.max_query_len, p.max_seq_len,
                    self.num_kv_heads, self.scale,
                )
                prefill_topk = minimax_m3_index_topk(
                    score, p.cu_seqlens_q, p.context_lens, p.max_query_len,
                    self.topk_blocks, self.init_blocks, self.local_blocks,
                )

        return decode_topk, prefill_topk

    # ------------------------------------------------------------------
    def _run_varlen_topk(self, score_hqk, num_valid_q):
        """score_hqk [H, total_q, Kstride] fp32 (out-of-range == -inf),
        num_valid_q [total_q] int32 -> topk [H, total_q, 16] (attend t_ptr)."""
        # our topk wants [H, K, Q]; transpose. (contiguous: the transpose kernel
        # reads a [H,K,Q] row-contiguous buffer.)
        max_score = score_hqk.permute(0, 2, 1).contiguous()  # [H, K, total_q]
        out = self._topk().topk_select_varlen(
            max_score,
            num_valid_q.to(torch.int32).contiguous(),
            int(self.init_blocks),
            int(self.local_blocks),
        )  # [total_q, H, 16] int32, asc block ids, -1 pad
        # -> [num_kv_heads(H), total_q, topk] (sparse_attn.py t_ptr layout).
        return out.permute(1, 0, 2).contiguous()

    def _decode_topk(self, idx_q, kv, block_table, seq_lens,
                     max_seq_len, decode_query_len):
        """Reuse the fused-decode SPLIT-K score launch, then our varlen topk.

        idx_q: [total_q, H, 128]; total_q == num_decodes * decode_query_len.
        """
        total_q, H, head_dim = idx_q.shape
        max_block = (max_seq_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
        score_block_stride = round_up(max_block, 16)
        # -inf init so blocks the score kernel does NOT write (beyond each
        # query's causal range) are never selected over real blocks.
        score = torch.full(
            (H, total_q, score_block_stride), float("-inf"),
            dtype=torch.float32, device=idx_q.device,
        )
        use_pdl = current_platform.is_arch_support_pdl()
        pdl = {"launch_pdl": True} if use_pdl else {}
        TARGET_GRID, MAX_NUM_KV_CHUNKS = 4096, 256
        target = max(1, min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, total_q * H)))
        num_kv_chunks = 1 << (target.bit_length() - 1)
        _decode_index_score_kernel[(total_q, num_kv_chunks)](
            idx_q, kv, score, block_table, seq_lens,
            H, head_dim, int(self.init_blocks), int(self.local_blocks),
            self.scale, decode_query_len,
            idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
            kv.stride(0), kv.stride(1), kv.stride(2),
            score.stride(0), score.stride(1), score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE, num_kv_chunks=num_kv_chunks,
            USE_PDL=use_pdl, **pdl,
        )
        # per-query num_valid = ceil(kv_len/128). For decode_query_len q-slots per
        # request, query slot j sees kv_len = seq_len - decode_query_len + j + 1.
        nv = self._decode_num_valid(seq_lens, decode_query_len, total_q)
        return self._run_varlen_topk(score, nv)

    @staticmethod
    def _decode_num_valid(seq_lens, decode_query_len, total_q):
        """num_valid[total_q] = ceil(kv_len/128), request-major flatten.

        kv_len for request r, query slot j (0..qlen-1) =
            seq_lens[r] - decode_query_len + j + 1.
        For qlen==1 this is simply ceil(seq_lens/128). Static-shape, device-only.
        """
        BS = SPARSE_BLOCK_SIZE
        if decode_query_len == 1:
            kv_len = seq_lens
        else:
            r = seq_lens.view(-1, 1)  # [R,1]
            j = torch.arange(decode_query_len, device=seq_lens.device).view(1, -1)
            kv_len = (r - decode_query_len + j + 1).reshape(-1)  # [R*qlen]
        kv_len = kv_len.clamp_min(0)
        return (kv_len + BS - 1) // BS  # [total_q]

    def _prefill_topk(self, idx_q, kv, block_table, cu_seqlens_q,
                      seq_lens, context_lens, max_query_len, max_seq_len):
        """Reuse minimax_m3_index_score, then our varlen topk (per-query causal).

        idx_q: [total_q, H, 128]. Score op leaves out-of-range slots unwritten;
        we -inf them per query before topk.
        """
        H = idx_q.shape[1]
        score = minimax_m3_index_score(
            idx_q, kv, block_table, cu_seqlens_q, seq_lens, context_lens,
            max_query_len, max_seq_len, H, self.scale,
        )  # [H, total_q, score_block_stride] (out-of-range slots = garbage)

        nv = self._prefill_num_valid(cu_seqlens_q, context_lens, idx_q.shape[0])
        # -inf out-of-range slots: block k is valid for query q iff k < nv[q].
        Kdim = score.shape[2]
        kidx = torch.arange(Kdim, device=score.device).view(1, 1, -1)
        mask = kidx >= nv.view(1, -1, 1)  # [1, total_q, Kdim]
        score = score.masked_fill(mask, float("-inf"))
        return self._run_varlen_topk(score, nv)

    @staticmethod
    def _prefill_num_valid(cu_seqlens_q, context_lens, total_q):
        """num_valid[total_q] = ceil(kv_len/128) per prefill query token.

        For request r with context_lens[r] cached tokens, the j-th query token
        (j in [0, query_len_r)) has kv_len = context_lens[r] + j + 1. We build
        this from cu_seqlens_q (per-request query spans), all on device.
        Static-shape within a captured graph (total_q fixed).
        """
        BS = SPARSE_BLOCK_SIZE
        dev = cu_seqlens_q.device
        # position of each flattened query token within its request:
        #   local_j[t] = t - cu_seqlens_q[req(t)]
        # req(t) = searchsorted(cu_seqlens_q, t, right=True) - 1
        t = torch.arange(total_q, device=dev, dtype=torch.int32)
        req = torch.searchsorted(cu_seqlens_q.to(torch.int32), t, right=True) - 1
        req = req.clamp_min(0)
        local_j = t - cu_seqlens_q.to(torch.int32)[req]
        kv_len = context_lens.to(torch.int32)[req] + local_j + 1
        kv_len = kv_len.clamp_min(0)
        return (kv_len + BS - 1) // BS  # [total_q] int32
