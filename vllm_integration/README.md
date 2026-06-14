# SM120 MSA adapters for vLLM MiniMax-M3 (M8 — DRAFT)

Adapter classes that route MiniMax-M3's pluggable sparse-attention impls to our
SM120 kernels on RTX PRO 6000 / 5090 (cap 12.x), instead of the portable Triton
fallback. **This is a code draft.** It will not run without (a) the kernels
built into the image and (b) the layout/entrypoint gaps below closed. Every gap
is flagged inline in the source with a file:line cite to what was mirrored.

## Files

| File | What |
| --- | --- |
| `sm120_sparse_impl.py` | `MiniMaxM3SparseSm120Impl(MiniMaxM3SparseImpl)` — block-sparse attend, mirrors `MiniMaxM3SparseTritonImpl` (sparse_attention.py:316). |
| `sm120_indexer_impl.py` | `MiniMaxM3IndexerSm120Impl(MiniMaxM3IndexerImpl)` — score + top-k, mirrors `MiniMaxM3IndexerTritonImpl` (indexer.py:375). |
| `patches.py` | Monkeypatch helpers for the two selectors. |
| `selector_patch.md` | Exact selector patch + image-overlay instructions. |
| `_loader.py` | Lazy JIT/AOT loader for the four `.cu` extension modules. |

## What maps cleanly (high confidence)

- **forward signatures.** Both abstract bases have a custom `forward`
  (`MiniMaxM3SparseImpl.forward` sparse_attention.py:301;
  `MiniMaxM3IndexerImpl.forward` indexer.py:364). The two impls reproduce the
  exact control flow of the Triton subclasses: profiling-run early-out, metadata
  lookup by `layer.layer_name` / `index_cache.prefix`, decode-`[:nd]` /
  prefill-`[nd:]` split, `(decode_topk, prefill_topk)` return.
- **q / output layout.** `query[:num_tokens].view(-1, num_heads, 128)`
  (sparse_attention.py:334) == our kernels' `q [Sq, Hq, 128]`
  (sm120_fmha_paged.cu:585). Direct.
- **indexer score -> topk composition.** Our `block_score_hmma` emits
  `max_score [H, nblk, total_q]` (sm120_indexer.cu:341), and
  `sm120_sparse_topk.topk_select` wants `max_score [H, K_tiles, Q]`
  (sm120_sparse_topk.cu:13) — **identical layout, no transpose.** This is the
  cleanest part of the whole integration.
- **force_begin / force_end.** topk_select's `force_begin`/`force_end`
  (sm120_sparse_topk.cu:14) map to M3's `init_blocks`/`local_blocks` (M8 doc
  line 52; index_topk.py:702 `init_blocks`,`local_blocks`).
- **GQA / kv-head topk.** vLLM's `topk_idx [num_kv_heads, total_q, topk]` is
  per-KV-head, GQA-shared (sparse_attn.py:444). Slicing `topk_idx[h, qs:qe, :]`
  and applying it to that kv-head's gqa q-heads is the correct GQA semantics.

## Open questions / mismatches (need resolution + live testing)

### A. Page size 128 vs 64 (BLOCKER for attend)
- M3 cache page == sparse block == **128** (sparse_attention.py:80;
  SPARSE_BLOCK_SIZE=128). Our `forward_sparse_paged` hard-requires page **64**:
  `k_cache.size(1) == 64` TORCH_CHECK (sm120_fmha_paged.cu:587). The draft
  `_split_kv_cache` produces 128-page views, which will FAIL that check.
- Fix (M8 doc lines 39, 76): raise the kernel to PAGE_SIZE=128 (two 64-subtiles,
  like the forward's blk_kv=128) or a native-128 page. Until then the attend
  cannot consume the M3 cache.

### B. Fused vs split KV cache (BLOCKER for attend)
- M3 cache is **one fused tensor** `[num_blocks, 2, 128, num_kv_heads, D]`,
  K=[:,0]/V=[:,1] (sparse_attn.py:443). Our kernel takes **two separate**
  `k_cache`/`v_cache` `[num_pages, 64, Hkv, 128]` (sm120_fmha_paged.cu:586). The
  draft slices `kv_cache[:,0]` / `[:,1]`; whether those are contiguous-enough for
  the kernel's pointer math (page/pos/head strides) needs verification — likely
  needs a kernel that takes the fused 5-D cache + the `2` stride directly.

### C. Per-request block_table + head-agnostic block_ids (perf + correctness)
- Our paged kernel's `block_table` is `[num_m_blocks, max_logical_blocks]`
  (per 64-query M-tile, sm120_fmha_paged.cu:597) and `block_ids` is **2-D,
  head-agnostic** `[num_m_blocks|seq_q, topk]` (sm120_fmha_paged.cu:599). vLLM
  gives a **per-request** block_table `[num_reqs, max_blocks]` and a
  **per-kv-head** topk. The draft works around this with a Python loop
  per-request × per-kv-head (one launch each) + replicating the request's
  block_table row to all M-tiles. Correct-ish but slow and clearly a draft.
- Fix (M8 doc lines 35, 79): add a kv-head-indexed `block_ids` mode (a
  `bids_head_stride` indexed by hkv, the perhead kernel already has this:
  sm120_fmha_perhead.cu:710-744) AND a per-request block_table path, so the
  whole batch runs in one launch. Recommended before serving.

### D. Indexer score-only entrypoint MISSING (BLOCKER for indexer)
- `idx_q` arrives **pre-projected/normed/roped** (index_topk.py:642-657; M8 doc
  lines 44-49). We need ONLY the score+maxpool half (`block_score_hmma`,
  sm120_indexer.cu:337). But the only pybind today is `block_scores`
  (sm120_indexer.cu:736) = the **full** project+norm+rope+score pipeline from
  hidden states + projection weights. The indexer impl raises
  `NotImplementedError` and documents the pseudocall to an intended
  `index_block_scores(idx_q, index_kv_cache, block_table, ...)` entrypoint that
  must be added.

### E. Indexer K is paged in vLLM, dense in our kernel (BLOCKER for indexer)
- vLLM index-K cache is **paged** `[num_blocks, 128, head_dim]` addressed via
  `block_table` (index_topk.py:643, 749). Our `block_score_hmma` reads a
  **dense** `k_idx [N, 128]` (sm120_indexer.cu:339) with no block_table. The new
  score-only entrypoint (item D) must take the paged index-K + block_table +
  cu_seqlens/prefix_lens, not the dense buffer.

### F. score layout transpose vs vLLM's own score op
- vLLM's `minimax_m3_index_score` returns `[num_idx_heads, total_q, max_block]`
  (index_topk.py:656) — axes (total_q, block) swapped vs our `[H, nblk, total_q]`.
  We deliberately keep OUR layout end-to-end (it feeds topk_select with no
  transpose, item under "maps cleanly"). This is fine BECAUSE we also replace
  the topk op; just don't try to feed our score into vLLM's `minimax_m3_index_topk`.

### G. topk fixed at 16
- `topk_select` hard-codes topk=16 (sm120_sparse_topk.cu:24). The selector
  requires `topk_blocks == 16`; configs with a different topk stay on Triton.
  If M3's deployed config differs, the kernel needs a templated/runtime topk.

### H. topk_idx output layout
- topk_select returns `[total_q, H, 16]` (sm120_sparse_topk.cu:25). vLLM
  consumers index `t_ptr [num_kv_heads, total_q, topk]` (sparse_attn.py:444).
  The indexer impl does `permute(1,0,2).contiguous()` to match. Cheap; verify
  the attend reads the contiguous result with the strides it expects.

### I. num_valid / per-query causal masking in topk
- `topk_select` takes a single scalar `num_valid` (sm120_sparse_topk.cu:14), but
  in prefill each query token has a different visible-block count (causal). The
  Triton path bakes per-query causality into the **score** (`-inf` future
  blocks, index_topk.py:154) so topk naturally excludes them. Our draft relies
  on the same: the score kernel must `-inf` future blocks per query so a global
  `num_valid` is safe. VERIFY the SM120 score kernel's causal fill matches
  per-query (sm120_indexer.cu causal path) or topk will over-select.

### J. decode path uses the prefill/paged forward
- Triton has a dedicated split-K decode (`minimax_m3_sparse_attn_decode`,
  sparse_attn.py:500; `minimax_m3_index_decode`, index_topk.py:746). We reuse the
  same paged forward with short `seq_q` (M8 doc line 31) and the indexer score
  path with `decode_query_len`. Functionally OK; perf-wise our v1 kernels have no
  flash-decoding split-K, so decode throughput is the main perf risk (M8 doc
  lines 71-75).

### K. fp8 KV not supported
- Triton dequants fp8 KV (sparse_attn.py:336, USE_FP8). Our bf16 paged kernel
  doesn't. Selector gates on `not is_quantized_kv_cache` so fp8 stays on Triton.
  M3 main attention is bf16, so the bf16 path is the serving path (M8 doc line 86).

## Test strategy (per M8 doc §63)
1. **Op-equivalence inside the image:** SM120 impl vs `MiniMaxM3*TritonImpl` on
   identical (q, paged KV, topk_idx, metadata). Attend: match to FP8-PV floor
   (~0.035 rms). topk: exact (our topk_select already matches golden bit-for-bit,
   tests/test_sm120_sparse_topk.py). Blocked until A,B,D,E land.
2. **Serve + quality/throughput** vs the MXFP4 Triton baseline (102 tok/s).
   Expect correctness; perf may regress until M10 (kernels are v1).
