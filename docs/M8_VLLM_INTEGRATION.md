# M8 — Integrating the SM120 MSA kernels into vLLM (MiniMax-M3)

Goal: serve MiniMax-M3 on SM120 (RTX PRO 6000) using our SM120 MSA kernels
instead of the portable Triton fallback the model currently uses on SM120.

## Why this is clean: vLLM already has pluggable MSA impls

The M3 model (`vllm/models/minimax_m3/common/`) selects indexer and attend
implementations at build time:

- `sparse_attention.py:376 select_main_impl_cls(...)`:
  ```py
  if current_platform.is_cuda() and current_platform.is_device_capability_family(100) and ...:
      return MiniMaxM3SparseMSAImpl     # SM100 (B200) CuTe MSA
  return MiniMaxM3SparseTritonImpl      # everything else -> Triton  (SM120 lands HERE)
  ```
- `indexer.py:436 select_indexer_impl_cls(...)`: same shape; SM100 MSA indexer
  score path is currently disabled, so even SM100 uses `MiniMaxM3IndexerTritonImpl`.

On SM120, `is_device_capability_family(100)` is False (cap 12.0 → family 120),
so both fall back to Triton. **Integration = add SM120 impl subclasses and make
the selectors return them on family 120 + bf16.** No model-graph surgery.

## The two adapter classes to write

### 1. `MiniMaxM3SparseSm120Impl(MiniMaxM3SparseImpl)`
`forward(self, q, kv_cache, topk_idx, metadata, ...)` (see `MiniMaxM3SparseImpl.forward`,
sparse_attention.py:301). Mirror `MiniMaxM3SparseTritonImpl` (sparse_attention.py:313)
but call our kernels:
- prefill → `forward_sparse_perhead`/paged (our `sm120_fmha_perhead`/`sm120_fmha_paged`).
- decode → our paged forward with short seq_q.
- `topk_idx` layout is `[num_kv_heads, total_q, topk]` (PER KV-HEAD, GQA-shared).
  Our per-head kernel takes `[seq_q, Hq, topk]` or `[Hq, seq_q, topk]` — expand
  the kv-head topk to q-heads (repeat-interleave by Hq/Hkv) OR add a kv-head
  indexing mode (cheap: change `bids_head_stride` to index by hkv).
- KV is paged: `kv_cache` page layout + `block_table` from metadata → our
  `forward_sparse_paged` (PAGE_SIZE must match vLLM's M3 page size; the M3
  Triton path forces a specific page size — confirm and match; our v1 paged
  kernel is PAGE_SIZE=64, M3 blocks are 128 → either set PAGE_SIZE=128 or split).

### 2. `MiniMaxM3IndexerSm120Impl(MiniMaxM3IndexerImpl)`
`forward(...) -> (decode_topk, prefill_topk)` (indexer.py:364). Mirror
`MiniMaxM3IndexerTritonImpl` (indexer.py:372) calling our kernels for:
- score: maps to `minimax_m3_index_score(idx_q[total_q,num_idx_heads,128],
  index_kv_cache[num_blocks,128,128], block_table, cu_seqlens_q, seq_lens,
  prefix_lens) -> score[num_idx_heads,total_q,max_block]`. NOTE: idx_q is
  ALREADY projected/normed/roped upstream — so we only need the **scoring +
  causal block-max-pool** half of our `sm120_indexer` (the `block_score_kernel`),
  not the full projection path. Our full kernel's projection half is unused here.
- topk: maps to `minimax_m3_index_topk(score, cu_seqlens_q, prefix_lens,
  max_query_len, topk=16, init_blocks, ...)`. Our `sparse_topk_select` already
  matches golden bit-for-bit; wire force_begin/force_end = init/local blocks.

## Build / packaging
- Base: the official `vllm/vllm-openai:minimax-m3` image (vllm 0.1.dev17492,
  torch 2.11+cu130, has CUDA 13 + nvcc? verify; if no nvcc, our kernels must be
  AOT-built into a wheel, since they JIT via cpp_extension at first import).
- Ship our `python/fmha_sm100/csrc/*.cu` + a small loader; build the extensions
  (JIT on first import, or AOT during image build). Add the two impl classes +
  a patch to `select_*_impl_cls` (overlay onto the image like 0xSero does, or a
  vLLM plugin).

## Test strategy (before serving)
1. **Op-equivalence**: inside the image, run our SM120 impl vs `MiniMaxM3*TritonImpl`
   on identical (q, paged KV, topk_idx, metadata). Expect match to the FP8-PV
   floor (~0.035 rms) for attend; exact for topk. This proves drop-in.
2. **Serve**: launch with the SM120 impls selected; smoke-test + the existing
   chat/throughput checks; compare quality (golden / eval) and tok/s vs the
   MXFP4 Triton baseline (102 tok/s, Phase 1).

## Risks / open items
- **Perf**: our kernels are v1 (correctness-first). Integrated correctness is
  expected, but they may be SLOWER than the tuned Triton fallback until the perf
  pass (M10) lands. Validate this is a step toward NVFP4-native speed, not a
  regression — benchmark honestly.
- **Page size 128**: M3 uses 128-token blocks; our paged kernel is PAGE_SIZE=64.
  Set PAGE_SIZE=128 (two 64-subtiles, like the forward's blk_kv=128) or a native
  128 page. Required before real serving.
- **kv-head vs q-head topk**: `[num_kv_heads, total_q, topk]` — add a kv-head
  indexing mode to the per-head kernel (one-line stride change) rather than
  materializing an expanded q-head tensor.
- **Indexer projection**: lives upstream of the score op; our score+maxpool half
  is what plugs in. The projection-faithful kernel (sm120_indexer full path) is
  useful for a standalone/fused variant but not required for the op contract.
- **NVFP4-KV**: only if matching the SM100 golden decode path; M3 main attention
  is bf16, so the bf16 attend is the serving path.
