# MiniMax-M3 Learned Block-Score Indexer — Implementable Spec

The lightweight "lightning indexer" head M3 uses to pick the top-k KV blocks for
block-sparse attention. This is a **separate small learned head**, not the
full-QK max-pool our dense-FMHA OnlyScore path currently produces.

Reference impl: [`python/fmha_sm100/indexer_ref.py`](../python/fmha_sm100/indexer_ref.py).

## 1. Sources and confidence

| Fact | Source | Confidence |
|---|---|---|
| Projection tensor names + shapes + dtype | NVFP4 checkpoint `model.safetensors.index.json` + safetensors headers | **Verified** (read from checkpoint) |
| index_n_heads/head_dim/block/topk/local, rope_theta, partial_rotary | NVFP4 `config.json` `text_config` | **Verified** (read from checkpoint) |
| Score = scale·(q·k), block max-pool, top-16, local-block | MSA paper arXiv:2606.13392; vLLM `models/minimax_m3/common/ops/index_topk.py` | **Verified** (matches paper eqs + reference kernel) |
| Gemma RMSNorm + partial NeoX RoPE on the index path; `scale = head_dim**-0.5` | vLLM `models/minimax_m3/nvidia/model.py` (`MiniMaxM3SparseAttention.__init__` / forward) | **High** (reference vLLM impl; not re-derived from weights) |
| Exact NeoX (split-half) vs interleaved RoPE variant; that index RoPE positions == token positions | inferred from `index_rotary_emb = self.rotary_emb` + "partial NeoX RoPE" comment | **Medium** (inferred, the rotary object is shared with main attn but I did not execute it) |
| `init_blocks = 0` for M3 | `index_local_blocks=1` present; no `index_init_block` key in this config | **Medium** (absence ⇒ 0; native schema uses `sparse_init_block`) |

## 2. Checkpoint tensors (per sparse layer)

From `model.safetensors.index.json` (rename note: the base checkpoint's
`index_*` keys are renamed to `indexer.*` in transformers conversion —
`brandonmmusic-max/minimax-m3-nvfp4-recipe` `conversion_mapping.py`:
`index_*->indexer.*`). Shapes/dtypes read from the safetensors header of
`model-00007-of-00048.safetensors` (layer 3, first sparse layer):

```
model.language_model.layers.{L}.self_attn.indexer.q_proj.weight  BF16 [512, 6144]
model.language_model.layers.{L}.self_attn.indexer.k_proj.weight  BF16 [128, 6144]
model.language_model.layers.{L}.self_attn.indexer.q_norm.weight  BF16 [128]
model.language_model.layers.{L}.self_attn.indexer.k_norm.weight  BF16 [128]
```

There are **only these four** tensors per indexer (228 total / 4 = 57 sparse
layers; layers 0-2 are `full_attention`, 3-59 are `minimax_m3_sparse`). Notably
absent: any per-head weight/`weights_proj`, softmax-scale, value, or bias tensor.
The index path is **BF16** (attention is kept BF16 in this NVFP4 export).

Shapes decode as:
* `q_proj [512, 6144]` = `(index_n_heads · index_head_dim, hidden)` = `(4·128, 6144)` → **4 index-query heads**, one per GQA group.
* `k_proj [128, 6144]` = `(1 · index_head_dim, hidden)` = `(128, 6144)` → **single shared index-key head**.
* `q_norm/k_norm [128]` = Gemma RMSNorm gain over `index_head_dim`.

## 3. Config (`text_config`)

```
index_n_heads        = 4     (== num_key_value_heads; one index-q head per GQA group)
index_head_dim       = 128   (= d_idx)
index_block_size     = 128   (one sparse block = one 128-token KV page)
index_topk_blocks    = 16    (k = 16)
index_local_blocks   = 1     (the local block at the query position is force-kept)
# init blocks: not present in this config -> 0
rms_norm_eps         = 1e-6
rope_theta           = 5_000_000
partial_rotary_factor= 0.5  ->  rotary_dim = 64  (of the 128 head dim)
```

(`sparse_attention_config` is `null` in this NVFP4 config; the values live in the
flattened `index_*` keys. The native-transformers/vLLM schema instead nests them
as `sparse_num_index_heads`, `sparse_index_dim`, `sparse_block_size`,
`sparse_topk_blocks`, `sparse_local_block`, `sparse_init_block`.)

## 4. The exact per-(query, kv-block) scoring math

Let `X ∈ R^{N×hidden}` be the layer input (same hidden states that feed q/k/v),
`d = index_head_dim = 128`, `Hkv = index_n_heads = 4`, `B = block_size = 128`.

1. **Project** (no bias):
   `Q_idx = X · W_q^T  →  [N, Hkv, d]`  (4 index-query heads)
   `K_idx = X · W_k^T  →  [N, 1, d]`    (single shared key head)
   (MSA paper: `Q^idx = X W_q^idx ∈ R^{N×Hkv×d}`, `K^idx = X W_k^idx ∈ R^{N×1×d}`.)

2. **Gemma RMSNorm** per head over `d` (gain `(1 + w)`, fp32):
   `Q_idx = GemmaRMSNorm(Q_idx, q_norm)`,  `K_idx = GemmaRMSNorm(K_idx, k_norm)`.
   (vLLM `model.py`: `self.index_q_norm = MiniMAXGemmaRMSNorm(idx_head_dim, eps)`,
   same `index_k_norm`; `MiniMAXGemmaRMSNorm.forward` → `gemma_rmsnorm`.)

3. **Partial NeoX RoPE** on the first `rotary_dim = 64` channels of both
   `Q_idx` and `K_idx`, using the token positions — the **same** rotary object as
   the main attention branch (vLLM `model.py`: `self.index_rotary_emb =
   self.rotary_emb`; forward comment "per-head Gemma QK-norm + partial NeoX RoPE on
   the main (q/k) and index (index_q/index_k) branches").

4. **Token-level scaled dot product** (no nonlinearity), single key head broadcast
   over the 4 query heads:
   `S^{(r)}_{i,j} = (Q_idx[i, r] · K_idx[j, 0]) · scale`,  `scale = d^{-1/2} = head_dim**-0.5`.
   (MSA paper: `S^{idx,(r)}_{i,j} = (Q^idx)_i^{(r)} (K^idx)_j^T / sqrt(d_idx)`.
   vLLM `model.py`: `self.scaling = self.head_dim**-0.5`, passed as the indexer
   `scale`. In `index_topk.py` the kernel multiplies by `sm_scale * 1.4426950409`
   (`= 1/ln2`); that log2e factor is a base-2 convenience that is **monotonic**, so
   it does not change top-k ordering and is omitted from the reference.)

5. **Causal block max-pool** to one score per 128-token KV block:
   `M^{(r)}_{i,b} = max_{ j ∈ block b, j ≤ i } S^{(r)}_{i,j}`.
   (MSA paper `M^{idx,(r)}_{i,b} = max_{j∈B_b, j≤i} S^{idx,(r)}_{i,j}`; vLLM
   `_index_block_score_kernel`: causal mask then `score = tl.max(qk, axis=1)`, one
   sparse block per 128-token K-tile.)

6. **Top-k + forced local** (downstream of scoring, done by `sparse_topk_select`):
   `I^{(r)}_i = TopK_b(M^{(r)}_{i,·}, k=16)`, computed **independently per index
   head r** (no cross-head reduce; `index_topk.py` asserts
   `num_idx_heads == num_kv_heads`), the result shared by all G query heads in the
   group. The **local block** containing position `i` is always included
   (`local_blocks=1`); `init_blocks=0` here. In the vLLM kernel forced blocks are
   encoded by adding large biases (`+1e29` local, `+1e30` init) before the top-k.

## 5. Output layout (maps to `sparse_topk_select`)

The reference returns `max_score[Hq_index, nblk, Q]` with `Hq_index =
index_n_heads = 4`, `nblk = ceil(N / 128)`. This is **exactly** the layout the
existing pipeline already consumes:

* cute kernel `fp4_indexer.py` writes `mScores` as
  `make_layout((heads_q, max_k_tiles, total_q), …)` → `[Hq, K_tiles, Q]`.
* `fp4_indexer_block_scores(...)` docstring: returns
  `[Hq, ceil(max_seqlen_k/128), total_qo_len]`, fp32, `-inf` outside range.
* golden `msa_golden/msa_sm100_golden_v1.safetensors`: `max_score F32 [8, 128, 512]`
  = `[heads, K_tiles, Q]`.

The one semantic difference vs the proxy: here the `Hq` axis is the **4 index
heads** (one per GQA group), not the 64 main-attention heads. `sparse_topk_select`
runs top-16 + forced-local per index head; the selected block ids are then shared
by all 16 main heads in that GQA group.

## 6. Pipeline shape (decode vs prefill)

Same scoring kernel, two schedules (vLLM `index_topk.py`):
* **prefill**: `_index_block_score_kernel` over query tiles, causal max-pool →
  `score[H, total_q, max_block]`, then `_topk_index_kernel`.
* **decode**: `_decode_index_score_kernel` (split-K over KV blocks) →
  `score[H, total_q, max_block]`, then split-K partial top-k + merge. Forced
  init/local blocks are written directly into the score (`1e30`/`1e29`).

## 7. Tensors the real indexer needs + delta vs our full-QK proxy

**Checkpoint tensors required** (per sparse layer, BF16):

| tensor | shape | role |
|---|---|---|
| `self_attn.indexer.q_proj.weight` | `[512, 6144]` | index-query proj, 4 heads × 128 |
| `self_attn.indexer.k_proj.weight` | `[128, 6144]` | index-key proj, single shared head |
| `self_attn.indexer.q_norm.weight` | `[128]` | Gemma RMSNorm gain on index-q |
| `self_attn.indexer.k_norm.weight` | `[128]` | Gemma RMSNorm gain on index-k |

(Plus the shared `rotary_emb` cos/sin cache from `rope_theta=5e6`, `rotary_dim=64`.)

**Current proxy** (dense FMHA `OnlyScore` mode, e.g. `test_onlyscore_pipeline.py`
and the `fp4_indexer_block_scores` path) computes, per main attention head:
`max_score[h, blk, q] = max_{j∈blk, j≤q} (q_attn[q,h] · k_attn[j,h]) · head_dim^{-1/2}`
— i.e. a block max-pool of the **real attention** Q·K (the same 64-head / 128-dim
q/k that feed softmax attention, already RoPE'd and qk-normed).

**Delta to be faithful to M3** — a hypothetical `sm120_block_score.cu` would have to:

1. **Use the learned index projections, not attention q/k.** Add `indexer.q_proj`
   (`[512,6144]`) and `indexer.k_proj` (`[128,6144]`) GEMMs from hidden states to
   produce `index_q [N,4,128]` and `index_k [N,1,128]`. The proxy reuses the
   attention q/k and computes no separate projection.
2. **Head count = 4 index heads, not 64.** The score tensor's leading axis becomes
   the 4 index heads (one per GQA group), and top-k runs per index head; the proxy
   currently scores all 64 (or `Hq`) attention heads. K is a **single shared head**
   broadcast over the 4 query heads (vs per-head/GQA-grouped K in the proxy).
3. **Apply Gemma RMSNorm `(1+w)` with the index q_norm/k_norm gains** over the 128
   index dim (separate from the attention q_norm/k_norm).
4. **Apply the same partial NeoX RoPE** (`rotary_dim=64`, theta `5e6`) to index q/k.
   (Numerically distinct from the attention path only because the inputs differ.)
5. **Scale = `head_dim^{-1/2}` with `head_dim=128`** — identical value to the proxy,
   so no change there. The `*1/ln2` log2e factor in the vLLM kernel is optional
   (monotonic; irrelevant to top-k).
6. **Max-pool, causal mask, block_size=128, output `[4, nblk, Q]`** — identical to
   the proxy's reduction/layout, so the downstream `sparse_topk_select` (top-16 +
   forced local block, `init=0`) is unchanged once fed the 4-head index scores.

In short: the reduction, masking, layout, and top-k are already correct in the
proxy; what is missing is the **separate learned projection + index RMSNorm +
RoPE + the 4-index-head / single-shared-key-head structure**. Swapping the proxy's
attention-q/k input for the projected, normed, rope'd index vectors (with 4 query
heads and 1 key head) makes it faithful.

