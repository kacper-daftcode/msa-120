# MSA golden-oracle capture format & validation results

Capture file (NVFP4 checkpoint, git-lfs):

    /home/kacper/models/MiniMax-M3-NVFP4/msa_golden/msa_sm100_golden_v1.safetensors  (9.7 MB)

The checkpoint README describes `msa_golden` as *"SM100 (B200) golden-oracle
captures of the MSA attention kernels (dense+maxscore, top-k select,
block-sparse attention, NVFP4-KV, fp8 decode, LSE conventions)"*. This **v1**
capture is a single safetensors with **8 tensors** and **empty `__metadata__`**
(no documented LSE base / scale / convention — everything below is reverse
engineered and verified numerically). There are **no separate decode / fp8 /
NVFP4 / LSE tensors** in this v1 file.

## Tensors

| name               | dtype | shape              | meaning |
|--------------------|-------|--------------------|---------|
| `q`                | bf16  | [512, 8, 128]      | queries: seq_q=512, Hq=8, head_dim=128 |
| `k_pages`          | bf16  | [64, 1, 128, 128]  | paged K: 64 pages × Hkv=1 × page=128 × d=128 |
| `v_pages`          | bf16  | [64, 1, 128, 128]  | paged V (same layout) |
| `kv_indices`       | i32   | [64]               | page order — identity 0..63 |
| `max_score`        | f32   | [8, 128, 128→512]  | **[Hq=8, K=128 blocks, Q=512]** indexer scores; block_size=64; contains `-inf` |
| `kv_block_indexes` | i32   | [512, 8, 16]       | **[Q=512, Hq=8, topk=16]** selected bs64 block ids, ascending, always 16 valid, max idx 63 |
| `dense_out`        | bf16  | [512, 8, 128]      | full **causal** attention output |
| `sparse_out`       | bf16  | [512, 8, 128]      | block-sparse attention output |

Derived geometry: `seq_k = 64 pages × 128 = 8192`, GQA ratio **8:1** (Hq=8,
Hkv=1), `scale = 1/sqrt(128) = 0.088388`. The `max_score` grid has 128 blocks
over 8192 keys → **block_size 64**. Selected block ids never exceed 63 because
the upper 64 blocks (keys 4096..8192) are `-inf`-masked for every row.

## Stage map (golden tensor → our SM120 kernel)

| MSA stage          | golden tensor(s)                  | our kernel |
|--------------------|-----------------------------------|------------|
| block score        | `max_score`                       | `block_max_score(q,k,scale,block_size)` |
| top-k select       | `max_score → kv_block_indexes`    | `topk_select(max_score,num_valid,fb,fe)` |
| block-sparse fwd   | `q,k,v,kv_block_indexes → sparse_out` | `forward_sparse(q,k,v,block_ids,scale,causal,blk_kv)` |
| dense (anchor)     | `q,k,v → dense_out`               | (no named MSA dense kernel; torch reference) |
| decode / fp8 / nvfp4 / LSE | **absent in v1**          | n/a |

## Validation results (RTX PRO 6000 Blackwell, sm_120; torch 2.11+cu130)

Run via `tests/test_golden_msa.py`.

### Stage A — top-k select — **MATCH (exact)**
Feed the **golden `max_score`** into our `topk_select(max_score,
num_valid_pages=128, force_begin=0, force_end=0)` and compare to **golden
`kv_block_indexes`**:

    exact rows (as sets): 4096/4096
    mean overlap        : 16.000 / 16

Our top-k selector reproduces the golden selection **bit-for-bit** on all
4096 (q,h) rows. This is the one stage where convention lines up perfectly
(both are a plain top-16 over the 128-block fp32 score grid; every row has
≥61 finite blocks so no clamping/forcing is exercised).

### Stage B — block score — **DOCUMENTED MISMATCH (conceptual)**
Our `block_max_score` is a **full-QK max-pool proxy**. Golden `max_score` is
the **learned NVFP4 indexer** score (sparse/masked: only ~61–64 of 128 blocks
are finite per row, and the finite set grows with q in a way a dense max-pool
cannot produce). Comparing on golden-finite entries only:

    rms = 2.7321e+01   maxabs = 6.2236e+01   rel-rms = 0.9223
    proxy-top16 == golden-top16 rows: 0/4096

The proxy and the learned indexer disagree completely — **as the task
warned**. This is reported, not forced to pass. (Consistency check: Stage A
proves the golden chain `golden max_score → golden indices` is exact, while
Stage B proves our proxy cannot *reconstruct* the golden max_score — both true.)

### Stage C — block-sparse forward — **DOCUMENTED MISMATCH (capture convention, NOT our kernel)**
Feed golden `q/k/v` + golden `kv_block_indexes` (per-query, bs64) into our
`forward_sparse` and compare to golden `sparse_out`:

    our forward_sparse (non-causal) vs golden sparse_out:  rel-rms = 1.5476
    our forward_sparse (causal=True) vs golden sparse_out: rel-rms = 4.9624

Apples-to-apples diagnostic — a pure-torch **fp32** bs64 block-sparse reference
using the **same** golden block_ids:

    torch fp32 bs64 sparse-ref vs golden sparse_out:        rel-rms = 1.5486
    our kernel       vs torch fp32 sparse-ref (same ids):   rel-rms = 0.0358

**Interpretation:** our kernel agrees with an independent fp32 reference to
**0.0358 rel-rms** (exactly the kernel's documented FP8-E4M3 PV noise floor of
~0.035). The fp32 reference *itself* is 1.55 off golden `sparse_out`. Therefore
the 1.55 gap is **entirely capture-side convention, not a bug in our kernel**.
Most likely causes (cannot be disambiguated from this v1 capture, which has no
metadata):

- **NVFP4-KV**: README lists "NVFP4-KV"; golden `sparse_out` is plausibly
  computed with NVFP4-quantized K/V, whereas our path uses bf16 Q/K and FP8
  PV with a neutral UE8M0 scale. A bf16 dense-equivalent reference cannot
  reproduce an NVFP4-KV result.
- **block_size / masking region**: golden indices are bs64; the sparse forward
  may apply a different intra-block causal/window mask than our non-causal v0.
- **LSE / normalization**: no LSE tensor is provided to cross-check softmax
  normalization base or sparse renormalization.

### Stage C addendum — convention diagnostic resolves it: NOT NVFP4-KV
`tests/diag_nvfp4_convention.py` (pure torch, no kernel) sweeps every plausible
convention against golden `sparse_out`. Results:

| KV-prec | block_size | causal | rel-rms vs golden sparse_out |
|---------|-----------:|--------|------------------------------:|
| bf16    | 64         | no/yes | **1.5486** |
| bf16    | 128        | no     | 1.1464 (best of sweep) |
| bf16    | 128        | yes    | 1.1472 |
| NVFP4   | 64         | no/yes | 1.5501 |
| NVFP4   | 128        | no/yes | 1.1499 / 1.1507 |

Key facts from the same script:

- **NVFP4-KV is a red herring for v1.** Quantizing K/V to the checkpoint's NVFP4
  scheme (block-16 E2M1 + E4M3 per-block scale + fp32 global; K/V quant error
  ≈0.095 rel-rms each) does **not** reduce the gap — it slightly *increases* it.
  The dense anchor is conclusive: full-causal with **bf16** KV is 0.0023 vs
  golden `dense_out`, but with **NVFP4** KV it jumps to **0.1335**. So the
  capture used plain bf16 K/V, not NVFP4-KV. Implementing an NVFP4-KV kernel
  would therefore *not* close the sparse gap. No kernel was shipped.
- **block_size 128 helps but does not close** (1.55 → 1.15); **causal vs
  non-causal is a no-op** here (every golden block id ≤63 sits far below each
  query's right-aligned causal horizon, so nothing is masked).
- **Structural smoking gun.** golden `sparse_out` rows have norm ≈0.42 while our
  block-sparse softmax over the golden indices has norm ≈0.58 but is nearly
  orthogonal to golden (mean row-cosine **0.18**). golden `dense_out` is *more*
  aligned to golden `sparse_out` (cosine **0.53**) than our reconstruction is.
- **Assumption-free oracle test.** A greedy search that may pick *any* 16 of the
  128 blocks (unconstrained by the golden indices) still cannot reach golden
  `sparse_out`: best rel-rms **0.70–0.92**, and the recovered blocks overlap the
  golden `kv_block_indexes` by only **1–3 / 16** (and include blocks >63 that the
  indexer never selects).

**Resolved interpretation:** golden `sparse_out` is **not** a block-sparse
softmax over the captured `q / k_pages / v_pages` using the captured
`kv_block_indexes` — by *any* KV precision, block size, mask, or even oracle
selection. The 1.55 gap is a **capture-side decoupling** (the `sparse_out`
tensor was produced from a different RoPE phase / head permutation / KV
projection than the captured q/K/V and indices), not NVFP4-KV, not block
geometry, not normalization, and not our kernel. Because the dense anchor proves
bf16 KV, there is **nothing for an NVFP4-KV kernel to fix** in this v1 capture.

### Stage D — dense convention anchor — **CONFIRMED**
Torch full **causal** attention (right-aligned: query *i* sees keys
`[0 .. seq_k-seq_q+i]`) vs golden `dense_out`:

    rms = 4.2775e-05   maxabs = 2.8963e-04   rel-rms = 0.0023

i.e. bf16-roundoff exact. This nails down the capture's dense convention
(full **causal**, right-aligned, scale 1/sqrt(128), GQA 8:1).

## Honest caveats

- Only **Stage A (top-k select)** is claimed as a match against golden. Do not
  read Stage B/C as kernel correctness — Stage B is a deliberate proxy, Stage C
  is blocked by a capture-side convention (NVFP4-KV / masking), proven by the
  fp32-reference diagnostic.
- This v1 capture has **no fp8-decode, no NVFP4 raw tensors, no LSE tensor, and
  empty metadata**, so the README's "fp8 decode / NVFP4-KV / LSE conventions"
  cannot be directly validated here — only the dense / maxscore-indexer /
  top-k-select / block-sparse stages are present, and of those only top-k
  select and dense convention line up cleanly with our bf16 SM120 kernels.
- The golden weights live outside the repo (`/home/kacper/models/...`) and are
  **not** committed.
