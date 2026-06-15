# SM120 MSA kernels -> vLLM MiniMax-M3-NVFP4 serving: results & verdict

Box: 4x RTX PRO 6000 Blackwell Server (SM120). Image:
`vllm/vllm-openai:minimax-m3` (torch 2.11.0+cu130, nvcc 13.0). Date: 2026-06-14.
Container: `minimax-m3-nvfp4`. vLLM M3 module path (image overlay):
`/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/`.

---

## 0. TL;DR / verdict

**The bs1-decode "eager-break removal" thesis is DISPROVEN (structural, case b),
and is moreover already moot at bs1 decode in the marlin baseline.** Swapping in
our SM120 kernels cannot win on that axis. The kernels themselves are real and
correct on SM120 (all four build + run in the prod image; paged attend and topk
pass golden tests on this exact box; **topk is set-exact op-equivalent to vLLM's
Triton reference**). But the full attend integration is hard-blocked on cache
layout + a missing indexer entrypoint (blockers A,B,D,E below) that require
kernel surgery, not a bridge. **Per the GOAL's "blockers too deep -> deliver
partial + gap list + restore marlin" clause, marlin is left serving** (verified
coherent: "capital of Poland" -> Warsaw; 17x23 -> 391).

Value delivered: parity proof for the bs1-critical top-k op + a validated build
recipe + the substrate for NVFP4-KV later. Not a serving perf win.

---

## 1. EARLY FINDING (reported first, as required): eager-break is STRUCTURAL

**Question:** is the per-token eager break a property of (a) the Triton impl, or
(b) the sparse-attention LAYER structurally?

**Answer: (b) structural — and independent of the attend impl.**

Evidence (file:line in the image overlay):

- `nvidia/model.py:661` decorates **`MiniMaxM3SparseAttention._run_attention`**
  with `@eager_break_during_capture`. That method wraps BOTH the indexer
  (`self.indexer(index_query)`) AND the attend (`self.impl.forward(...)`) in a
  *single* break (model.py:661-671). The decorator is applied to the **method**,
  not to any impl, so it fires identically for `MiniMaxM3SparseTritonImpl`,
  `MiniMaxM3SparseMSAImpl` (the existing B200 path), or our SM120 impl.
- `compilation/breakable_cudagraph.py` `eager_break_during_capture`: the break is
  decided purely by capture context + cudagraph mode. It inspects nothing about
  the impl. Two consequences:
  - `if not is_breakable_cudagraph_enabled(): return fn` -> with breakable
    cudagraph OFF the decorator is a pure pass-through (no break at all).
  - `if mode == CUDAGraphMode.FULL: return fn(...)` -> **under FULL cudagraph the
    decorated op is captured into the graph as ordinary nodes; NO eager break.**

**So an impl-swap cannot remove the eager break.** The break is a layer-level
construct gated on cudagraph mode, not on which kernel backs the op.

**Stronger still — at bs1 decode the baseline already has zero per-token eager
breaks.** The marlin launch auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` with
`cudagraph_mode=FULL_AND_PIECEWISE`, and the startup log shows it capturing a
**`(decode, FULL)` graph for all 51 decode sizes incl. bs1** (plus a PIECEWISE
set for mixed prefill-decode). Under that FULL decode graph the eager-break
decorator is a pass-through, so the 57 sparse + 3 dense attention segments are
captured *inside* the full decode graph. The sawtooth the thesis targets lives in
the **PIECEWISE / mixed-batch** path (prefill-adjacent), not in steady-state bs1
decode. Measured baseline bs1 decode = **91.08 tok/s, TPOT 10.98 ms/tok**
(`results/marlin_baseline.json`) — consistent with a captured full graph, not a
60-break-per-token sawtooth.

**Implication for the project thesis:** the value of our kernels is NOT
eager-break removal / bs1-win. It is (1) per-op correctness parity on SM120 and
(2) the substrate for NVFP4-KV attention later. We did not chase a bs1 win we
could not structurally obtain.

Bonus finding: even the existing **B200 MSA impl** (`MiniMaxM3SparseMSAImpl`,
`nvidia/sparse_attention_msa.py`) does MSA only for **prefill** and falls back to
the **Triton split-K kernel for decode** ("no MSA decode yet", line 51). So at
bs1 decode, B200 and marlin both run the same Triton decode -- another reason an
attend swap can't move bs1 decode.

---

## 2. Kernels build AND run on SM120 in the prod image

JIT-compiled all four `.cu` inside the live container (torch.cpp_extension):

| kernel | builds | entrypoint(s) |
| --- | --- | --- |
| `sm120_sparse_topk.cu` | OK (~31s) | `topk_select` |
| `sm120_indexer.cu` | OK (~32s) | `block_scores` |
| `sm120_fmha_perhead.cu` | OK (~29s) | `forward_sparse`, `forward_sparse_perhead` |
| `sm120_fmha_paged.cu` | OK (~29s) | `forward_sparse_paged` |

Two image-specific build gotchas were found and resolved (now codified in
`_loader.py:prepare_build_env`):

1. **Missing CUDA math headers.** The base image has the nvcc compiler
   (`/usr/local/cuda-13.0`) but NOT `cusparse.h` / `cusolverDn.h` /
   `cusolver_common.h` (torch ATen pulls these in transitively). They live only
   in the pip wheel `nvidia/cu13/include`. Fix: **symlink just those three** into
   `/usr/local/cuda/targets/x86_64-linux/include`. (Putting the whole cu13
   include dir in front of `/usr/local/cuda` instead breaks `__cudaLaunch`
   host-stub codegen by shadowing nvcc's matching crt/runtime headers.)
2. **Arch family target.** Must use `-gencode=arch=compute_120f,code=sm_120f`
   (the SM120 *family* target), NOT plain `compute_120`. The attend kernels emit
   `mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X` (consumer-Blackwell
   block-scaled MMA, used for the FP8-PV GEMM, sm120_fmha_paged.cu:86). `ptxas`
   **rejects** that instruction on `sm_120` ("not supported on .target sm_120")
   but **accepts** it on `sm_120f`. The repo's own tests already use `_120f`;
   this is just a hard requirement to record.

**The M8 doc's AOT-wheel concern is moot: first-import JIT works in-image.**

### Run evidence on THIS box (CUDA_VISIBLE_DEVICES=0, alongside marlin)

- `test_sm120_paged.py`: **12/12 PASS, O rms = 0.000e+00** (mha/gqa, causal,
  per-tile/per-query block_ids, partial last block, multi-M-block).
- `test_sm120_sparse_topk.py`: **ALL PASSED** (bad=0 across all cases),
  on "NVIDIA RTX PRO 6000 Blackwell Server Edition".

So the kernels are not merely compiling -- they execute correctly on SM120.

---

## 3. OP-EQUIVALENCE: topk_select vs vLLM Triton minimax_m3_index_topk

`op_equivalence_topk.py` (run in-image) feeds one canonical random score tensor
to both ops in their respective layouts and compares the **selected KV-block
set** per (head, query). Triton returns score-sorted ids; our kernel returns
ascending ids; both pad with -1 -- so the criterion is set equality of the
de-padded selection.

Result (`results/op_equivalence_topk.txt`): **8/8 cases set-exact**, across block
counts {8,16,17,20,32,40,64}, with init/local forced blocks {0..4}, including
partial-fill (B<topk) and exactly-topk. **VERDICT: ALL MATCH (set-exact).**

This is the **bs1-critical** op: top-k block selection runs every decode token in
the indexer. It is a drop-in match for vLLM's reference selection.

(The attend op-equivalence -- our `forward_sparse_paged` vs `minimax_m3_sparse_attn`
-- is **blocked** by the cache-layout gaps in section 4; the kernel cannot
consume the M3 paged cache without surgery, so we cannot yet feed both ops the
identical paged KV. The attend MATH is independently proven correct by
test_sm120_paged.py rms=0 on the page-64 regime.)

---

## 4. Integration gap list (what blocks a live attend swap)

The two impl classes (`sm120_sparse_impl.py`, `sm120_indexer_impl.py`) and the
selector patch (`selector_patch.md`) are drafted; the loader + build recipe are
validated. The remaining blockers are KERNEL changes, not glue:

| # | blocker | where | effort |
| --- | --- | --- | --- |
| A | **Page size 64 vs 128.** `forward_sparse_paged` hard-checks `k_cache.size(1)==64` (sm120_fmha_paged.cu:595-600) and bakes `PAGE_SIZE==BLK_N==64` (line 26). M3 cache page == sparse block == **128**. | kernel | medium: split each 128-page into two 64-subtiles (block_table/block_ids in 128-units), like the forward's blk_kv=128. |
| B | **Fused vs split KV cache.** M3 cache is one fused 5-D tensor `[num_blocks,2,128,Hkv,D]` (K=[:,0]/V=[:,1]); kernel takes two separate `[num_pages,64,Hkv,128]`. Slicing `[:,0]/[:,1]` is non-contiguous for the kernel's page/pos/head pointer math. | kernel | medium: take the fused 5-D cache + the `2` stride directly. |
| C | **Per-launch GQA + per-request block_table.** Kernel grid is `(num_m_blocks, num_heads_q)`, one launch, head-agnostic `block_ids`, block_table rows == num_m_blocks. vLLM gives per-request block_table + per-kv-head topk. The draft works around this with a **Python loop per (request x kv-head)** + `.contiguous()` copies + one launch each -- a perf disaster at bs1 (1 req x 4 kv-heads x 57 layers/token). | kernel + impl | high (for perf): add a kv-head-indexed `block_ids` mode + per-request block_table so the batch runs in one launch. The perhead kernel already has a per-head bids stride (sm120_fmha_perhead.cu:710-744) to crib from. |
| D | **Indexer score-only paged entrypoint MISSING.** `idx_q` arrives pre-projected/normed/roped; we need only score+maxpool. But the only pybind is `block_scores` = the FULL project+norm+rope pipeline from hidden states + weights. No `index_block_scores(idx_q, paged_index_k, block_table, ...)` exists. | kernel | high: new entrypoint exposing `block_score_hmma` over a paged index-K. |
| E | **Indexer K paged vs dense.** vLLM index-K is paged `[num_blocks,128,head_dim]` via block_table; our score kernel reads a dense `k_idx[N,128]`. The new entrypoint (D) must take the paged index-K + block_table + cu_seqlens/prefix_lens. | kernel | high (folds into D). |
| - | topk fixed at 16 (sm120_sparse_topk.cu:24). M3 config `index_topk_blocks=16` -> **matches**, no blocker for this model. |  | none |
| - | fp8 KV unsupported by the bf16 paged kernel. M3 main attn is bf16 -> bf16 is the serving path; selector gates fp8 to Triton. |  | none |

M3-NVFP4 config confirmed: 60 layers, 64 q-heads, **4 kv-heads**, head_dim 128,
`index_n_heads=4`, `index_block_size=128`, `index_topk_blocks=16`,
`index_local_blocks=1`. (3 dense MLP + 57 sparse MoE layers.)

**Why we stopped here.** A,B alone are medium kernel rewrites; C is needed for
the decode path not to be catastrophically slow; D+E are an essentially new
indexer kernel. Landing AND validating all of that in one GPU slot -- on the same
box that must stay serving -- is too deep and too risky for the payoff, given the
thesis (the only perf upside) is already disproven. The honest call is partial
integration + this gap list + restore marlin.

---

## 5. Serving benchmark

Only the **marlin baseline** was benched (the MSA attend swap is blocked, so
there is no MSA-served config to compare). Baseline, captured live before any
work and re-verified coherent after:

| metric | marlin baseline |
| --- | --- |
| decode bs1 | **91.08 tok/s** (TPOT 10.98 ms/tok, TTFT 93 ms) |
| prefill (~512 tok) | 5443 tok/s |
| coherence | "capital of Poland" -> " Warsaw."; 17x23 -> 391 |

No MSA-served numbers: would be dishonest to fabricate a comparison for a config
that cannot run. Full JSON: `results/marlin_baseline.json`.

---

## 6. Deliverables in this dir

- `kernels/` -- the four validated `.cu` + `include/` (build with `_loader.py`).
- `_loader.py` -- container-validated JIT loader (`prepare_build_env()` codifies
  the cusparse/cusolver symlink + `sm_120f` flag).
- `op_equivalence_topk.py` + `results/op_equivalence_topk.txt` -- the topk
  op-equivalence proof.
- `sm120_sparse_impl.py`, `sm120_indexer_impl.py`, `patches.py`,
  `selector_patch.md` -- the drafted bridge (gaps A-E flagged inline).
- `results/marlin_baseline.json` -- baseline serving numbers.
- `_vllm_ref/minimax_m3/` -- the exact image overlay source studied (for the
  file:line cites above).

## 7. Recommended next steps (if resumed)

1. Land blockers A+B (page-128 + fused 5-D cache) -> then attend op-equivalence
   for `forward_sparse_paged` vs `minimax_m3_sparse_attn` becomes runnable
   (expect attend ~0.035 rms FP8-PV floor).
2. Land C (one-launch GQA) before any serve attempt -- the per-(req x head)
   Python loop will lose badly otherwise.
3. Land D+E (paged score-only indexer entrypoint) for the full impl swap.
4. Reframe the win: pursue **NVFP4-KV attention** (the real differentiator on
   SM120), not eager-break removal.
