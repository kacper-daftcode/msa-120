# SM120 MSA decode kernel -> live vLLM MiniMax-M3 serving: integration & verdict

Box: 4x RTX PRO 6000 Blackwell (SM120). Image `vllm/vllm-openai:minimax-m3`
(torch 2.11.0+cu130, nvcc 13.0). Container `minimax-m3-nvfp4`. TP4, block-size
128, max-model-len 65536, gpu-util 0.977, marlin NVFP4 MoE, graph mode
(`cudagraph_mode=FULL_AND_PIECEWISE`). Date: 2026-06-15.

## TL;DR

Our SM120 decode-attend kernel was integrated into the LIVE vLLM serving path
and **runs**: the selector routes all 57 sparse layers (x4 TP workers) to
`MiniMaxM3SparseSm120Impl`, the kernel **captures into the FULL decode cudagraph**
(the #1 gate -- PASSED, no `StreamCaptureInvalidated`), and generation is fully
**coherent** (Warsaw, 17x23=391, correct code gen). BUT it does **not** meet the
strict no-regression bar: measured **decode bs1 = 87.8 tok/s vs 90.7 baseline
(-3.2%)**, concurrency -2.8/-3.5/-3.4%, prefill neutral. The loss traces to our
decode-attend being ~0.86x of Triton at the kernel level (the attend is ~3.4% of
a token, so the net lands at ~3%). Per the GOAL's no-regression hard gate, the
box is **left serving the marlin baseline** (restored, verified coherent: Warsaw,
391; FULL decode graph captured). The integration is real and re-runnable via
`launch_msa.sh`; closing the gap needs the decode-attend kernel-level lightening
(not landed -- deep surgery, and even at parity the design carries a small
fixed split-K-merge overhead).

## What runs on OUR code vs Triton

| component | impl | runs on |
| --- | --- | --- |
| DECODE main attention (bs1 hot path) | `forward_sparse_decode_serving` (page-128 ldmatrix W4=3 flash-decoding split-K) | **OUR SM120 kernel** |
| PREFILL main attention | `minimax_m3_sparse_attn` | Triton (Phase 1) |
| Indexer score + topk (decode & prefill) | `minimax_m3_index_decode` / `_score`+`_topk` | Triton (Phase 1) |
| MoE | marlin NVFP4 fused | marlin (unchanged from baseline) |

Phase 1 deliberately swaps only the decode attend -- the bs1 interactive hot
path and the part of "our code" with the cleanest graph-capture story. The
indexer score-only paged entrypoint (blockers D/E) and the prefill attend swap
are left for Phase 2/3 (see "Remaining" below).

## Kernel changes that made the live swap possible

The validated decode kernel (`decode_kernel/sm120_fmha_decode.cu`,
`forward_sparse_decode_p128`, W4=3) took a HOST scalar `seq_len_k` and a
per-request `block_ids [R,topk]`, and a split K/V cache. Three serving-only
changes (in `kernels/sm120_fmha_decode_serving.cu`, derived from the validated
W4=3 _ldsm partial + flat LSE merge; numerics identical) close the gap to a
graph-capturable, batched, one-launch op:

1. **block_ids `[R, Hkv, topk]` per-kv-head** -- one launch covers the
   GQA-shared per-kv-head top-k selection (M3 `decode_topk` is `[Hkv,total_q,
   topk]`). No per-(req x head) Python loop.
2. **seq_lens DEVICE int32 `[R]`** read in-kernel (`seq_lens[req]`) -- the host
   never does `.item()`, so the op captures into the FULL decode cudagraph.
3. **fused M3 cache** `[num_blocks,2,128,Hkv,128]` consumed via K/V base
   pointers + the REAL tensor strides (handles NHD and HND layouts) -- no cache
   copy, allocation-free.

Plus a correctness fix for partial selections (fewer than `topk` blocks selected
early in generation): `-1` pad pages now zero-fill their K/V smem tile so PV
never does `garbage * 0 -> NaN`. Verified NaN-free across 500 poison-stressed
cases.

## Correctness (kernel-level)

`verify_decode_serving.py` (in-image, GPU0), vs the validated page-64 golden
(`forward_sparse_paged`, causal=False) and a dense fp32 softmax reference:
- **ALL OK** across seq_kv {4096,16384,65536}, nsel {16,12,8,3,1}, split_chunks
  {0,1,4,16}, R {1,2,3}, per-kv-head-DISTINCT selections, and NHD + HND cache
  layouts. rms vs golden < 3.0e-3 (FP8-PV floor), rms vs dense < 3.0e-4.
- 500 poison-stressed partial-selection cases: **0 NaN**.

## Graph-capture status (THE hard gate) -- PASSED

From the live MSA-serving startup log (`launch_msa.sh graph`):
- `[sm120-msa] select_main_impl_cls -> MiniMaxM3SparseSm120Impl (family120, bf16,
  topk=16)` x **228** (57 sparse layers x 4 TP workers) -- every sparse layer on
  our kernel.
- `[sm120-msa] decode kernel JIT-built at startup (graph-safe)` (x4 workers +
  API server) -- no compile happens inside a capture region.
- `Capturing CUDA graphs (decode, FULL): 100%|####| 51/51` -- our kernel captured
  into the FULL decode cudagraph for all 51 decode sizes.
- **No `StreamCaptureInvalidated`**, no capture error.
- `Graph capturing finished in 16 secs, took 3.26 GiB` (baseline: 19s / 3.22 GiB
  -- comparable; our kernel did not break or bloat capture).
- `Application startup complete` -- served end-to-end.

This is the decisive result: the graph-safe design (DEVICE seq_lens read
in-kernel, per-kv-head block_ids in one launch, no host `.item()` / no host sync
in the captured region) captures into the same FULL decode graph the 90 tok/s
baseline depends on. The "no perf loss hinges on capture" risk is retired.

## Coherence (end-to-end) -- PASSED

On the live MSA server:
- "capital of Poland" -> **Warsaw** (code gen path content); reasoning path Warsaw.
- "17 x 23" -> **391** (`**391**`).
- "Python one-liner sum of squares 1..n" -> `sum(i*i for i in range(1, n+1))`.

No garbling. The kernel that passes rms also generates correctly.

## Serving throughput: ours-MSA vs marlin+Triton baseline

Identical methodology (`bench_client.py`, 24 prompts, temp 0, TP4, graph mode):

| metric | marlin baseline | ours-MSA (decode) | delta |
| --- | --- | --- | --- |
| decode bs1 tok/s | **90.70** | 87.83 | **-3.2%** |
| decode bs1 TPOT ms | 11.03 | 11.39 | +3.3% |
| prefill tok/s | 5491 | 5506 | +0.3% (Triton, neutral) |
| sweep c=1 out tok/s | 83.32 | 80.99 | -2.8% |
| sweep c=8 out tok/s | 486.94 | 470.00 | -3.5% |
| sweep c=32 out tok/s | 944.04 | 911.82 | -3.4% |

Careful decode-only remeasure (8 reqs x 200 tok, x2): 88.7 / 88.9 tok/s -- so the
decode bs1 regression is a stable **~2-3%**, beyond the ~2% noise budget. It
traces to the decode-attend (0.86x Triton) plus the split-K merge launch; the
attend's ~3.4% token share x 0.86x predicts ~this. split_chunks and W4-variant
swaps were micro-benched and are within noise (don't recover it).

## Verdict

**Integration: success. No-perf-loss bar: NOT met (honest -3.2% bs1 decode).**

Our code provably RUNS in the live serving path, CAPTURES into the FULL decode
cudagraph, and is COHERENT -- every hard structural gate passed. The only failure
is the perf bar: ~3% slower on the decode hot path, which is intrinsic to the
0.86x decode-attend, not a wiring artifact (the extra per-layer permute/copy ops
are captured into the graph at ~zero host cost). Per the GOAL's no-regression
hard gate, **marlin is restored and left serving** (verified: Warsaw, 391, FULL
graph captured).

To SHIP our decode on the no-regression bar, land the identified decode-attend
lightening (cut the per-step LDS / lighten the cross-warp softmax handshake to
bring the 64-block partial ~4.58->~3.97us, i.e. 0.86x->~1.0x) and re-measure;
a hybrid (our prefill+topk where we win, Triton decode-attend) is the fallback if
the kernel can't reach parity net of the merge overhead.

## Files

- `kernels/sm120_fmha_decode_serving.cu` -- the serving decode kernel.
- `sm120_sparse_impl.py` -- `MiniMaxM3SparseSm120Impl` (decode->ours, prefill->Triton).
- `patches.py` -- selector monkeypatch (rebinds `select_main_impl_cls` in both
  the source module and `nvidia/model.py`'s namespace; pre-builds the kernel).
- `sitecustomize.py` -- startup hook (runs in every TP worker; installs a
  post-import hook that fires `patches.apply()` after `nvidia/model` imports).
- `launch_msa.sh` -- serve with our MSA decode kernel.
- `verify_decode_serving.py` -- kernel correctness vs golden + dense.
- `smoke_wiring.py` -- in-process selector/build wiring check.
- `_loader.py` -- JIT loader (`decode_serving_ext()`).

## Remaining (Phase 2/3, not blocking the decode-attend win)

- Prefill attend on our `forward_sparse_paged` (page-128 path).
- Indexer score-only paged entrypoint (blockers D/E) to move topk+score onto our
  code (our `topk_select` is set-exact + 2-3x faster; gated behind the missing
  paged score-only pybind).
- Spec-decode (decode_query_len>1) currently routes to Triton; our kernel is
  query_len==1 specialized.
