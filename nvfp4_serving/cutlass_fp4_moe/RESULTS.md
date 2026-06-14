# W4A4 cutlass NVFP4 MoE vs marlin — head-to-head verdict

**Date:** 2026-06-14  **GPU:** 4x RTX PRO 6000 Blackwell (SM120, 96 GiB/GPU)
**Model:** MiniMax-M3-NVFP4, TP4, block-128, max-model-len 65536, bf16 KV.

## TL;DR verdict

**The W4A4 cutlass FP4 path does NOT beat marlin in any regime that actually
serves, and marlin has been restored as production.** Two hard walls block a fair
win:

1. **Graph capture OOMs.** The genuine FP4 grouped-GEMM path is real and
   numerically perfect (bit-identical to the validated per-expert loop, and rel
   RMS 0.088 vs a bf16-dequant reference on REAL tokens — below the 0.13 NVFP4
   noise floor). It is CUDA-graph-*capturable in isolation*. But in the full
   64-layer model, vLLM captures 102 graphs (51 PIECEWISE + 51 FULL). The static
   `[E=128, C]` all-experts batched buffers, multiplied across every captured
   size and layer, push total memory past 96 GiB during capture for every
   `(capacity C, gpu-mem)` combination tried — see the matrix below. So the FP4
   path can only run with `--enforce-eager`.

2. **Eager has no graph replay → ~24x slower decode.** Without graph capture,
   the per-forward Python + per-kernel-launch + host-sync overhead of the
   grouped FP4 GEMM dominates bs1 decode. Marlin hides all of this behind a
   captured graph. The FP4 kernel is ~4.1 ms/MoE-layer for a single decode token
   (kernel-launch/host-sync bound, not compute bound) → ~4 tok/s.

The honest characterization the task asked for: **FP4 compute's *relatively*
best regime is PREFILL** (it loses 3.3x there vs 23x at bs1), exactly because
prefill batches tokens so the FP4 GEMM amortizes its launch overhead — but it
still loses everywhere because it is stuck in eager.

## Head-to-head numbers

| regime | marlin (graph, prod) | cutlass W4A4 FP4 (eager) | FP4 / marlin |
|---|---|---|---|
| decode @ bs1 (tok/s) | **90.8** | 3.95 | 0.043x (23x slower) |
| prefill (tok/s, 512-tok) | **5344** | 1638 | 0.31x (3.3x slower) |
| concurrency 1 (out tok/s) | **79.4** | 3.89 | 0.05x |
| concurrency 8 (out tok/s) | **474** | 17.3 | 0.036x |
| concurrency 32 (out tok/s) | **962** | (not run; eager too slow) | — |

marlin baseline: `bench/results/marlin_graph_0977.json` (decode/prefill) and
`bench/results/marlin_graph_sweep.json` (sweep).
FP4 (bounded run): `bench/results/cutlass_fp4_eager.json`.

Both kernels load 4-bit weights, so bs1 decode is memory-bandwidth bound for
*both*; the gap there is pure framework overhead (graph vs eager), not FP4 vs
bf16 compute. The task's caveat — "bs1 decode MoE is memory-bandwidth-bound, FP4
compute likely helps prefill more" — is borne out: prefill is the only place the
FP4 path is within one order of magnitude.

## What was built (all under `cutlass_fp4_moe/`, committable)

* `batched_graphsafe_moe.py` — the W4A4 fast path: fixed-capacity static `[E, C]`
  routing table (graphsafe, no nonzero/.item()/host-tensor) feeding ONE
  multi-group `flashinfer.group_gemm_nvfp4_nt_groupwise` per projection
  (gate_up + down) over all E experts, with the SFA-offset SCATTER fix for the
  flashinfer 0.6.12 multi-group bug, swigluoai epilogue between them. Persistent
  per-plan activation buffers (reused across all cudagraph sizes) to minimise
  graph-pool memory. **Bit-identical to the per-expert loop** (vs_loop = 0.0).
* `batched_dynamic_moe.py` — the eager correctness path (no capacity drop): one
  grouped GEMM over only the ACTIVE experts, sized to live tokens. Correct for
  any batch but uses dynamic shapes (eager only). This is what actually serves.
* `modelopt.py` — `ModelOptNvFp4FusedMoE` override gated behind
  `VLLM_M3_CUTLASS_FP4_MOE=1`: keeps raw NVFP4 weights, `is_monolithic=False`,
  `supports_internal_mk=True`, builds batched (num_groups=E) mma weight scales in
  `process_weights`, and `apply` runs the dynamic (eager) or static (graph) FP4
  path. Includes a capture-safe, real-token-gated self-check
  (`VLLM_M3_UNFUSED_SELFCHECK=1`).
* `launch_cutlass.sh` — based on `launch_marlin.sh`, drops
  `VLLM_TEST_FORCE_FP8_MARLIN` + the marlin_moe.py mount, mounts the FP4 modules,
  gates behind `VLLM_M3_CUTLASS_FP4_MOE=1`. `graph` (static) or `eager` (dynamic).
* `test_batched_graphsafe.py` — numerical proof (vs per-expert loop = 0.0, vs
  bf16-dequant at noise floor). `coherence_check.sh` — live coherence probe.

## Coherence proof (live, eager FP4 server)

* "capital of Poland" → **Warsaw** ✓
* 17 × 23 → **391** ✓
* factorial(n) code-gen → correct Python ✓
* in-process self-check on REAL tokens (T=188): **rel RMS = 0.088** vs bf16
  dequant (below the 0.13 NVFP4 noise floor) ✓  — and it correctly read 0.98 at
  cap=16 because that tiny static capacity DROPS tokens, proving the check is
  not a false 0.0000 on a dummy batch.

## Graph-capture OOM matrix (why graph mode is out)

| cap C | gpu-mem | estimator | result |
|---|---|---|---|
| pad4(T*k) | 0.977 | off | OOM in mem-profiling (48 GiB single alloc — C explodes with profiling T) |
| 256 | 0.977 | off | KV = -27 GiB (buffers too big) |
| 256 | 0.977 | on | KV = -25 GiB (estimator over-reserves) |
| 64 | 0.88 | off | OOM during graph capture (graph 20/51) |
| 16 | 0.95 | off | OOM during graph capture |
| 16 | 0.97 | on | KV = -25 GiB |
| 16 | 0.93 | off | OOM during graph capture |
| 16 | 0.85 | off | OOM during graph capture |
| 16 | 0.82 | off | served, but KV (1.78 GiB) < max-len need; and cap=16 DROPS prefill tokens → rel RMS 0.98 (incoherent prefill) |

The dilemma is structural: a static capacity large enough to not drop prefill
tokens (cap ≳ 512 for T up to 8192) makes the all-E buffers too big to capture;
a capacity small enough to capture (cap=16) drops most prefill tokens and
corrupts output. There is no `(C, gpu-mem)` that is simultaneously
graph-capturable AND prefill-correct on this 96 GiB box.

## Why FP4 didn't win (root causes, honest)

1. **Memory, not math.** The genuine FP4 grouped GEMM works and is exact. The
   blocker is that the static all-experts batched buffer is memory-heavy and
   collides with vLLM's 102-graph capture footprint on top of 66.5 GiB of
   weights + KV.
2. **Launch/host-sync bound at bs1.** Decode bs1 does negligible FP4 *compute*
   (8 experts × 1 token); the cost is kernel launches + host syncs, which only a
   captured graph removes. Marlin wins by being graph-captured, not by faster math.
3. **A truly competitive FP4 path** would need either (a) a single fused FP4 MoE
   kernel (one launch, no Python per-expert loop, no per-call quantize loop) that
   is itself graph-capturable with a small static footprint, or (b) a capacity-MoE
   formulation with cap sized per captured batch and far fewer captured graph
   sizes. flashinfer 0.6.12 does not expose such a fused swigluoai FP4 MoE on
   SM120, and building one is out of scope here.

## Decision

FP4 does not clearly win → **marlin restored to production** via
`/home/kacper/launch_marlin.sh` (graph mode), per the task's instruction. The
cutlass FP4 path remains committed under `cutlass_fp4_moe/` behind
`VLLM_M3_CUTLASS_FP4_MOE=1` for future work (it is correct and serves coherently
in eager; it needs a fused/low-footprint graph-capturable kernel to be fast).
