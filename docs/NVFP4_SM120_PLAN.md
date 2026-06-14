# Plan: fast MiniMax-M3-NVFP4 serving on SM120 (4x RTX PRO 6000)

Goal: serve M3-NVFP4 via vLLM on SM120 — correct (swigluoai) and fast (native
NVFP4 MoE + our SM120 MSA attention) — benchmarked vs the MXFP4 baseline (102 tok/s).

## State
- Phase 1 (MXFP4): works, 102 tok/s (baseline).
- MSA kernel suite (this repo, 21 commits): indexer (30x HMMA), top-k (golden-exact
  vs B200), block-sparse forward (causal/per-query/per-head/BLK128/paged), e2e chain,
  golden harness. All validated on SM120.
- NVFP4 blocker map (from live investigation):
  - #1,#2 config layer_types/hidden_act -> trivial patches (DONE).
  - #3 swigluoai NVFP4 **MoE** -> the real kernel gap. b12x IS in the image
    (flashinfer 0.6.12) but does plain SiLU (`_supports_activation==SILU`); only
    FLASHINFER_TRTLLM applies the clamp and needs SM100. => no swigluoai NVFP4 MoE
    backend for SM120.
  - #4 weight key naming (checkpoint `model.language_model.*`/`mlp`/`w1w3w2` vs vLLM
    `language_model.model.*`/`block_sparse_moe`/`gate,up,down`).
  - #5 fp8 KV default (broken on SM120) -> bf16 KV.

## Two orthogonal compute axes
1. Attention = MSA  <- OUR work (kernels ready, validated). Currently vLLM uses the
   Triton MSA fallback on SM120; our kernels are the drop-in perf upgrade (M8).
2. MoE = NVFP4 experts  <- the current blocker (#3, swigluoai). NOT our kernels.

## Workstreams
- WS-0 housekeeping: push msa-120 (user auth); this doc.
- WS-A load checkpoint (#1,#2,#4,#5): config patches (done) + weight remapper + bf16 KV.
  Owner: me (iterative, live container). Result: checkpoint loads.
- WS-B NVFP4 MoE (#3):
  - B1 (fast signal): force plain-SiLU b12x, serve, measure quality.
  - B2a (correct, cheap): un-fused vLLM MoE = working SM120 NVFP4 linear GEMMs +
    swigluoai in torch between them. No new PTX. Slower, correct.
  - B2b (correct, fast): patch fused b12x to apply swigluoai (flashinfer CuTe-DSL).
- WS-C MSA integration (M8, our work): SM120 impl adapters + selector patch +
  package kernels in image + op-equivalence vs Triton. Optional for "works",
  required for "fast".
- WS-D e2e + benchmark: serve [B] + [Triton or our MSA]; quality vs MXFP4; tok/s.
- WS-E perf polish: forward ldmatrix/occupancy, indexer, consolidation; MoE perf.

## Phases
- alpha (load & run): A + B1 + Triton-MSA -> first e2e NVFP4 (plain-SiLU quality).
- beta (correct): + B2a -> correct swigluoai quality.
- gamma (fast attention): + WS-C -> our MSA replaces Triton.
- delta (polish): + WS-E + B2b.

## Parallelization
me: WS-A (live). agents: B2a torch ref + design; C1 adapters draft; forward perf (E1).

## Risks
plain-SiLU quality magnitude (B1 decides); un-fused MoE perf (B2a); our kernels v1
(polish in E); bf16-KV context ceiling; vLLM version drift.

## Validation
every step numeric vs reference (torch / MXFP4 baseline / golden B200); e2e coherent
generation + benchmark; every commit with a test.
