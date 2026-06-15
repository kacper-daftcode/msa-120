# MiniMax-M3-NVFP4 SM120 serving — custom-kernel opportunity map (adversarially verified)

Workflow: 7-component parallel audit -> adversarial ROI refutation -> synthesis. Verdict: every bs1-decode component is SKIP (max adjusted ROI 8/100); the gap to the weight-BW floor (4.7x) is latency (PCIe all-reduce + dequant + launch), all hardware/structural floors. The only unproven regime is throughput/concurrency.

## Per-component (audit ROI -> adversarial adjusted ROI)

| component | token% | runs on | floor | ROI->adj | verdict | measured killer |
|---|---|---|---|---|---|---|
| all-reduce | 17.4% | ncclDevKernel_AllReduce_Sum_bf16_RING_LL | latency | 62->8 | skip | The ROI's load-bearing assumption — a "~1-2us PCIe round-trip latency floor" enabling 13.8us -> 3-4us — is empirically false on THIS box. Measured on the actual 4x RTX PRO 6000 (all-pairs PHB, no NVLi |
| dense-gemm-projlinear | 43% | MarlinNvFp4LinearKernel (W4A16), pinned  | latency | 22->8 | skip | The two pillars of the proposal are both already exhausted or off the critical path, verified in-repo. (1) Launch-fusion is empty: minimax_m2.py (the M3 arch) already uses QKVParallelLinear (q/k/v = O |
| moe-marlin-gemm | 15.6% | vLLM marlin W4A16 MoE: `ops.moe_wna16_ma | bandwidth | 8->2 | skip | At the audited bs1-decode regime a custom SM120 kernel beats stock by ~0%, and the audit's own numbers prove it: measured MoE marlin = 1.696 ms/token vs the @80% HBM weight-stream floor of 1.670 ms =  |
| moe-route-permute | 11% | vLLM STOCK precompiled kernels in _moe_C | latency | 14->3 | skip | The serving win is near-zero and likely a net regression, not the claimed ~3.5%. (1) The 820us/token route and 405us moe_align figures come from captureB, which SERVING_LEVERS.md and RECIPE.md explici |
| norm-act-quant | 8.4% | Four separate stock kernels per layer, a | latency | 14->7 | skip | The headline fusion (swigluoai act_and_mul + cvt_fp16_to_fp4 in one kernel, eliminating a bf16 HBM round-trip) targets a dataflow pair that DOES NOT EXIST in the shipped config. I traced the actual M3 |
| msa | 7% | Triton `_decode_index_score_kernel` in v | bandwidth | 8->2 | skip | The win is structurally unreachable at bs1 and almost certainly NET-NEGATIVE, for a reason stronger than the audit's own "tiny share" framing. (1) Token share confirmed but tiny: the live captureB nsy |
| dense-attn-fmha | 3.2% | vLLM FlashAttention v2 (FLASH_ATTN backe | latency | 6->4 | skip | For token throughput the net win is ~0-to-negative, and this is not a model but a MEASURED, directly-analogous outcome: SERVING_INTEGRATION.md shows our SM120 sparse decode-attend was fully integrated |

## Synthesis / marathon plan

This is a synthesis task. All the audit and adversarial-verify data is provided inline. No file system investigation is needed — I just need to fold the objections into a decisive plan.

# MiniMax-M3-NVFP4 SM120 Custom-Kernel Optimization Marathon

**Baseline: 90.8 tok/s decode bs1. Gap to weight-BW floor: 4.7x, entirely latency (120 PCIe all-reduces/token + dequant + launch). GPU already 92.2% busy under FULL cudagraph capture — so launch overhead is *already amortized to ~0*; only on-device execution latency and inter-kernel gaps remain.**

## 1. Component Audit Table

| Component | Token % | Runs on | Ours faster? | Floor | Adjusted ROI | Verdict |
|---|---|---|---|---|---|---|
| dense-gemm-projlinear | 43% | stock (Marlin W4A16) | no kernel | latency (HBM weight-stream wall at M=1) | **8** | **SKIP** |
| all-reduce (TP4) | 17.4% | stock (PYNCCL ring) | no kernel | latency (PCIe/PHB, ~9µs/leg) | **8** | **SKIP** |
| moe-marlin-gemm | 15.6% | ours-but-slower (cutlass W4A4 23x slower) | no | **bandwidth (1.02x off floor)** | **2** | **SKIP** |
| moe-route-permute | 8–11% | stock | no kernel | latency (tiny-grid) | **3** | **SKIP** |
| norm-act-quant | 8.4% | stock | no kernel | latency (launch) | **7** | **SKIP** |
| msa (indexer-score) | 7% (score 1.46%) | ours-not-integrated | no | bandwidth (long-ctx) / latency (bs1) | **2** | **SKIP** |
| dense-attn-fmha | 3.2% | ours-not-integrated | no | latency (wave-bound) | **4** | **SKIP** |

**Every component verdict is SKIP for bs1 decode tok/s.** This is the honest read of the data, and it is consistent: max adjusted-ROI in the stack is 8/100.

## 2. Ranked Marathon Order (by expected serving tok/s gain)

The ranking below is ordered by *realistic, objection-adjusted* expected gain. Every entry's headline number is the **adversarially-corrected** estimate, not the auditor's optimistic one — and the cheapest kill test comes **before** any kernel work, because in every case the kill test is expected to fire.

### Rank 1 — All-reduce one-shot P2P (the headline candidate — evidence says NO)
- **Kernel to build:** SM120 one-shot P2P all-reduce (or force-enable vLLM's existing `cross_device_reduce_1stage`, `csrc/custom_all_reduce.cuh:298`).
- **Technique:** flag-barrier → direct `ld.128` P2P reads of 3 peers' 12KB staging buffers → local fused reduce → barrier, graph-captured.
- **Auditor's claim:** 13.8µs → 3-4µs/op, ~3.7x component, +13% e2e (→106-108 tok/s).
- **Realistic gain (objection-folded): ≈ 0%, likely negative.** The load-bearing "1-2µs PCIe round-trip floor" is **empirically false on this box**. Measured: single 12KB peer read GPU0←GPU1 = **9.2µs** GPU-side; 3 peers sequential = 27µs, parallel = 43µs (root-complex contention makes overlap *worse*). PHB routes P2P through the CPU root complex, not a switch. A correct one-shot needs ≥1 peer-read round-trip + ≥1 barrier round-trip = **well over 13.8µs**. vLLM's `should_custom_ar` has a *second, independent* gate (`custom_all_reduce.py:237`) with the comment "for 4 or more non NVLink-capable GPUs, custom allreduce provides little performance improvement over NCCL" — a measured perf judgment, not the bypassable NVLink-only `is_fully_connected` gate the auditor leaned on. The 37x figure is a *bandwidth* statement (3MB hits 48.7 GB/s); as a *latency* floor it is ~9µs, not 0.37µs.
- **Cheapest prove-or-kill (~1 hr, no kernel):** force-enable the shipped one-shot AR (`disable_custom_all_reduce=False`, patch the two ws>2/PHB gates, `VLLM_SKIP_P2P_CHECK=1`, `VLLM_CUSTOM_ALLREDUCE_ALGO=1stage`), microbench one 12KB bf16 all-reduce across 4 TP ranks vs `ncclAllReduce RING_LL`, CUDA-event timed inside a captured graph, 1000+ iters. **Kill criterion: if custom-1stage per-op ≥ ~13.8µs, skip permanently.** The standalone probe already run (9.2µs/leg, 27µs 3-peer) is the 5-minute version and *already predicts skip*.

### Rank 2 — Dense GEMM fused dequant+GEMV (the 43% candidate — evidence says NO)
- **Kernel to build:** SM120 fused NVFP4-dequant→GEMV for M=1 W4A16 dense linears.
- **Technique:** in-register e2m1 LUT decode (skip bf16 weight materialization + WMMA-on-M=1), cp.async double-buffered weight stream, FP32-accumulate CUDA-core GEMV; plus stacked-projection fusion.
- **Auditor's claim:** ~1.2-1.4x component, +9-10% e2e (→101 tok/s).
- **Realistic gain (objection-folded): ≈ 2-4%, inside run-to-run noise (baseline already swings 90.7–92.3).** Three pillars all collapse: **(1) Launch-fusion is empty** — `minimax_m2.py` (the M3 arch) already uses `QKVParallelLinear` (qkv = ONE GEMM) and `gate_up_proj` (ONE GEMM); each attn/MLP layer is already maximally fused at 2 linears. The auditor's "114→60" fusion recovers nothing. **(2) The launch tax is a cudagraph *replay*, not a host-launch tax** — SERVING_LEVERS Lever D proved `-O3`/inductor was a no-op because cudagraphs already remove launch overhead. **(3) The WMMA-skip is illusory** — bs1 is at the weight-streaming wall; a CUDA-core GEMV still streams the identical 3.30 GB/GPU, so the wasted tensor-core lanes are hidden *behind weight-load latency* and were never on the critical path. Prior `decode_kernel/RESULTS.md` marathon: fuse-to-cut-launches *regressed* 8.1→15-17µs (wave starvation), best hand-tuned kernel stayed 0.6-0.86x of reference (a LOSS), bs1 gap concluded **structural**.
- **Cheapest prove-or-kill (~1 nsys query, zero kernel):** on the existing `captureB_serve.nsys-rep`, restricted to pure-bs1-decode tokens, measure per dense-GEMM: (a) inter-kernel **gap** before each marlin launch, (b) the GEMM's **DRAM throughput %**. **Kill: gaps ~0 (graph replay) AND DRAM% high (HBM-bound) ⇒ both levers dead.** Same query confirms qkv/gate_up are already single GEMMs (count distinct marlin launches/layer = 2 attn + 2 MLP), refuting the fusion claim instantly.

### Rank 3 — norm-act-quant fusion (8.4% — evidence says the fusion pair *doesn't exist*)
- **Auditor's claim:** ~1.3x bucket, +1.9% e2e.
- **Realistic gain: ≈ 0%.** The headline fusion (swigluoai act + `cvt_fp16_to_fp4` in one kernel, removing a bf16 round-trip) targets a dataflow pair **that does not exist in the shipped config.** The W4A16 dense method is explicitly "No activation quantization" (consumes bf16); marlin MoE takes bf16 activations (`get_marlin_input_dtype()=None`) and runs swigluoai *inside* the kernel. The `cvt_fp16_to_fp4` in the trace is the **indexer/MLA KV-cache quant**, a separate attention-side dataflow — not an MLP activation feed. There is nothing to fuse into. Residual win collapses to one fewer standalone bf16 act launch — pure launch-count, already amortized by the cudagraph.
- **Cheapest prove-or-kill (5 min, no kernel):** grep the live per-layer op sequence; confirm the `cvt_fp16_to_fp4` is the indexer/MLA KV quant, not an MLP act→fp4 adjacency. **Kill on confirmation — dead on arrival.**

### Rank 4 — moe-route-permute fused route+align (8-11% — evidence says net regression)
- **Realistic gain: ≈ 0 to negative.** The 820µs/405µs figures come from `captureB`, a **mixed prefill+decode** capture; `moe_align` cost scales with tokens-in-flight, so the route share is prefill-inflated. At true bs1 decode (M=4 padded) align is far cheaper. GPU is 92.2% busy / 7.8% gap shared across 350+ kernels (tens of ns/gap under capture). **Decisive prior art:** a set-exact, 2-3x-faster custom topk kernel **regressed decode -3.1% (-6.5% at c=1)** purely from per-layer host-op feed across 57 captured layers. No integration hook exists (route kernels are intrinsic to forced-marlin; the un-fused-swigluoai path that would host a custom route kernel currently outputs **GIBBERISH**, RECIPE.md).
- **Cheapest prove-or-kill:** pure bs1 decode-only nsys capture; if true decode route cost ≪ 820µs (expected), dead. Upper-bound probe: stub `moe_align_block_size` to a no-op in-graph, measure tok/s delta — if within 2% noise, skip.

### Rank 5 — dense-attn-fmha paged-128 decode (3.2% — evidence says parity-at-best)
- **Realistic gain: ≈ 0 to slightly negative.** Directly-analogous: our SM120 sparse decode-attend was fully integrated, graph-captured (51/51 FULL, 0 invalidations), coherent — and still netted **-3.1%** (89.1 vs 91.9), because the wall is the bs1 64-work-unit / 0.34-waves / 25%-SM-active ceiling (W4 occupancy bought zero; DRAM 0.16% warm; MMA feed byte-identical to Triton). Stock FA2 splitkv is already at that latency-optimal point. The fp8-KV lever is a **2x-context capacity play, zero tok/s** by the auditor's own admission, and was removed (RECIPE blocker #5) because it forces FlashInfer/trtllm-gen/SM100.
- **Cheapest prove-or-kill (minutes):** no-op the 3 dense layers' attend compute in the live server, measure tok/s vs 91.9. If deleting attend entirely is <~1% (near-certain at 3.2%-of-token), no kernel can win.

### Rank 6 — moe-marlin-gemm W4A4 (15.6% — hard bandwidth floor)
- **Realistic gain: 0%.** Measured 1.696 ms vs 1.670 ms @80% HBM floor = **1.02x** — genuinely at the bandwidth floor. Both marlin and W4A4 stream the identical 2.39 GB/GPU; at M=1 FP4 tensor cores buy nothing (no compute). Prior cutlass W4A4 head-to-head: **0.043x at bs1 (23x slower)**, OOMs on graph capture. roi_score=8 is internally inconsistent with the audit's own "~1.0x — NONE." A custom kernel beats stock by ~0% here.
- **Cheapest prove-or-kill:** one nsys profile of the live bs1-decode marlin server; if MoE-marlin ≥0.95x floor and inside the captured graph (both strongly evidenced), dead.

### Rank 7 — msa indexer-score paged split-K (1.46% — ROI-negative)
- **Realistic gain: ≈ -6%-class regression.** Even a perfect-to-floor 1.5x saves <0.5% gross. The killer: Triton's `minimax_m3_index_decode` is ONE fused graph-captured op; swapping only the score sub-stage **splits** it, forcing the same per-layer host-feed x57 captured layers that netted **-6%** on the *larger* decode-topk win. The 1.5x is the seq=16384 tail; real per-launch avg (3.04µs across seq mix) has smaller headroom, on the same bs1 wave ceiling.
- **Cheapest prove-or-kill (~1 hr, no CUDA):** wiring null-test — split the fused op but keep Triton's *own* fast score as a separate captured op feeding Triton's topk, run decode A/B. If merely splitting regresses (precedent predicts yes), any custom score inherits the loss. Kill.

## 3. Classification: build-new vs fix-ours vs hard-floor-skip

**(A) Build a NEW custom kernel — none are worth building for bs1 decode tok/s.**
The two greenfield headline candidates (all-reduce one-shot, dense GEMV) both fail their kill tests on this specific box (PHB 9µs/leg latency; cudagraph already amortizes launch + HBM weight-stream wall). norm-act-quant and route-permute new kernels target dataflow that is already fused or doesn't exist.

**(B) Ours exists but needs faster/integration — all regress at the bs1 wave ceiling.**
- `moe-marlin` (cutlass W4A4): exists, 23x slower, OOMs capture, and the regime is bandwidth-bound so even a perfect port = 0%.
- `dense-attn` / `msa-score` (sm120 FMHA / block_score_hmma): exist but prefill-shaped, not paged-128-decode drop-ins; the analogous integrated kernel already measured **-3.1% to -6%** from per-layer host-feed across 57 captured layers. Integration tax > kernel win.

**(C) Stock is optimal / hard floor — SKIP (this is everything, for bs1 decode).**
- **dense GEMM**: cudagraph-replay (not launch) bound + HBM weight-stream wall; structurally capped (`decode_kernel/RESULTS.md`).
- **all-reduce**: PCIe/PHB ~9µs/leg latency floor; NCCL ring already near-optimal for 4-PCIe small-msg (vLLM's own measured gate confirms).
- **MoE marlin**: 1.02x off the bandwidth floor.
- **norm-act-quant**: the fusion pair doesn't exist in the shipped config.

## 4. The SINGLE Highest-ROI Target to Start

**Start with the all-reduce one-shot AR microbench — NOT to build it, but because it is the cheapest, most self-contained, highest-leverage *kill test* in the stack, and it is the one place the marathon premise ("our SM120 kernel beats stock") is even theoretically defensible.**

**Justification:**
- It is the **only** component where a custom kernel uses a *genuinely different mechanism* (bypasses the ring entirely) rather than chasing FLOPs/launches the cudagraph already hides.
- It is **fully self-contained** (one collective, no model-correctness/gibberish risk, no 57-layer host-feed integration tax that sank every "ours" kernel).
- vLLM **already ships the exact kernel** (`cross_device_reduce_1stage`), so the prove-or-kill is **force-enable + microbench, ~1 hr, zero CUDA written**.
- It has the **highest auditor ROI (62) and the cleanest single decisive measurement** — so resolving it first either unlocks the only real +13% e2e win in the stack, or kills the marathon's headline thesis for an hour of work.

**First experiment (the decisive kill test):**
1. Launch with `disable_custom_all_reduce=False`; patch the two gates (`custom_all_reduce.py:148` ws>2 `fully_connected`, `:239` `should_custom_ar`) to allow ws=4 PHB; set `VLLM_SKIP_P2P_CHECK=1`, `VLLM_CUSTOM_ALLREDUCE_ALGO=1stage`.
2. Microbench a single 12KB bf16 all-reduce across the 4 TP ranks: custom-1stage vs `ncclAllReduce RING_LL`, CUDA-event timed **inside a captured graph**, 1000+ iters.
3. **Kill criterion: custom-1stage per-op ≥ ~13.8µs ⇒ skip permanently.** The standalone P2P probe already run (12KB GPU0←GPU1 = 9.2µs, 3-peer = 27µs) is the 5-minute version and **already indicates skip** — so run the in-graph microbench to confirm, and if it lands ≥10µs (near-certain on PHB), **conclude the marathon: there is no custom-kernel bs1-decode tok/s win on this box, and effort should pivot to prefill/concurrency (where FP4 tensor cores and BW-bound batched scores actually pay off) or to the throughput regime rather than bs1 latency.**

---

**Bottom line (honest):** The evidence does **not** support either headline candidate. All-reduce is killed by the measured ~9µs/leg PHB latency (the "1-2µs floor" is false). Dense GEMM is killed because its two levers (launch-fusion, WMMA-skip) target costs that the cudagraph and the HBM weight-stream wall have already neutralized — and qkv/gate_up are *already* fused. Every other component is a bandwidth floor, a non-existent dataflow, or a -3% to -6% integration regression at the bs1 wave ceiling. **The decisive, intellectually-honest marathon plan is: run the ~1-hour all-reduce kill test first; when it fires, stop optimizing bs1 decode and redirect the marathon to prefill/concurrency**, where compute-bound FP4 and BW-bound batched kernels make "our SM120 kernel beats stock" actually true.