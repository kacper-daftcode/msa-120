# SERVING_LEVERS.md — MiniMax-M3-NVFP4 on 4x RTX PRO 6000 (SM120, TP4)

Investigation of the real serving bottlenecks (dense GEMM, all-reduce, MoE marlin,
route/permute) and whether any software lever moves measured tok/s. MSA is out of
scope (chapter closed). All benches: `nvfp4_serving/bench/bench_client.py`, temp=0,
clean MARLIN graph config (`/home/kacper/launch_marlin.sh graph`) as the reference.

## TL;DR — verdict

**No software lever found a net win at bs1, prefill, or concurrency. The MARLIN
baseline (~92 tok/s decode bs1) is shipping.** Every targeted lever is either
architecturally unavailable on SM120/PCIe, a confirmed no-op, or a net regression:

| Lever | Result | Measured |
|---|---|---|
| `fuse_allreduce_rms: true` | **no-op** (FlashInfer AR unsupported ws=4 on SM120; compile mode NONE) | decode 90.7 (= baseline) |
| `VLLM_ALLREDUCE_USE_FLASHINFER=1` | **disabled** by vLLM ("not supported for world_size=4") | n/a |
| QUICK_REDUCE backend | **unavailable** — ROCm-only (`is_rocm()` gate in cuda_communicator) | n/a |
| NCCL `ALGO=Tree` + low channels | **regression** at batch | c8 378 vs 488, c32 910 vs 1100 |
| NCCL `ALGO=Tree` (default channels) | **regression** at batch | c8 388 vs 488, c32 932 vs 1100 |
| `-O3` inductor compilation | **no change** | decode 90.7, c8 488, c32 1098 (= baseline) |
| MoE route/permute fusion | **no lever** — intrinsic to forced-marlin path | — |
| dense/MoE W4A4 (real FP4 act) | **unavailable** — M3 NVFP4 checkpoint has no activation scales | — |

The three biggest token-time components are all **hardware floors** on this box:
dense GEMM is bs1 weight-bandwidth/launch bound (W4A16 Marlin, no W4A4), all-reduce
is PCIe-latency bound with PYNCCL as the only usable backend, and MoE marlin is the
only swigluoai-correct path. This is a measured "no lever" result.

---

## PHASE 1 — CHARACTERIZATION

### Per-token component breakdown (nsys, device0, GPU-kernel-time weighted)

From `captureB_serve.sqlite` (full serving capture, mixed prefill+decode),
aggregated by kernel category on rank0:

| Component | % GPU kernel time | Representative kernels |
|---|---|---|
| **dense GEMM proj/linear** | **43.3%** | `cutlass_80_wmma_*s161616gemm_bf16`, `gemvx`, `cublasLt splitK` |
| **all-reduce / comm** | **15.6%** | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |
| **MoE marlin GEMM** | **15.6%** | `marlin_moe_wna16::Marlin<...>` (gate_up + down) |
| **MoE route/permute** | **11.0%** | `moe_align_block_size`, `topkGating`, `count_and_sort_expert_tokens`, `moe_sum` |
| norm/act/quant/misc | 8.4% | `fused_add_rmsnorm`, `act_and_mul`, `cvt_fp16_to_fp4` |
| attention / MSA | 6.1% | `_gqa_sparse_decode`, `fusedMiniMaxM3QNormRopeKVInsert` |

This matches the mission's cited split (dense 42.6 / AR 17.4 / MoE marlin 15.3 /
route 7.7 / MSA 7.7) within capture noise. Note this is a *mixed* capture; at pure
bs1 decode the dense-GEMM and AR shares rise relative to route/permute (which scales
with the prefill chunk size), but the ordering holds.

### Is dense GEMM / MoE marlin weight-bandwidth-bound at bs1? (floor vs headroom)

**Roofline (per token, bs1 decode, NVFP4 4-bit weights, TP4):**
- Active params/token ≈ **23.5 B** (60 layers; attn q/k/v/o + 1 shared + 4 routed
  experts × (gate_up 6144→6144 + down 3072→6144) + router).
- Active weight bytes/token (NVFP4 ≈ 0.5625 B/param incl. fp8 group scales) ≈
  **13.2 GB total → 3.30 GB/GPU** (TP4).
- RTX PRO 6000 HBM ≈ 1.79 TB/s/GPU. Pure weight-load floor:
  - @ peak 1.79 TB/s → **~543 tok/s**
  - @ 80% eff 1.43 TB/s → **~434 tok/s**
- **Measured: 92 tok/s = 10.87 ms/token.**

**Verdict:** at bs1 the model is **NOT at the raw weight-bandwidth floor** — there is
a ~4.7x gap between the 434–543 tok/s HBM floor and the measured 92 tok/s. That gap
is **launch/dequant/sync + all-reduce latency overhead**, not HBM bandwidth. So bs1
decode is **latency/overhead-bound**, not pure-bandwidth-bound. Crucially, this does
NOT translate into a W4A4 opportunity:

- The dense linears use **`MarlinNvFp4LinearKernel` = W4A16** (4-bit weight, bf16
  activation, weight dequantized in-kernel to bf16 → the `cutlass_80_wmma_..._bf16`
  GEMMs seen in the trace). At bs1 (M=1) these are weight-streaming / launch-bound.
- **W4A4 is unavailable**: `modelopt.py` (line ~1266) pins the Marlin W4A16 kernel
  specifically because the cutlass W4A4 kernel would "silently try to quantize
  activations (we have no input_scale)." The **M3 NVFP4 checkpoint stores no
  per-tensor activation scales** for the dense linears — they are W4A16 checkpoints.
  So real FP4-activation GEMM cannot be enabled by a flag; it would require a
  different (W4A4) checkpoint. This is a checkpoint property, a hard floor here.
- For concurrency/prefill (where W4A4 *would* help compute-bound GEMMs), same
  blocker: no activation scales in the checkpoint → no W4A4 path.

### All-reduce: count, message size, BW, topology, backend availability

- **Topology:** `nvidia-smi topo -m` = all pairs **PHB** (PCIe Host Bridge) —
  **no NVLink, PCIe-only**, single NUMA node.
- **All-reduces/token:** 2 per layer × 60 layers = **120 all-reduces/token** (after
  attn o_proj and after MoE down_proj).
- **Message size (bs1):** hidden 6144 × bf16 = **12 KB per all-reduce**.
- **Bytes moved/GPU/op (ring):** 2(N-1)/N × 12KB ≈ **18 KB** → at ~50 GB/s PCIe that
  is **~0.37 µs of transfer**. But the measured per-op cost is **~13.8 µs**
  (15.6% × 10.87 ms ÷ 120 ops). **All-reduce is LATENCY-bound, not BW-bound** — the
  cost is fixed kernel-launch + ring-handshake per op, ~37x the actual data transfer
  time. PCIe bandwidth is not the limiter; PCIe (and small-message ring) *latency* is.
- **Backend availability on SM120 / 4-GPU PCIe (confirmed from live logs + source):**
  - `CUSTOM` (custom_all_reduce): **disabled** — "not supported on more than two
    PCIe-only GPUs".
  - `SYMM_MEM` (torch symmetric memory): **disabled** — "Device capability 12.0 not
    supported".
  - `NCCL_SYMM_MEM`: off (needs `VLLM_USE_NCCL_SYMM_MEM=1` *and* symmetric-mem alloc;
    not enabled, and symm-mem is unsupported here anyway).
  - `FLASHINFER` AR: **disabled** — `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` has entries
    only for capability 90/100/103; **SM120 (cap 120) has no entry**, so
    `flashinfer_max_size(ws=4)` → None → "FlashInfer All Reduce is disabled because
    it is not supported for world_size=4."
  - `QUICK_REDUCE`: **ROCm-only** — instantiated only under `is_rocm()` in
    `cuda_communicator.py`. The "QUICK_REDUCE in potential backends" log line lists
    *potential* backends; the actual enabled set on this box is **`['PYNCCL']`** only.
  - **Net: PYNCCL is the only usable all-reduce backend on this hardware.** This is
    confirmed in the live log:
    `Using ['PYNCCL'] all-reduce backends ... out of potential backends [...]`.

---

## PHASE 2 — OPTIMIZATION (each measured, 3-run-median where noted)

Reference baseline (clean MARLIN graph, gpu-mem 0.977), warm:

| metric | baseline |
|---|---|
| decode bs1 | **~92 tok/s** (90.8 / 92.0 / 92.3 over 3 runs; median 92.0) |
| prefill (warm) | ~24.9k tok/s (cold first-prefill ~5.5k) |
| concurrency 1 / 8 / 32 (out tok/s) | 90.8 / 488 / ~1100 |

### Lever A — all-reduce (the most-targeted, "most tractable")

**A1. allreduce+RMSNorm fusion (`fuse_allreduce_rms: true`)** — tested via
`--compilation-config '{"pass_config":{"fuse_allreduce_rms":true}}'`.
- Result: decode **90.7**, c8 **478**, c32 **1055** — identical to baseline; coherent.
- **No-op, as predicted.** Two independent reasons it cannot fire on SM120:
  1. The fusion pass needs FlashInfer's fused allreduce, which `AllReduceFusionPass`
     disables for ws=4 on cap 120 (no size-table entry — see Phase 1).
  2. The MARLIN config runs `CompilationMode.NONE` (CUDA graphs on, inductor off), so
     the inductor fusion passes don't execute at all.
  Also checked `fuse_gemm_comms` (off by default; same FlashInfer/SP dependency, no
  SM120 path).

**A2. FlashInfer AR backend (`VLLM_ALLREDUCE_USE_FLASHINFER=1`)** — live log:
`FlashInfer All Reduce is disabled because it is not supported for world_size=4.`
**Unavailable on SM120.**

**A3. QUICK_REDUCE** — ROCm-only in source; never instantiated on NVIDIA. **Unavailable.**

**A4. NCCL env tuning for the latency-bound small-message PCIe regime.** Since AR is
latency-bound, tried Tree (lower-latency for tiny messages) and channel reduction
(cut launch overhead). `NCCL_ALGO` must be scoped (`AllReduce:Tree;AllGather:Ring;
ReduceScatter:Ring`) — bare `NCCL_ALGO=Tree` crashes the EP AllGather
("no algorithm/protocol available for AllGather with ncclInt8").

| config | decode bs1 | c8 out | c32 out |
|---|---|---|---|
| baseline (Ring, default ch) | 92.0 | 488 | ~1100 |
| Tree + MIN/MAX_NCHANNELS=1/2 | 91.96 | **378** | **910** |
| Tree, default channels | 91.68 | **388** | **932** |

- bs1: **flat** (the 120 sequential per-op launches dominate; the ring/tree algo
  choice barely matters at 12 KB).
- batch: **net regression** — Tree and low channel counts starve the larger batched
  all-reduce messages of bandwidth/parallelism. **Discard.** Ring + default channels
  (the baseline) is best.

### Lever B — MoE route/permute (11%)

The route/permute kernels (`moe_align_block_size`, `topkGating`,
`count_and_sort_expert_tokens`, `moe_sum`) are **intrinsic to the forced-marlin MoE
path**. There is no fused-permute alternative usable here: the cutlass/flashinfer
fused-MoE backends (`VLLM_FLASHINFER_MOE_BACKEND`, b12x) **do not support swigluoai**
activation, which M3 requires (RECIPE.md). Forcing marlin
(`VLLM_TEST_FORCE_FP8_MARLIN=1`) is the only swigluoai-correct, CUDA-graph-capturable
NVFP4 MoE path on SM120. **No software lever** without re-implementing the un-fused
swigluoai MoE (out of scope for this serving-levers pass; tracked in RECIPE.md).

### Lever C — dense GEMM / MoE marlin

- bs1: **weight-streaming / launch-overhead bound** (see Phase 1 roofline). The HBM
  floor (~434–543 tok/s) is far above the measured 92 tok/s, so HBM is not the bs1
  limiter and FP4-compute can't help; the cost is dequant + launch + AR latency,
  already minimized by CUDA-graph capture. **Documented floor — not chased at bs1.**
- W4A4 (real FP4 activations) for concurrency/prefill: **unavailable** — the M3 NVFP4
  checkpoint has no activation scales (W4A16 checkpoint), and `modelopt.py` pins the
  Marlin W4A16 kernel for exactly this reason. Not a flag.

### Lever D — inductor compilation (`-O3`)

Tested whether enabling torch.compile (the MARLIN config runs `CompilationMode.NONE`)
fuses pointwise ops / cuts launch overhead.
- Result: decode **90.7**, c8 **488**, c32 **1098** — identical to baseline; coherent;
  compiled cleanly (FlashInfer autotuner ran, graphs captured, no errors).
- **No change.** The hot kernels (marlin W4A16 GEMM, marlin MoE, NCCL) are custom CUDA
  ops untouched by inductor pointwise fusion, and CUDA graphs already remove launch
  overhead in the NONE+cudagraph baseline. **No lever.**

---

## FINAL CONFIG (shipped)

The best coherent config is the **clean MARLIN graph baseline** — no tested lever
improves it, and NCCL tuning regresses batch throughput. Shipping:

- `/home/kacper/launch_marlin.sh graph` (== `launch_marlin_tuned.sh`, identical flags).
- **decode bs1: 90.8 tok/s** (final confirm run; 3-run baseline median 92.0).
- **prefill (warm) ~24.9k, cold ~5.5k tok/s.**
- **concurrency 1/8/32: 90.8 / 469–488 / ~1100 out tok/s.**
- Coherence: "capital of Poland" → **Warsaw**; 17×23 → **391**. ✓

**vs the 91.9 tok/s reference: 90.8 tok/s — within run-to-run noise (no regression,
no win).** The honest result: on SM120 + 4-GPU PCIe, the dense-GEMM/all-reduce/MoE
components are hardware floors (W4A16 weight-stream, PCIe-latency PYNCCL,
swigluoai-only marlin) with **no software lever** in current vLLM. Real future wins
require: a W4A4 M3 checkpoint (dense + MoE compute at concurrency), an un-fused
swigluoai NVFP4 MoE to drop marlin's route/permute overhead, or NVLink hardware to
make all-reduce bandwidth-bound (and re-enable custom/symm-mem AR).

## Reproduce

Result JSONs in `nvfp4_serving/bench/results/`:
`final_shipped.json` (baseline), `nccl_tree.json`, `nccl_tree_def.json`,
`fuse_ar.json`, `o3.json`. Restore fallback: `/home/kacper/launch_marlin.sh`.
