# THROUGHPUT_MARATHON.md — MiniMax-M3-NVFP4 on 4x RTX PRO 6000 (SM120, TP4, PHB)

Two decisive questions, every claim a measured number on THIS box (image
`vllm/vllm-openai:minimax-m3`, TP4, block-128, max-len 65536, bf16 KV, marlin
graph @ gpu-mem 0.977):

- **(A)** Is the all-reduce one-shot custom kernel truly dead, or does it beat
  PYNCCL in-graph?
- **(B)** Is there ANY custom-kernel marathon target in the throughput/concurrency
  (compute-bound) regime, or is stock optimal across ALL regimes?

## TL;DR verdict

- **(A) Custom one-shot all-reduce is DEAD for production.** In an isolated
  in-graph A/B it wins the bs1 12 KB collective by **1.17x** (14.4 vs 16.8 µs,
  reproducible, numerically correct). But that win is too small to register
  end-to-end (decode bs1 **92.9 vs 95.5 tok/s** — noise), and forcing it in vLLM
  is a **catastrophic net loss** at every other size: it explodes to 174–4469 µs
  for M≥8 messages, destroying prefill (TTFT **1152 ms vs 20 ms**, 56x worse) and
  batched throughput (c32 **220 vs 1108 tok/s**). vLLM applies one custom-AR to
  ALL message sizes once enabled, so the box-wide effect is a regression. **Keep
  PYNCCL.**

- **(B) A real compute-bound win EXISTS in the kernel, but is unreachable on this
  checkpoint.** The dense GEMM and MoE GEMM both **flip from latency/bandwidth-
  bound (bs1) to compute-bound at M≳256**, and the genuine W4A4 FP4 GEMM beats the
  bf16 (marlin-W4A16) compute path by **1.34x @ M=512, 2.28x @ M=2048, 3.29x @
  M=8192** — exactly the prefill/high-concurrency regime. **But** the M3 NVFP4
  checkpoint is **W4A16 with no activation scales**, and `modelopt.py` pins the
  Marlin W4A16 kernel because cutlass W4A4 "would silently try to quantize
  activations (we have no input_scale)." The win is a **checkpoint property, not a
  kernel we can ship** here. On the *current* M3-NVFP4 checkpoint, **stock marlin
  is optimal across all regimes (bs1 AND throughput).** The marathon target is
  real but gated on a W4A4 checkpoint — see "If a target exists" below.

Marlin restored to production (`/home/kacper/launch_marlin.sh graph`), PYNCCL-only,
Warsaw + 17x23=391 verified.

---

## TASK A — all-reduce one-shot KILL TEST (in-graph A/B + end-to-end)

### Setup / how the gates were forced
- This build (`v0.1.dev17492`) has **no `VLLM_CUSTOM_ALLREDUCE_ALGO` env** (it
  exists only on newer upstream). The 1-stage vs 2-stage choice is internal to the
  C++ kernel, keyed on the `fully_connected` flag passed to `init_custom_ar` and
  the message size: for `fully_connected && world_size<=4 && bytes<512 KB` the
  kernel runs **`cross_device_reduce_1stage` (one-shot)** — which is exactly the
  12 KB bs1 decode collective. (If `fully_connected=False` the dispatch launches
  *no* kernel, so one-shot can only be reached by forcing full connectivity.)
- **In-graph microbench** (`bench/allreduce_killtest*.py`, torchrun ws=4):
  directly instantiated `CustomAllreduce` with `fully_connected=True`, vs
  `PyNcclCommunicator`. CUDA-event timed INSIDE a captured graph (20 ops/graph,
  4000 ops total), eager-loop control to confirm the collective was truly
  captured (graph≈eager → genuine).
- **End-to-end**: patched `platforms/cuda.py:is_fully_connected` to return True
  under `VLLM_M3_FORCE_FULLY_CONNECTED=1` (opt-in), `VLLM_SKIP_P2P_CHECK=1`. Live
  log confirms `Using ['CUSTOM', 'PYNCCL'] all-reduce backends for group 'tp:0'`
  and `Registering 12240 cuda graph addresses` (custom AR captured into the 102
  decode graphs). Patched launch: `/home/kacper/launch_marlin_customar.sh`.

### A1 — in-graph microbench (the real arbiter), per-op µs, ws=4 PHB

| message (M tok × 6144 bf16) | bytes | NCCL (PYNCCL) | custom one-shot | winner |
|---|---|---|---|---|
| **M=1 (bs1 decode)** | 12 KB | **16.8 µs** | **14.4 µs** | **custom (1.17x)** |
| M=8 | 96 KB | 24.7 µs | 173.9 µs | nccl (7.0x) |
| M=32 | 384 KB | 80.3 µs | 1029.2 µs | nccl (12.8x) |
| M=64 | 768 KB | 73.2 µs | 955.4 µs | nccl (13.0x) |
| M=256 | 3 MB | 184.0 µs | 4469.1 µs | nccl (24.3x) |

- custom rel-err vs NCCL = 0.004–0.007 (bf16 reduction-order noise → correct).
- M=1 win is **reproducible**: 5 trials → median **1.17x** (custom 14.4 µs, NCCL
  16.8 µs; range 1.15–1.19x). So the prior "expect skip / 9.2 µs/leg" prediction
  was too pessimistic: custom one-shot does NOT hit the kill criterion (it beats
  NCCL, not ≥ NCCL) **at 12 KB only**.
- For M≥8 the one-shot all-gathers the full tensor to every rank and reduces
  locally; on PHB (PCIe, no NVLink) that read-bandwidth path is catastrophic.
  NCCL's RING_LL scales; one-shot does not.

### A2 — end-to-end A/B (forced custom AR vs PYNCCL, same client config)

| metric | PYNCCL (prod) | FORCED custom one-shot | delta |
|---|---|---|---|
| decode bs1 (tok/s) | **95.5** | 92.9 | −2.7% (noise; TPOT 10.47 vs 10.77 ms) |
| prefill (tok/s) | **25 293** | 447 | **−56x** |
| prefill TTFT (ms) | **20.4** | 1152 | **+56x** |
| concurrency 8 (out tok/s) | **504** | 213 | −58% |
| concurrency 32 (out tok/s) | **1108** | 220 | −80% |
| concurrency 64 (out tok/s) | **1563** | 327 | −79% |

- Coherence under custom AR held at bs1 (Warsaw, 391) — it is *correct*, just slow.
- The end-to-end collapse is the M≥8 microbench explosion manifesting: every
  prefill/batch all-reduce carries many tokens (M≫1) and hits the slow one-shot
  path, because `should_custom_ar` only gates on `bytes < max_size (8 MB)`, not on
  the small-message regime where one-shot is the right tool.

### A — VERDICT: **DEAD.**
The 1.17x bs1 in-graph win is real but (i) invisible end-to-end (∆TPOT within
noise — the 2.4 µs/op × 120 ops = 0.29 ms/token is lost in the 10.5 ms token) and
(ii) inseparable from a 56x prefill / 80% batch regression, because vLLM cannot
size-gate custom-AR to only the 12 KB decode op. There is **no custom-AR kernel
opportunity** that nets positive on this PHB box. PYNCCL stays.

> A genuine win would require a *size-gated* dispatch (one-shot only for
> bytes < ~64 KB, PYNCCL otherwise) — a vLLM dispatch-policy change, not a new
> kernel — and even then it buys ≤0.3 ms/token (< noise). Not worth a marathon.

---

## TASK B — THROUGHPUT / CONCURRENCY characterization

### B4 — throughput scoreboard (marlin baseline, `bench/results/throughput_scoreboard.json`)

| concurrency | out tok/s | per-req tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|
| 1 | 89.2 | 97.6 | 91 | 10.2 |
| 8 | 504.0 | 65.7 | 52 | 15.2 |
| 32 | 1107.6 | 35.7 | 81 | 28.0 |
| 64 | **1563.1** | 32.0 | 110 | 31.2 |

decode bs1 (sequential) = **95.5 tok/s**, prefill (warm) = **25.3k tok/s**.
Throughput scales **17.5x** from bs1 to c64 and is still climbing at c64 (the box
is not saturated) — so the high-concurrency regime is where compute matters.

### B1/B3 — per-component roofline: which components FLIP to compute-bound?

Isolated kernel roofline (`bench/kernel_roofline.py`, single GPU = TP4 per-rank
shapes). bf16 GEMM is the faithful compute-side proxy for marlin W4A16 (marlin
dequantizes NVFP4→bf16 then runs the same bf16 tensor-core GEMM). "ms/floor" =
kernel ms ÷ NVFP4 weight-load time at HBM peak (1.79 TB/s); near 1 ⇒ weight-
streaming/bandwidth-bound, ≫1 with rising TFLOP/s ⇒ compute-bound.

**Dense GEMM (6144×6144 per rank), achieved TFLOP/s and bound type:**

| M (tokens) | ms | TFLOP/s | GB/s | bound type |
|---|---|---|---|---|
| 1 (bs1 decode) | 0.0187 | 4.0 | 4040 | **latency/launch-bound** |
| 8 | 0.0227 | 26.6 | 3332 | latency-bound |
| 32 | 0.0288 | 83.8 | 2648 | transition |
| 64 | 0.0275 | 175.6 | 2801 | transition |
| **256** | 0.0754 | **256.2** | 1084 | **COMPUTE-bound** |
| 512 | 0.1339 | 288.6 | 658 | compute-bound |
| 2048 (prefill) | 0.4642 | 333.1 | 271 | compute-bound |
| 8192 (prefill) | 1.5889 | **389.2** | 174 | **compute-bound (≈roof)** |

**MoE GEMM (gate_up+down, H=6144 I=768, bf16 proxy):** same shape of flip —
M=1: 1.2 TFLOP/s (ms/floor 5.3, launch-bound) → M=256: 141 → M=2048: 249 →
M=8192: **320 TFLOP/s** (compute-bound).

**Component × batch bound-type summary:**

| component | bs1 (M=1) | concurrency 8–64 | prefill (M=2k–8k) |
|---|---|---|---|
| dense GEMM (proj/linear) | latency/launch-bound (4 TF) | **flips ~M=256** | **compute-bound** (390 TF, ≈bf16 roof) |
| MoE GEMM (gate_up/down) | launch-bound (1.2 TF) | **flips ~M=256** | **compute-bound** (320 TF) |
| all-reduce (TP) | **latency-bound** (PCIe handshake, 12 KB) | latency→partly BW-bound | latency/BW-bound — never compute-bound |
| MoE route/permute | launch/index-bound | launch-bound | grows with batch, memory-bound |
| attention (MSA decode) | latency/BW-bound | BW-bound | BW-bound (KV stream) |

**Which components flip to compute-bound at batch:** the **dense GEMM** and the
**MoE GEMM** — and *only* those. All-reduce, route/permute, and attention never
become compute-bound (they are PCIe-latency / index / KV-bandwidth bound at all
batch sizes). So the only place a custom **FP4-compute** kernel could win is the
two GEMMs, in the M≳256 (prefill / high-concurrency) regime.

### B2 — cutlass W4A4 NVFP4 GEMM vs bf16 (marlin) at batch: does the gap FLIP?

Isolated **kernel-only** time (no Python permute/quantize overhead), single-group
NVFP4 grouped GEMM (`group_gemm_nvfp4_nt_groupwise`) vs equal-FLOP bf16 GEMM:

| M (tokens) | FP4 ms | bf16 ms | FP4 TFLOP/s | bf16 TFLOP/s | **FP4 speedup** |
|---|---|---|---|---|---|
| 1 | 0.0285 | 0.0106 | 0.7 | 1.8 | 0.37x (FP4 loses) |
| 8 | 0.0286 | 0.0124 | 5.3 | 12.1 | 0.44x |
| 64 | 0.0286 | 0.0144 | 42.3 | 83.6 | 0.51x |
| 256 | 0.0289 | 0.0268 | 167 | 181 | 0.93x (crossover) |
| **512** | 0.0306 | 0.0411 | 316 | 235 | **1.34x (FP4 wins)** |
| **2048** | 0.0585 | 0.1335 | 661 | 290 | **2.28x** |
| **8192** | 0.1366 | 0.4500 | **1132** | 344 | **3.29x** |

**YES — the gap flips at M≈512.** At bs1 the FP4 kernel *loses* 2.7x (fixed
kernel overhead dominates ~1-token work); but in the compute-bound regime FP4
tensor cores deliver **up to 3.3x** over bf16, hitting **1.13 PFLOP/s** at M=8192.
This is the single most direct confirmation of the throughput thesis: **FP4
compute genuinely wins where the GEMM is compute-bound.** It contradicts the bs1
picture (where FP4 lost 23x end-to-end in eager) precisely because bs1 isn't the
FP4 regime — prefill/concurrency is.

### B3 — is the dense GEMM compute-bound enough at batch for W4A4 to help? + blocker

Yes: at M≥256 the dense GEMM hits 256–390 TFLOP/s (≈bf16 tensor roof) while DRAM
falls to ~270 GB/s (15% of HBM) — it is **compute-bound**, so halving the tensor
work (W4A4) would help directly (the B2 numbers show 1.3–3.3x). **BLOCKER
(confirmed in `m3_patch/modelopt.py`):** the M3 dense linears are **W4A16_NVFP4
checkpoints with no `input_scale`**; `ModelOptNvFp4W4A16LinearMethod.__init__`
**pins `MarlinNvFp4LinearKernel`** specifically because the cutlass W4A4 kernel
"would silently try to quantize activations (we have no input_scale)." Same for
MoE (no `w13/w2_input_scale` with real per-tensor values for the routed experts).
**W4A4 is a checkpoint property, not a flag** — unreachable on this checkpoint.

### B2-serving cross-check (from `cutlass_fp4_moe/RESULTS.md`, prior pass)
The *full serving* cutlass-FP4 MoE path lost 23x at bs1 and 3.3x at prefill — but
that was **eager-only** (graph capture OOMs on the 102-graph footprint), so the
loss was framework overhead (no graph replay), not FP4 math. The B2 isolated
kernel numbers above strip that overhead and show the **math** flips to a win at
batch. The two are consistent: FP4's problem on this box is graph-capturability +
the missing W4A4 checkpoint, not the GEMM.

---

## HONEST VERDICT — is there a custom-kernel marathon target?

**On the current M3-NVFP4 (W4A16) checkpoint: NO. Stock marlin is optimal across
ALL regimes (bs1 and throughput).** Every component that flips to compute-bound at
batch (dense GEMM, MoE GEMM) is blocked from the FP4-compute win by the missing
activation scales; everything that *isn't* blocked (all-reduce, route/permute,
attention) never becomes compute-bound on PHB, so a custom kernel can't beat the
bandwidth/latency floor there. The one micro-win found (1.17x bs1 custom AR) is
both invisible end-to-end and inseparable from a catastrophic prefill/batch
regression. **A measured "no marathon" on this checkpoint — with the numbers.**

**The target is real but gated on hardware/checkpoint, not on a kernel we lack:**

| would-be target | regime | measured ceiling | what unlocks it |
|---|---|---|---|
| **W4A4 dense + MoE FP4 GEMM** | prefill / c≥32 | **1.3–3.3x** GEMM speedup (B2) | a **W4A4 M3 checkpoint** with per-tensor activation scales (+ a low-footprint graph-capturable fused FP4 MoE to avoid the 102-graph OOM) |
| size-gated one-shot AR | bs1 decode | ≤0.3 ms/token (<noise) | a vLLM dispatch-policy change (not a kernel); not worth it |
| custom AR (any) | prefill/batch | negative on PHB | NVLink hardware (re-enables custom/symm-mem AR bandwidth-bound) |

### If a target exists — the FIRST kernel to build
Given a W4A4 M3 checkpoint (the prerequisite), the first kernel is a **single
fused, graph-capturable W4A4 NVFP4 swigluoai MoE kernel** with a *small static
footprint*: one launch per projection over all active experts (no per-expert
Python loop, no per-call activation-quantize loop), capacity sized per captured
batch so it does not blow the 102-graph capture memory (the exact wall that forced
the existing cutlass path to eager — see `cutlass_fp4_moe/RESULTS.md`). The
batched_graphsafe path already proves the **math** (bit-exact vs the per-expert
loop) and the B2 numbers prove the **speed** (up to 3.3x at prefill) — what is
missing is (a) the W4A4 checkpoint and (b) a capture-friendly memory layout. Build
order: W4A4 checkpoint → low-footprint fused FP4 MoE kernel (graph-capturable) →
W4A4 dense linears. Until the checkpoint exists, **ship marlin**.

---

## Production state (restored)
- `/home/kacper/launch_marlin.sh graph` — marlin W4A16, PYNCCL-only all-reduce.
- Coherence: "capital of Poland" → **Warsaw**, 17×23 → **391**. ✓
- Custom-AR experiment kept (opt-in, off by default): patched
  `m3_patch_customar/cuda.py` + `launch_marlin_customar.sh` (needs
  `VLLM_M3_FORCE_FULLY_CONNECTED=1`); **do not ship** (Task-A verdict).

## Reproduce
- A microbench: `bench/allreduce_killtest_v2.py`, `bench/allreduce_m1_repeat.py`
  (torchrun ws=4) → `bench/results/allreduce_killtest_v2.json`,
  `allreduce_m1_repeat.json`.
- A end-to-end: `launch_marlin_customar.sh` then `bench/bench_client.py` →
  `bench/results/customar_e2e.json` vs `throughput_scoreboard.json`.
- B roofline: `bench/kernel_roofline.py` → `bench/results/kernel_roofline.json`.
- B scoreboard: `bench/bench_client.py --concurrency 1,8,32,64` →
  `bench/results/throughput_scoreboard.json`.
