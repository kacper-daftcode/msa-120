# DECODE_DIAGNOSIS — SM120 MiniMax-M3 bs1 decode (Phase 1: Speed-of-Light + static)

Hardware: **NVIDIA RTX PRO 6000 Blackwell Server Edition** (SM120 / CC 12.0), 188 SMs,
GDDR7. Memory clock 12481 MHz max → ncu's peak-BW reference ≈ **1588 GB/s** (the
sustained boost-clock peak; the marketing spec is ~1.79 TB/s at the rated clock).
All DRAM-% numbers below are **% of that ~1.59 TB/s ncu reference**.

Shapes: bs1 decode, seq_kv=16384, Hq=64 q-heads / Hkv=4 kv-heads, head_dim=128,
GQA=16, 16 selected blocks × 128 keys = 2048 keys. bf16 K/V.

Tools: Nsight Compute 2025.3.1 in a `--cap-add SYS_ADMIN,PERFMON` container,
counters verified non-zero (no ERR_NVGPUCTRPERM). nsys 2025.3.2 for real
(concurrent) kernel durations — **ncu serializes replay so its `Duration` is
inflated ~1.7×; the latency truth is nsys, the SoL %s are ncu**.

Real durations (nsys, median, steady state):
| kernel | OURS | Triton |
|---|---:|---:|
| partial | **5.18 µs** (`..._partial_p128_sub64`) | **3.97 µs** (`_gqa_sparse_decode_kernel`) |
| merge   | **2.98 µs** (`..._merge_bf16`) | **1.79 µs** (`_merge_topk_attn_out_kernel`) |
| **total** | **8.18 µs** | **5.76 µs** (1.42× behind) |

---

## A) ncu Speed-of-Light — the headline ("jakie % mem BW utylizujemy")

| metric | OURS partial | Triton partial | OURS merge | Triton merge |
|---|---:|---:|---:|---:|
| **DRAM throughput % (= mem-BW util)** | **30.1 %** | **30.0 %** | **4.9 %** | **4.2 %** |
| **achieved DRAM GB/s** | **479** | **478** | **77** | **67** |
| Compute (SM) throughput % | 8.8 % | 3.1 % | 2.9 % | 2.3 % |
| L2 hit % | 41.6 % | 11.8 % | 34.4 % | 34.5 % |
| L1/TEX hit % | 28.4 % | 0.6 % | 47.6 % | 31.5 % |
| Achieved occupancy % | 8.1 % | 8.3 % | 4.3 % | 8.3 % |
| Theoretical occupancy % | **16.7 %** (smem-limited) | 8.3 % (smem+reg) | 83.3 % | **100 %** |
| Active warps / scheduler | 1.06 | 1.05 | 1.01 | 1.01 |
| #1 warp stall | **Long scoreboard (L1TEX global-load), 65.6 %** | Long scoreboard (L1TEX), 51.8 % | Long scoreboard (L1TEX), 65.6 % | Short scoreboard (MIO/smem), 37.8 % |
| sectors utilized / 32 (coalescing) | **4.0 / 32** (global loads from DRAM) | — | 18.2 / 32 | — |
| Grid (blocks) | **128** (4,32,1) | **64** (16,4,1) | **128** (1,64,2) | **64** (1,64,1) |
| Waves / SM | 0.34 | 0.34 | 0.03 | 0.03 |

### The answer to "what % of memory bandwidth do we utilize"
- **Partial: ~30 % of peak DRAM BW (479 GB/s of ~1.59 TB/s).**
- **Merge:  ~5 % of peak DRAM BW (77 GB/s).**

### What that proves — NOT memory-bandwidth-bound; LATENCY-bound
The mission framed this as "are we BW-bound" — **the measurement says no.** Both
our partial AND Triton's partial sit at the *same* ~30 % DRAM, ~8 % occupancy,
~1 active warp/scheduler, and the **#1 stall for every kernel is the long-scoreboard
wait on an L1TEX (global-load) dependency** (65.6 % of our partial's stall cycles).
At bs1 the grid is a fraction of a wave (0.34 / 0.03 waves per SM), so there are
**~1 warp per scheduler** to hide global-load latency — the SMs sit idle waiting on
loads. This is a **latency / low-occupancy regime, not a bandwidth ceiling**: we are
nowhere near saturating DRAM (30 %), and we are nowhere near saturating compute (9 %).

**Verdict per kernel:**
- **OURS partial → latency-bound** (long-scoreboard L1TEX 65.6 %, 0.34 waves, 1.06 warps/sched, DRAM 30 %, SM 9 %).
- **OURS merge → latency-bound** (long-scoreboard L1TEX 65.6 %, 0.03 waves, achieved occ 4.3 % vs 83 % theoretical, DRAM 5 %).
- Triton partial/merge are in the **same latency regime** but finish faster — so the
  gap is **per-block work + geometry**, not a different bound (see §C).

---

## B) Static resource picture (ptxas -v + SASS, no GPU)

Built `-gencode=arch=compute_120f,code=sm_120f -O3 -Xptxas -v -lineinfo`. Raw:
`nsys_stats/ptxas_verbose_ours.txt`, SASS `nsys_stats/sass_ours_full.txt`.

| kernel | regs/thread | smem/block | spills | stack | barriers | theoretical-occ LIMITER |
|---|---:|---:|---:|---:|---:|---|
| OURS partial `_sub64` | **126** | **40.45 KB** dyn (+80 B static) | **0 B** | 1024 | 1 (ptxas) / 8 (SASS BAR) | **shared memory** → Block Limit Shared Mem = 2 → 16.7 % theo occ |
| OURS partial `_p128_4w` | 234 | ~57 KB | 0 | — | 1 | regs+smem |
| OURS partial `_p128` (1-warp) | 246 | ~71 KB | 0 | — | 1 | regs+smem |
| OURS merge | **42** | 128 B dyn | **0 B** | 1024 | 1 | none (83 % theo); achieved 4.3 % = workload/imbalance |
| Triton partial | 145 | 73.7 KB dyn | n/a | — | 15 (SASS) | smem (Block Limit Shared Mem = 1) → 8.3 % theo occ |
| Triton merge | 32 | 2.05 KB dyn | n/a | — | 4 (SASS) | none (100 % theo) |

No spills anywhere. Our partial's theoretical occupancy is capped at **16.7 % by the
40 KB dynamic smem** (Block Limit Shared Mem = 2 blocks/SM). Raising occupancy would
need a smaller smem footprint — but note occupancy is *not* the proximate latency
fix here (Triton wins at 8.3 % theo occ).

### SASS hot-loop instruction mix (per block)

(Refined exact opcode counts — `LDG.E` = plain global incl. scalar `.CONSTANT`,
`LDGSTS.E.128` = 128-bit cp.async, `LDSM` = ldmatrix, `LDS`/`LDS.` = manual shared loads.)

| op | OURS sub64 partial | Triton partial | OURS merge | Triton merge |
|---|---:|---:|---:|---:|
| LDG.E (plain global; 4 are scalar `.CONSTANT`) | **59** | **5** | 110 | 5 |
| **LDGSTS.E.128 (cp.async, K/V/Q bulk)** | **11** | **67** | 0 | 0 |
| **LDSM (ldmatrix — MMA fed by HW)** | **0** | **48** | 0 | 0 |
| **LDS (manual shared load — MMA fed by SW)** | **362** | **25** | 38 | 13 |
| STS | 39 | 8 | 1 | 20 |
| HMMA | 32 | 64 | 0 | 0 |
| QMMA (fp8) | 8 | 0 | 0 | 0 |
| MUFU (exp/special) | 32 | 22 | 8 | 4 |
| F2FP (bf16↔fp conv) | 50 | 16 | 1 | 4 |
| **BAR.SYNC** | **8** | 15 | 1 | 4 |
| SHFL | 9 | 18 | 24 | — |

### The single dominant limiter, per kernel

- **OURS partial — the MMA-feed path is software (manual LDS), not hardware (ldmatrix).**
  Our K/V *do* stream via cp.async (11 `LDGSTS.E.128`, 128-bit), so the bulk DRAM→smem
  path is fine. The limiter is the **smem→MMA feed: we issue 362 manual `LDS` per block
  and 0 `ldmatrix`, while Triton issues 48 `LDSM` (ldmatrix) and only 25 LDS.** Those
  362 LDS are the QK `lds_u32` emulation and especially the **PV V-transpose gather**
  (reading V column-wise one element at a time) — ncu: **5.4-way bank conflict, 74 %
  excessive shared-load wavefronts**. With ~1 warp/scheduler the resulting long-scoreboard
  stall is **65.6 % of stall cycles**. **This is the #1 limiter, and the named fix is to
  replace the manual LDS gather with `ldmatrix`/`ldmatrix.trans` (LDSM) on ldmatrix-laid
  smem, as Triton does.**
- **OURS merge — too many blocks doing too little, latency-exposed.**
  Grid 128 = (R=1, Hq=64, **HD_SPLIT=2**); the z-split of 2 doubles the grid vs
  Triton's 64 (no z-split), each block reduces only 64 head-dims over 32 chunks
  with achieved occ 4.3 %. Long-scoreboard L1TEX = 65.6 %. The reduction reads
  **32 chunk-partials** (because our partial split-K granularity is page-64 → 2×topk
  chunks) vs Triton merging **16**. So the merge inherits the partial's 2× chunk count.

---

## C) Why Triton (64 blocks) beats us (128 blocks) — the structural diagnosis

Both kernels are latency-bound at the *same* 30 % DRAM. The 1.42× gap is **geometry +
per-block load efficiency**, NOT bandwidth and NOT a different limiter:

1. **Block count:** Triton partial = **64 blocks** (one 128-key page × 4 kv-heads,
   all 16 GQA heads per block). Ours = **128 blocks** (split-K at **page-64**
   granularity → 2×topk=32 chunks × 4 kv-heads). We launch **2× the blocks**, each
   doing half the keys, so 2× the QK/softmax/partial-output overhead and 2× the
   chunk-partials the merge must reduce. *(Measured: at chunks<32 our partial is
   even worse — 7.7 µs — because fewer/bigger blocks expose load latency more; 128
   blocks is our sweet spot, but it's still 2× Triton's geometry.)*
2. **Per-block MMA-feed path:** Triton feeds the tensor cores with **ldmatrix**
   (67 LDGSTS / **48 LDSM** / 25 LDS / 5 LDG); we feed them with a **software gather
   of 362 manual LDS** (5.4-way bank-conflicted) and **0 LDSM** — our K/V cp.async
   (11 LDGSTS) is fine, the loss is entirely the smem→MMA stage. In ncu single-kernel
   replay our partial and Triton's have *near-identical* elapsed cycles (16420 vs 16362)
   — the real-world 5.18 vs 3.97 gap comes from our 2× block count crossing more
   scheduling waves on the latency-bound path, compounded by the slower per-block LDS feed.
3. **Merge:** ours reduces 32 chunks over 128 (z-split=2) blocks; Triton reduces 16
   chunks over 64 blocks. Halving our chunk count (page-128 split-K) and dropping the
   z-split would match Triton.

**Prior agent's conclusion "block count is the bs1 wall (128 useful work-units)" is
contradicted by this data:** Triton reaches 3.97 µs with **only 64 blocks**, so 64 is
not the wall — Triton's per-block load path (cp.async/ldmatrix, no exposed LDG) is the
edge. The optimization target is therefore: **make the partial's load path async
(cp.async→ldmatrix, kill the 153 exposed LDG + bank-conflicting LDS) and the merge's
geometry match Triton (16 chunks, no z-split).**

Raw artifacts: `ncu_out/ours_partial_details.txt`, `ncu_out/ours_merge_details.txt`,
`ncu_out/triton_details.txt`, `ncu_out/*.ncu-rep`, `nsys_stats/ptxas_verbose_ours.txt`,
`nsys_stats/sass_*.txt`, `nsys_stats/captureA_*.txt`.
