# GEOMETRY_TUNING — SM120 MiniMax-M3 bs1 decode: the geometry/wave/scheduling residual

Phase-3 of the bs1 decode optimization. The MMA-FEED limiter was already fixed in
`LDMATRIX_REWRITE.md` (ldmatrix QK + ldmatrix.trans PV, bank conflicts 339968→0,
LDSM/HMMA byte-identical to Triton — NOT touched here). This phase is purely the
geometry / wave-occupancy / partial→merge / inter-step side.

Incoming best: `forward_sparse_decode_p128(use_4warp=3)` (`_partial_p128_ldsm`)
**6.70 µs** (partial 4.59 + merge 2.08) = **0.85× Triton 5.77** (partial 3.97 + merge 1.79).

Hardware: RTX PRO 6000 Blackwell (SM120), **188 SMs**, ~1.59 TB/s ncu-ref DRAM.
Tools: ncu 2025.3.1 (`--cap-add SYS_ADMIN,PERFMON`), nsys 2025.3.2 (real back-to-back
durations). Shape: bs1, seq_kv=16384, Hq=64/Hkv=4, GQA=16, dim=128, topk=16.

---

## 1. THE CENTRAL PUZZLE — solved: it's a CACHE/REPLAY artifact, not a wave problem

"Faster in ncu isolation (8.03 vs Triton 9.12 µs) but slower live (4.59 vs 3.97)."
Tested all three mission hypotheses by direct measurement:

### (a) Wave/tail under-fill — REJECTED. Both kernels are at the IDENTICAL operating point.
ncu LaunchStats, partial, seq=16384:

| metric | OURS W4=3 ldsm | Triton |
|---|---:|---:|
| grid (blocks) | 64 | 64 |
| smem/block | 78.85 KB | 73.73 KB |
| **occupancy_limit_shared_mem** | **1 block/SM** | **1 block/SM** |
| occupancy_limit_registers | 3 block/SM | 3 block/SM |
| regs/thread | 158 | 145 |
| **waves/SM** | **0.34** | **0.34** |
| **warps/scheduler** (smsp warps_active/cyc) | **0.93** | **0.97** |
| max warps/active-cycle | 8.33 % | 8.33 % |
| **sm__cycles_active (active-SM%)** | **25 %** | **23 %** |
| warps_active/SM | 3.98 | 3.99 |

Triton does NOT resident-occupy more warps/scheduler — it is byte-for-byte the same
under-filled geometry (1 block/SM, 0.93 vs 0.97 warps/sched, ~25% of SMs busy). The
wave/occupancy hypothesis is **wrong**: occupancy is not the differentiator.

### (b) partial→merge skew / block-finish spread — REJECTED.
nsys cuda_gpu_trace, steady state. Kernels run STRICTLY SERIAL in both impls (no
partial/merge overlap). Per-block duration StdDev is tiny (ours 42 ns over 30 launches,
Triton 32 ns). The partial→merge GPU-idle gap is **ours 1216 ns vs Triton 9408 ns** —
**our handoff is 8× TIGHTER**, not looser. Block-finish skew is not the problem.

### (c) inter-step overlap — REJECTED.
No kernel overlaps the next step in either impl (host-launch serialized). Our
merge→next-partial gap (7936 ns) is smaller than Triton's (30208 ns, its heavier Python
launch path). Inter-step overlap is not the problem.

### ROOT CAUSE (won): ncu's default cache-FLUSH penalizes Triton's zero-reuse loads.
The discrepancy is a **profiler artifact**. Re-ran ncu with `--cache-control none`
(warm caches, which is what live steady-state has: the same 16 KV pages are reloaded
every decode step and stay L2-resident):

| metric (partial, warm cache) | OURS W4=3 | Triton |
|---|---:|---:|
| **gpu duration** | **6.91 µs** | **6.14 µs** |
| sm__cycles_elapsed | 12710 | 11280 |
| **DRAM throughput** | **0.16 %** (fully L2-resident) | 7.0 % |
| L1 hit | 17.8 % | 0.56 % |
| #1 stall: long-scoreboard (memory) | **9.3 %** | 33.0 % |
| #1 stall: **wait** (MMA/exp fixed-latency pipe) | **35.3 %** | 23.6 % |

With warm caches the ordering **FLIPS to match live** (ours slower). In ncu's default
mode caches are flushed every replay pass, so Triton — which has near-zero cache reuse
(L1 0.56 %, long-scoreboard 51.8 % cold) — pays full DRAM latency every pass and looks
slow (9.12 µs). Live, both kernels' KV is L2-resident (our DRAM drops to 0.16 %), erasing
that penalty and exposing the true difference: **pure per-block compute**.

### What the live gap actually is: +25 % instruction count, all in softmax (not MMA).
ncu executed-instruction mix, warm, grid total:

| op | OURS W4=3 | Triton | delta |
|---|---:|---:|---:|
| **total instructions** | **436 864** | **349 312** | **+87 552 (+25 %)** |
| tensor (HMMA) | 16 384 | 16 384 | **0 (identical)** |
| ldmatrix (LDSM) | 12 288 | 12 288 | **0 (identical)** |
| shared_ld (LDS, executed) | **30 208** | 6 400 | **+23 808** |
| shared_st (STS) | 3 072 | 2 048 | +1 024 |
| global_ld | 512 | 1 536 | −1 024 |
| local (spills) | **0** | 0 | 0 |

The MMA feed is byte-identical to Triton (LDSM, HMMA both exactly match). The entire
+25 % is the **softmax + cross-warp reduction path** (LDS + the exp2f/fmax/F2FP math):
our 4 warps key-split, each computes the partial softmax for ALL 16 heads over its 32
keys, then combines across warps through `sRed` smem. At ~0.93 warps/scheduler that extra
work cannot be hidden — it shows up as the `wait` stall (35 %) and the extra cycles.

**Verdict on the puzzle: hypothesis (a)/(b)/(c) all rejected. The residual is per-block
instruction overhead (softmax, +25 %), made visible because the GPU is under-filled
(25 % SM-active) so nothing hides it. The "isolation vs live" inversion is a warm-vs-cold
L2 artifact of ncu replay.**

---

## 2. LEVERS — each measured (latency + waves/occupancy + correctness 46/46)

All builds pass the 46-case gate (rms vs golden 1.3e-3–2.9e-3 < 1e-2, split_chunks=1 exact).

### Lever 1 — raise resident occupancy to 2 blocks/SM (W4=4 `_ldsm2`, shared KV buffer)
Collapsed sK+sV into ONE shared page buffer (K loads it, QK consumes it, then V loads
into the same bytes). Smem **78.85 → 44.03 KB ⇒ occupancy_limit_shared_mem = 2 blocks/SM**
(verified by ncu). **Result: NO change** — partial 4.61→4.62 µs, `sm__warps_active`
3.97→3.97, `sm__cycles_active` 24.75 % unchanged.

> **Measured why it's a DEAD END at bs1: we are BLOCK-starved, not smem-starved.** Grid =
> 64 blocks < 188 SMs, so every block already gets its own SM; the scheduler never wants
> to pack 2 blocks onto one SM because there aren't 2 blocks competing for it. Raising the
> per-SM block *capacity* does nothing when the grid doesn't fill the SMs once. The
> "1 warp/scheduler" deficit is fixable only by MORE BLOCKS, not more blocks-per-SM.
> (Kept as W4=4, correct, for the record / future batched use where it WILL help.)

### Lever 2/3 — more blocks: the R-scaling proves SM-fill headroom is FREE, then key-split
`forward_sparse_decode_p128(W4=3)` swept over R (independent requests = proxy for blocks):

| R | blocks | total µs | per-request µs |
|---:|---:|---:|---:|
| 1 | 64  | 6.67 | 6.65 |
| 2 | 128 | **6.84** | **3.42** |
| 4 | 256 | 10.89 | 2.73 |

64→128 blocks costs only **+0.17 µs total** (per-request HALVES) because 124 of 188 SMs
sit idle. **Filling SMs with more blocks is nearly free at bs1.** The non-redundant way to
reach 128 blocks at bs1 (head-split would force MMA M=8, wasting half the tensor core) is
to **KEY-split each 128-page across 2 blocks** (64 keys each, M=16 preserved, half the MMA
+ half the cross-warp-softmax span per block): new kernel `_p128_ksplit` (W4=5), grid
(Hkv, 2·topk, R) = **128 blocks**, merge reduces 2·topk = 32 chunks.

**Result (nsys live, seq=16384):**

| path | partial | merge | total | vs Triton 5.71 |
|---|---:|---:|---:|---|
| W4=3 page-128, 16-chunk (shipped) | 4.58 | **2.08** | **6.66** | 0.86× |
| **W4=5 key-split, 32-chunk** | **3.96** | 3.04 | 7.01 | 0.81× |
| Triton | 3.95 | 1.76 | 5.71 | 1.00× |

> **The partial reaches TRITON PARITY: 3.96 vs 3.95 µs.** Key-split fully closed the
> partial gap (halving per-block work + filling 128 SMs). This DEFINITIVELY proves there
> is **no partial-side wall** — the geometry lever works exactly as the R-scaling predicted.
> BUT the merge rose 2.08 → 3.04 µs because it now reads **2× the chunk-partials** (32 vs
> 16). The merge is long-scoreboard bound on the O-load chain (ncu: 48 % long-scoreboard,
> 0.06 waves) — its latency scales with chunk count. Net 7.01 > 6.66: **the merge erases
> the partial win.** This is the structural page-64↔chunk-count coupling, now measured on
> BOTH halves at BOTH operating points.

### Lever 4 — finer cp.async granularity: scoped OUT (DRAM→smem stage). Not pursued; the
warm-cache data shows DRAM at 0.16 % (fully L2-resident live), so cp.async granularity is
irrelevant to the live latency — it only mattered in ncu's cold replay.

### Lever 5 — faster 32-chunk merge (chunk-parallel, `_merge_bf16_par`, W4=5 default)
Built a merge that splits the 32-chunk reduction across 4 warps (per-thread O-load chain
32→8) + smem LSE combine. **Result: 3.04 µs — NOT better than the flat 32-chunk merge
(2.99–3.05).** Per-chunk our merge is already FASTER than Triton (2.99/32 = 0.093 vs
1.76/16 = 0.110 µs/chunk); the bottleneck is total DRAM bytes/latency of reading 2× the
O-partials, which no thread-mapping changes. Also tried MERGE_HD_SPLIT∈{1,2,4} (best 4 =
2.99) and MERGE_V2 (4.26, worse).

---

## 3. PROGRESSION & VERDICT

| stage | partial | merge | total | vs Triton |
|---|---:|---:|---:|---|
| incoming best W4=3 (ldmatrix + page-128 geometry) | 4.58 | 2.08 | **6.66** | 0.86× |
| W4=4 shared-buffer (2 blk/SM) — occupancy lever | 4.62 | 2.08 | 6.70 | 0.85× |
| W4=5 key-split (partial=Triton-parity) + par-merge | **3.96** | 3.04 | 7.01 | 0.81× |
| **SHIPPED: W4=3** | **4.58** | **2.08** | **6.66** | **0.86×** |
| Triton | 3.95 | 1.76 | 5.71 | 1.00× |

**Honest verdict: the shipped best stays W4=3 at 6.66 µs (0.86× Triton). The geometry
phase did not net a speedup, but it DECISIVELY characterised the wall with measured
numbers:**

1. **The puzzle was a profiler artifact** (ncu cold-flush vs live warm-L2), not a wave/
   tail/overlap problem — all three named hypotheses rejected by direct measurement.
   Occupancy is identical to Triton (both 1 block/SM, 0.93 warps/sched, 25 % SM-active).
2. **Occupancy can't be raised at bs1** — block-starved (64 < 188 SMs), so 2 blocks/SM
   (W4=4, smem 79→44 KB, verified) buys nothing; warps_active stays 3.97.
3. **The partial has NO wall** — key-split (W4=5) hit Triton parity (3.96 vs 3.95 µs) by
   filling 128 SMs, confirming the R-scaling.
4. **The wall is the partial↔merge CHUNK-COUNT COUPLING** (measured on both halves): the
   fast partial needs 32 chunks → merge 3.04; the fast merge needs 16 chunks → partial
   4.58. They can't both be held. Breaking it needs a **cross-block online-LSE reduction**
   so the 2 key-split blocks write ONE 16-chunk slot (a different kernel contract / global
   atomics — high risk, deferred), OR accepting the residual.

### The capping metric, with the number
At bs1 there are only **64 independent work-units** of non-MMA-wasting work (16 pages ×
4 kv-heads), on **188 SMs** ⇒ the grid is **0.34 waves/SM, 25 % SM-active, ~1 warp/
scheduler**. The per-block softmax overhead (+25 % instructions vs Triton, +23 808 LDS,
0 in the MMA feed) is fully exposed because nothing hides it at 1 warp/scheduler. Key-split
to 128 blocks fixes the partial (→ Triton parity) but the merge's 2× O-load latency
(long-scoreboard 48 %, 0.06 waves) gives it all back. **The last 0.95 µs is the partial↔
merge chunk-count coupling at the bs1 64-independent-unit ceiling, not occupancy, not
bandwidth (DRAM 0.16 % live), not the MMA feed (byte-identical to Triton).**

## 4. Deliverables in `sm120_fmha_decode.cu` (all old paths intact)
- `sm120_fmha_decode_partial_p128_ldsm2` — W4=4, shared-KV-buffer (2 blk/SM). Correct, kept.
- `sm120_fmha_decode_partial_p128_ksplit` — W4=5, key-split-K (partial = Triton parity). Correct, kept.
- `sm120_fmha_decode_merge_bf16_par` — chunk-parallel merge (auto-ON for W4=5; `MERGE_PAR=1` elsewhere). Correct, kept.
- Default dispatch and **W4=3 shipped path UNCHANGED**. W4∈{0,1,2} legacy paths untouched.

## 5. Reproduce
```
# bench container (GPU0 exclusive), ncu+nsys host-mounted (see DECODE_DIAGNOSIS.md §setup):
sudo docker exec msa-bench bash -lc 'cd /work && W4=3 python3 decode_kernel/verify_decode_p128.py'  # 46/46
sudo docker exec msa-bench bash -lc 'cd /work && W4=3 CHUNKS128=16 python3 decode_kernel/bench_decode_p128.py'  # 6.66
sudo docker exec msa-bench bash -lc 'cd /work && W4=5 CHUNKS128=16 python3 decode_kernel/bench_decode_p128.py'  # partial 3.96
# warm-cache ncu (mimics live; the puzzle's resolution):
ncu --kernel-name regex:ldsm --cache-control none --metrics dram__throughput...,smsp__warp_issue_stalled_wait... python3 decode_kernel/ncu_driver.py
```
