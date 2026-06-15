# OPTIMIZATION_LOG — SM120 MiniMax-M3 bs1 decode (Phase 2)

Target: total ≤ **5.77 µs** (partial ≤ 3.98, merge ≤ 1.80) to match vLLM Triton.
Start: **8.11 µs** (partial 5.17 + merge 2.93), Triton 5.71 (partial 3.94 + merge 1.76).
Correctness gate every change: rms < 1e-2 vs golden `forward_sparse_paged` across the
46-case matrix (split_chunks=1 must stay bit-exact). Bench = torch profiler CUDA
self-time median over 200 launches, container `msa-ncu`, GPU0 exclusive.

The Phase-1 SoL diagnosis (`DECODE_DIAGNOSIS.md`) is the basis for every lever here:
**both kernels are LATENCY-bound (long-scoreboard L1TEX global-load stall, ~65 % of
stall cycles), NOT memory-BW-bound — ours and Triton both sit at ~30 % DRAM (478 GB/s
of ~1.59 TB/s) on the partial and ~5 % on the merge.** The 1.42× gap is geometry +
per-block load efficiency, not bandwidth.

---

## Levers tried, each measured (latency + the SoL metric moved + correctness)

| # | change | partial | merge | total | vs 8.11 | correctness | verdict |
|---|---|---:|---:|---:|---:|---|---|
| 0 | **baseline** sub64 fp8-PV, 128-blk partial / 32-chunk merge | 5.17 | 2.93 | **8.11** | — | ALL OK | start |
| 1 | `MERGE_HD_SPLIT=1` (drop merge z-split → 64 blk like Triton) | 5.17 | **3.11** | 8.28 | +0.17 | OK | **WORSE** — fewer blocks = fewer warps to hide the 32-chunk load chain; the z-split=2 (128 blk) is *better* latency hiding here |
| 2 | `MERGE_HD_SPLIT=4` | 5.17 | 3.02 | 8.19 | +0.08 | OK | worse |
| 3 | **merge v2** (multi-head/block, bf16x2 vectorized O loads, per-thread gmax) | 5.16 | **4.25** | 9.40 | +1.29 | OK | **WORSE** — packing 4 heads/block halved per-warp coalescing width and the per-thread 32-chunk `Mc` recompute (×4 heads, no smem share) cost more than the extra in-flight warps bought. Reverted, default OFF. |
| 4 | `PV_BF16=1` (native bf16 HMMA PV, wider bf16x2 LDS, no fp8 conv) | **5.40** | 2.94 | 8.34 | +0.23 | OK (more accurate, rms 1.3e-3) | NEUTRAL/-3% — partial is NOT fp8-conversion-bound; extra MMA issue + regs cost > the saved F2FP. Confirms prior result. |
| 5 | **page-128 4-warp partial** (W4=1, full 128-page/block → 64 blk, 16 chunks) | **6.53** | **2.08** | 8.61 | +0.50 | OK | merge is the prize (**2.08, near Triton 1.76**) but the full-128-page partial loses 1.36 µs: 1 block does 2× the QK+PV of a 64-page block with single-buffered 64 KB V (can't double-buffer at 99 KB optin) and a long exposed V-transpose-gather PV chain (325 conflicting LDS). |
| 6 | chunk sweep on sub64 (CH=16/20/24/28 → fewer/bigger blocks) | 7.6–7.7 | 2.09 | 9.7 | +1.6 | OK | **WORSE** — each block then does ≥2 sub-tiles *serially* on single-buffered smem at ~1 warp/scheduler, exposing the loop latency. CH=32 (1 sub-tile/block, no loop) is the partial sweet spot. |

**Best end-to-end remains the baseline: 8.11 µs (sub64 fp8-PV, 32-chunk merge).**

---

## Why we did NOT reach Triton's level — the proven structural wall

The diagnosis pinned the limiter; the Phase-2 sweep then *measured* that the two
fast operating points are mutually exclusive on this design:

1. **Fast partial ⇔ 1 sub-tile per block (CH=32, 128 blocks).** sub64 is fast (5.17)
   *only* because each block computes exactly one 64-key sub-tile — no serial inner
   loop, so the single-tile load latency is the whole cost. Doing ≥2 sub-tiles/block
   (CH<32, lever 6) or a full 128-page/block (lever 5) adds a serial, latency-exposed
   loop on single-buffered smem (V can't double-buffer: 2×64 KB > 99 KB optin), pushing
   the partial to 6.5–7.7 µs.
2. **Fast merge ⇔ 16 chunks (page-128 split-K).** The merge is latency-bound on a
   serial 32-chunk O-load chain at ~1 warp/scheduler (achieved occ 4.3 %). 16 chunks
   → 2.08 µs (lever 5), 32 chunks → 2.93 µs. But 16 chunks *requires* the full-128-page
   partial, which is slow (point 1).

So **page-64 split-K** gives the fast partial (5.17) but forces **32 chunks → 2.93 merge**;
**page-128 split-K** gives the fast merge (2.08) but forces the **slow 6.53 partial**.
The two optima can't be held simultaneously without breaking the split-K granularity ↔
chunk-count coupling. Reconciling them needs a cross-block online-LSE merge (2 sub-tile
blocks writing one chunk slot) — a different kernel contract, not a tuning knob — or a
Triton-style cp.async→ldmatrix load path that makes the full-128-page block as cheap as
two 64-page blocks. **That load-path rewrite is the only remaining lever and is the
real edge Triton has** (SASS: Triton 67 cp.async / 48 ldmatrix / 7 LDG per block vs ours
11 cp.async / 0 ldmatrix / 153 LDG + 325 conflicting LDS).

## The exact remaining limiter, with the SoL number that caps it

- **Partial — exposed global/shared-load latency from the load path, not bandwidth.**
  DRAM 30 % (479 GB/s of ~1.59 TB/s), SM 9 %, achieved occ 8.1 %, **long-scoreboard
  L1TEX stall = 65.6 % of stall cycles**, 153 plain LDG + 325 LDS (5.4-way bank
  conflict) per block vs Triton's 67 cp.async + 48 ldmatrix + 7 LDG. We are 1588 GB/s ×
  0.30 = nowhere near the BW ceiling; the cap is **load-issue/latency at ~1 warp/sched**,
  removable only by the cp.async→ldmatrix load-path rewrite (high risk, multi-hour,
  smem-cap-constrained — V double-buffer doesn't fit in 99 KB optin).
- **Merge — serial 32-chunk O-load latency.** DRAM 4.9 % (77 GB/s), achieved occ 4.3 %,
  long-scoreboard L1TEX 65.6 %. Floored at ~2.9 µs for 32 chunks; drops to ~2.08 µs at
  16 chunks (proven, lever 5) — but 16 chunks is unreachable while the partial stays on
  page-64 split-K. The merge is already near-optimal *for its chunk count*; its only win
  is downstream of the partial's geometry.

## Verdict

**We did NOT reach Triton's 5.77 µs. Best stays 8.11 µs (1.42×), unchanged from the
incoming baseline.** Phase 2 confirmed — by direct measurement of every tuning lever —
that the gap is the **structural page-64↔32-chunk coupling plus the per-block load path
(exposed LDG/conflicting LDS vs Triton's cp.async/ldmatrix)**, exactly as the Phase-1
SoL diagnosis predicted. No tuning knob closes it; only a load-path rewrite (cp.async
staging + ldmatrix-fed MMA on a full 128-page block, matching Triton's 64-block /
16-chunk geometry) can, and that is the recommended next step. The kernel remains
correct (46/46 cases, split_chunks=1 bit-exact) at the shipped config.

Code changes this session: added `sm120_fmha_decode_merge_bf16_v2` (multi-head merge,
**default OFF** — measured slower) behind `MERGE_V2`; baseline dispatch unchanged.
