# SM120 DECODE-specialized block-sparse paged attention — results

Box: RTX PRO 6000 Blackwell Server (SM120, **188 SMs**, 1.7 TB/s), image
`vllm/vllm-openai:minimax-m3` (torch 2.11.0+cu130, nvcc 13.0). Run on
`CUDA_VISIBLE_DEVICES=0` in a separate `msa-bench` container (small tensors),
alongside the live `minimax-m3-nvfp4` marlin prod container — prod untouched.
Date 2026-06-14. M3 shapes: 64 q-heads, 4 kv-heads (GQA group = 16), head_dim
128, page/block 64, topk = 16 selected 128-blocks (= 32 page-64 pages), bf16.

Build recipe (validated): `_loader.py::prepare_build_env()` symlink of
cusparse/cusolver headers + `-gencode=arch=compute_120f,code=sm_120f` (required;
plain `sm_120` rejects the block-scaled fp8 MMA).

---

## TL;DR VERDICT

We did **NOT** beat Triton at bs1 decode. Best measured remains **~8.1us**
(page-128 sub64, fp8 PV, 2-kernel split-K+merge, T/ours = 0.71x). Triton ~5.74us.

**This session attacked the two NAMED levers (fuse-the-merge, bf16-native PV).
Both were implemented, proven correct on the 46-case matrix, and BOTH MEASURED AS
LOSSES OR NEUTRAL at bs1 *and across the whole batch curve R=1..16*:**

| lever | result | why |
|---|---|---|
| **L1: fused single-launch merge** (last-block-does-merge, L2-hot partials, no 2nd launch / DRAM round-trip) | **REGRESSED 8.1 -> 15-17us** | At bs1 there are only `Hkv*R` (req,kvh) groups (=4). Fusing makes the *last* chunk-block of each group do the merge **serially** for all 16 GQA heads x 128 dims x 32 chunks, on 4 SMs while 184 idle. That replaces the cheap **128-way-parallel** merge KERNEL (R*Hq*2 blocks) with a **4-way-serial** in-kernel merge. The "true" single-block fused design (chunks=1) is even worse (86us): 4 resident blocks on 188 SMs is hopelessly serial. **Fusion is the wrong tradeoff precisely because bs1 lacks merge parallelism** — and it stays a loss at every batch size tested. |
| **L2: bf16-native PV** (skip bf16->fp8, bf16xbf16 HMMA) | **NEUTRAL / -3%** (partial 5.16 -> 5.3-5.5us) | The partial is **not** fp8-conversion-bound at bs1. Removing the conversion via a native bf16 PV trades 2x k32 QMMA for 4x k16 HMMA (same 64 keys), and the V-transpose smem gather is identical cost. Net wash, slightly worse from the extra MMA issue + register pressure. It IS more accurate (rms vs GOLD 1.3e-3 vs 2.0e-3) and is the right path for a FUTURE fp8/nvfp4 quant-KV — kept behind `PV_BF16`, default OFF. |

The page-128 sub64 path (integration-ready, blockers A+B resolved) stands as the
ship target. Both new code paths are retained behind flags (`FUSED_MERGE`,
`PV_BF16`, both default 0) — correct, measured, documented as non-wins at bs1.

**The corrected, now doubly-confirmed conclusion: the bs1 gap to Triton is
STRUCTURAL, not a missing kernel trick.** The merge's ~3us is a genuine
128-block reduction (not launch latency — profiler self-time excludes that), and
fusing it is counterproductive at bs1's low group count. The partial's 5.2us is
not conversion-bound. Triton ALSO scales well with batch (it is not slower at
occupancy — see the batch table), so the "advantage at batch" thesis did not
hold either: we stay ~0.6-0.7x of Triton from R=1 to R=16.

| metric | old prefill | page-64 v1 | **page-128 sub64 (new best)** | Triton |
|---|---|---|---|---|
| kernel-only us (bs1) | 178.9 | 8.42 | **8.05** | **5.74** |
| ratio vs Triton (T/ours) | 0.032x | 0.68x | **0.71x** | 1.0x |
| native KV layout | page-64 | page-64 | **page-128 (M3 native)** | page-128 |
| integration blocker A (page-64 check) | — | **BLOCKS** | **resolved** | n/a |

Flat across context (kv = 4k/16k/64k). The new path is the right one to ship
(native 128-page, ~5% faster than page-64); the remaining 1.40x is now an
instruction-efficiency + merge-launch problem, not a tiling/occupancy one.

---

## THIS SESSION — fused-merge + bf16-PV experiments (the two named levers)

All measured kernel-only (torch-profiler CUDA self-time, median/100), bs1,
seq_kv=16384, sub64 page-128 best (W4=2, chunks=32). Knobs added:
`FUSED_MERGE` (default 0), `PV_BF16` (default 0). 46-case correctness gate PASSED
for every path (`vs p64 rms=0.000e+00` bit-exact for the fp8 2-kernel ship path;
fused merge also bit-exact; bf16-PV rms vs GOLD 1.3e-3, *more* accurate).

| config | partial us | merge us | total us | T/ours | verdict |
|---|---|---|---|---|---|
| **fp8 PV, 2-kernel (SHIP)** | 5.29 | 2.98 | **8.14** | **0.71x** | best, unchanged from prior |
| bf16 PV, 2-kernel (L2) | 5.47 | 2.96 | 8.36 | 0.69x | NEUTRAL/-3%: partial not conv-bound |
| fp8 PV, **fused merge** (L1) | — | (in-kernel) | **17.5** | 0.33x | **REGRESSED**: 4-way-serial in-kernel merge |
| fp8 PV, fused, chunks=1 single-block | — | — | 88 | 0.07x | catastrophic: 4 resident blocks, fully serial |

### Batch curve (kernel-only profiler us, seq_kv=16384, chunks=32, fp8 2-kernel)
The kernel's hoped-for "advantage at occupancy" does NOT materialize vs Triton —
Triton scales just as well. We are ~0.6-0.7x of Triton from R=1 to R=16.

| R | triton us | triton/req | ours us | ours/req | T/ours |
|---|---|---|---|---|---|
| 1  | 5.74  | 5.74 | 8.18  | 8.18 | 0.70x |
| 2  | 6.33  | 3.16 | 10.29 | 5.14 | 0.61x |
| 4  | 11.47 | 2.87 | 15.76 | 3.94 | 0.73x |
| 8  | 16.77 | 2.10 | 24.90 | 3.11 | 0.67x |
| 16 | 25.10 | 1.57 | 44.09 | 2.76 | 0.57x |

Our per-request work DOES drop with batch (8.18 -> 2.76us/req), confirming the
bs1 work-starvation thesis — but Triton's drops in lockstep (5.74 -> 1.57), so we
never cross it. Triton was successfully driven at batch (decode_query_len=1, R
requests); the comparison is apples-to-apples. **Fused merge is a loss at EVERY
R** (bs1 17.5us, R16 56us) because per-(req,kvh) serial merge work dominates.

### Why fusion loses at bs1 (the decisive measurement)
The merge is NOT launch-bound in the profiler metric (self-time excludes host
launch). Its 2.98us at chunks=32 is genuine GPU reduction work spread over
**128 blocks** (R*Hq*HD_SPLIT = 1*64*2). Fusing it into the partial means the
*last* of the 32 chunk-blocks per (req,kvh) does the WHOLE merge for that group
(16 heads x 128 dims x 32 chunks) alone — 4 such blocks at bs1, on 4 SMs, serial.
That is strictly worse than the 128-way-parallel separate merge kernel. The
separate-kernel 2.98us is near-optimal for this geometry; the only thing fusion
could save is host launch latency, which (a) the profiler metric already excludes
and (b) CUDA-graph capture makes free in serving anyway. **There is no bs1 win in
fusing the merge.**

---

## PAGE-128 PROGRESSION (this session, 8.42us -> 8.05us)

All variants live in `sm120_fmha_decode.cu` alongside the untouched page-64 path;
selected via the `forward_sparse_decode_p128(..., split_chunks, use_4warp)`
binding. Correctness gate (`verify_decode_p128.py`, 46 cases) passed for EVERY
variant: rms vs GOLDEN page-64 kernel <= 2.1e-3, vs dense fp32 <= 3.0e-3, and
within 1.8e-3 of the page-64 decode kernel on identical data.

| step | partial us | merge us | total us | T/ours | note |
|---|---|---|---|---|---|
| page-64 v1 (baseline, 128 blk) | 5.41 | 3.00 | 8.42 | 0.68x | prior best |
| **p128 single-warp** (`_p128`), 64 blk | 8.94 | 2.31 | 11.25 | 0.51x | **REGRESSED**: page-128 halves the block count (16 chunks max vs 32), and bs1 is block-starved. The RESULTS "halve load round-trips" hypothesis is FALSE at bs1 — load/sync was never the limiter; **block count is**. (NSTAGE128=2 double-buffer won't even launch: 134KB > 99KB optin smem.) |
| **p128 4-warp** (`_p128_4w`), 64 blk | 6.61 | 2.10 | 8.71 | 0.66x | limiter-2 fix: all 4 warps compute (QK split by keys, PV split by head-dim, no zero-pad, real k32). 2.25x->1.4x of Triton per-block. K/V committed as separate cp.async groups so QK overlaps the V load. Still only 64 blocks. |
| **p128 sub64 4-warp** (`_p128_sub64`), 128 blk | **5.16** | 2.96 | **8.05** | **0.71x** | **BEST.** Splits split-K at 64-KEY granularity on the native 128-page cache => up to 2*topk=32 chunks => 128 real-work blocks (0.68/SM), matching the page-64 occupancy while keeping the integration-ready 128-page layout. 4 warps split 64 keys (16/warp). |
| p128 sub64 + dh_split=2, 256 blk | 6.00 | 2.95 | 8.95 | 0.65x | **REGRESSED**: doubling blocks by splitting the 128 output dims redoes the full QK + reloads V on the new blocks; that redundant work costs more than the idle-SM occupancy it buys. Confirms 128 *real-work* blocks is the bs1 ceiling. |

### The decisive occupancy measurement (why >128 blocks doesn't help bs1)
Scaling the sub64 partial with the request count R (real extra work, not
redundant) at chunks=32:

| R | total blocks | partial us | partial / request |
|---|---|---|---|
| 1 | 128 | 5.01 | 5.01 |
| 2 | 256 | 6.94 | **3.47** |
| 4 | 512 | 11.87 | **2.97** |

So 256+ blocks of *real* work would drop the per-request partial to ~3us
(Triton territory) — but at bs1 there is only enough real work for ~128 blocks.
Adding blocks via redundant compute (dh_split) does NOT help. **This is the hard
bs1 wall: ~128 useful work-units, period.** It dissolves with batch size.

---

## 1. CORRECTNESS (proven first, before any timing)

`verify_decode.py` compares the new `forward_sparse_decode` against:
- **(a) the GOLDEN `forward_sparse_paged(..., causal=False)`** — the existing,
  rms-0 page-64 kernel run per-request at decode semantics (q_len=1 attends to
  ALL selected KV, i.e. non-causal). Threshold rms < 1e-2 (bf16).
- **(b) a dense fp32 softmax reference** over the gathered selected KV.

Across seq_kv ∈ {4096, 16384, 65536} × selected ∈ {16, 8, 3} × split_chunks ∈
{auto,1,4,8,16} × {R=1, R=2 multi-request} → **ALL OK**:

```
vs GOLD  rms <= 2.2e-03   (max_abs <= 9e-03)
vs DENSE rms <= 2.9e-03
split_chunks=1  ->  rms 0.000e+00 vs GOLD (bit-exact: split-K + LSE-merge is sound)
```

The bit-exact `chunks=1` case proves the flash-decoding split-K partials + the
LSE-merge epilogue reproduce the single-pass result exactly; the small rms at
higher chunk counts is only the fp8-PV / bf16-partial rounding, identical in
character to the golden kernel's own fp8 PV.

---

## 2. THE FIX IMPLEMENTED (flash-decoding geometry)

New file `sm120_fmha_decode.cu`, new entrypoint `forward_sparse_decode`
(the prefill `forward_sparse_paged` path is untouched). Two kernels:

**Partial (split-K) kernel** — `grid = (num_kv_heads, split_chunks, num_requests)`:
- **GQA-group M-tile:** the MMA "M" dimension is the **16 q-heads** sharing a
  kv-head (16 REAL rows, zero dead rows) — not the prefill's 64-query tile with
  1 real row. Kills the 98.4% dead-row MMA waste.
- **Split-K:** the selected pages are partitioned across `split_chunks`
  thread-blocks per (request, kv-head); each computes a partial O + running
  max/denom (LSE) over its page subset.
- bf16 QK (HMMA m16n8k16), fp8-e4m3 P·V (block-scaled QMMA m16n8k32) — same
  numerics as the golden kernel. fp32 accumulation, bf16 partial-out.
- Double-buffered (`cp.async`, NSTAGE=2) K/V so page t+1 loads while page t
  computes; Q load overlapped with the first page load.

**Merge kernel** — `grid = (R, Hq, HD_SPLIT=2)`, flash-decoding LSE merge:
reduces the split-K partials per (request, q-head) into final bf16 O with the
two-pass max/denom rescale. O-partial layout `[R, Hkv, GQA, C, 128]` so a head's
chunk-partials are **contiguous → coalesced** reads; the chunk-reduction loop is
4-way unrolled with independent accumulators to expose memory-level parallelism.

---

## 3. OCCUPANCY — before vs after

| | grid | blocks | blocks/SM (188 SMs) | useful MMA rows |
|---|---|---|---|---|
| **old prefill kernel** | (num_m=1, Hq=64) | 64 | **0.34** | 1/64 = 1.6% |
| **this partial** | (Hkv=4, chunks=32, R=1) | **128** | 0.68 | 16/16 = 100% |
| **this merge** | (R=1, Hq=64, hd=2) | **128** | 0.68 | n/a |

The dead-row waste is fully eliminated (16/16 real rows). Block count rose 64 →
128. Note blocks/SM is still 0.68 at bs1 (only 128 work-units exist for one
request × 4 kv-heads × 32 chunks) — this is inherent to bs1 and is the residual
limiter (see §5); it improves automatically with batch size and is not a
correctness issue.

---

## 4. OUR vs TRITON (kernel-only, torch-profiler CUDA self-time, median/100)

| seq_kv | Triton us | ours us (best, chunks=32) | ratio T/ours |
|---|---|---|---|
| 4096 | 5.74 | **8.41** | 0.68x |
| 16384 | 5.75 | **8.42** | 0.68x |
| 65536 | 5.75 | **8.44** | 0.68x |

Per-kernel split (seq_kv=16384, chunks=32):

| kernel | ours us | Triton us |
|---|---|---|
| partial (`_gqa_sparse_decode`) | 5.42 | 3.97 |
| merge (`_merge_topk_attn_out`) | 3.00 | 1.78 |
| **total** | **8.42** | **5.74** |

### Optimisation progression (honest log)

| step | total us | note |
|---|---|---|
| old prefill kernel | 178.9 | 0.34 blk/SM, 1.6% useful rows |
| v1: GQA-tile + split-K + merge (warp0) | ~15.0 | 11.9x; correct |
| 4-warp key-split | ~16.7 | REGRESSED (2x QMMA on zero-pad + cross-warp merge) → reverted |
| bf16 O-partial | ~15.0 | merge read halved, but merge not yet the limiter |
| coalesced O-partial layout `[R,Hkv,GQA,C,128]` | ~14.2 | merge 5.0→2.8us |
| revert to warp0 compute + 2-stage pipeline + Q-overlap | ~9.9 | partial 11→5.4us |
| 4-way-unrolled MLP merge reduction | **8.42** | merge 4.0→3.0us |

Things tried that did **not** help (measured, reported): 4-warp key-split (2x
QMMA waste); merge head-dim split to 32-thread blocks (under-coalesced, worse);
NSTAGE=1 single-buffer for +occupancy (no change → confirms partial is
latency-bound, not occupancy/smem-capped at bs1).

---

## 5. REMAINING LIMITER (CORRECTED — the real bs1 ceiling)

The original RESULTS named "page-64 vs page-128 load round-trips" as the primary
≈1.4x limiter. **This session disproved that.** Building the page-128 inner tile
(single-warp `_p128`) made things *worse* (11.25us), because page-128 has half
the splittable units => half the blocks, and at bs1 the kernel is **block-count
bound, not load-latency bound** (separating the K/V cp.async groups so QK
overlaps the V load moved the partial by <1%). The 4-warp + 64-key-subtile path
recovered the 128-block occupancy on the native page-128 layout and reached
8.05us — the new best — but the limiters are now precisely:

1. **bs1 has only ~128 real-work units (the hard wall).** 16 selected pages x 16
   GQA heads x 4 kv-heads, split-K to 128 blocks (0.68/SM) is the most *real*
   parallelism available. The R-scaling table above proves the partial would hit
   ~3us at 256–512 blocks, but those blocks must carry real work — at bs1 they
   don't exist, and faking them (dh_split output-dim split) regresses because the
   extra blocks redo QK + reload V. **This dissolves entirely with batch size**
   and is not fixable at bs1 by tiling.

2. **per-block instruction efficiency vs Triton (partial 5.16 vs 3.97us at the
   SAME 64-block-equivalent work).** Two concrete costs our partial pays that
   Triton's `tl.dot` path does not: (a) the PV reads bf16 V from smem and does a
   per-MMA-operand `__bfloat162float` + fp8-pack (the M3 serving cache is bf16,
   blocker 5, so we can't assume an fp8 cache) — this is on the QMMA critical
   path; (b) two `__syncthreads` + smem round-trips per sub-tile for the
   cross-warp softmax max/sum reduction (inherent to the key-split). A
   head-split (4 warps own 4 GQA heads each, zero cross-warp comm) would remove
   both syncs but wastes 12/16 MMA rows — untested; it is the next SASS
   experiment if the partial must drop further.

3. **the merge is a launch-bound second kernel (2.96us at chunks=32, ~37% of
   total).** It has a ~1.8us floor (launch + base O-partial read) that already
   ≈ Triton's whole merge, plus ~1us for the 32-chunk reduction. Triton runs 16
   chunks at bs1, not 32, so its merge is smaller — but our partial *needs* 32
   chunks for the 128-block occupancy (the partial/merge optimum is a hard
   tradeoff; 32 is measured-best). The only real win here is removing the second
   launch (CUDA-graph capture, or fusing partial+merge into one persistent
   kernel keeping partials in L2/regs) — out of scope for a kernel-only microbench
   but the clearest path to ~6us, and free in the cuda-graphed serving path.

**Honest reach estimate (UPDATED after this session's two-lever attack):** the
kernel-only floor on this geometry is ~8us and the two named levers do NOT lower
it:
- The fused single-launch merge was BUILT and MEASURED: it REGRESSES to 15-17us
  at bs1 (and every batch size) because bs1 has only `Hkv*R` merge groups, so
  fusion serializes the merge onto ~4 SMs. The separate 128-block merge kernel is
  already the right structure; its ~3us is genuine reduction work, not launch
  latency (profiler self-time), and is free to overlap under CUDA-graph capture.
- bf16-native PV was BUILT and MEASURED: NEUTRAL (-3%). The partial is not
  fp8-conversion-bound at bs1; it is latency/sync-bound (4-warp cross-key softmax
  reduction = ~2 __syncthreads/sub-tile). Removing the conversion buys nothing.

The remaining ~1.4x at bs1 is therefore **structural at bs1**, not a missing
SASS trick: ~128 real work-units (partial, latency/sync-bound per the 4-warp
key-split), plus a ~3us 128-block merge reduction that fusion makes worse. The
one untested SASS lever that could move the PARTIAL is a **head-split** (4 warps
own 4 GQA heads each, zero cross-warp softmax sync — removes the ~64 syncs) at
the cost of 12/16 dead MMA rows; whether the saved syncs beat the dead-row MMA
waste is unknown and is the only remaining real experiment. Beating Triton at
bs1 otherwise requires matching its `tl.dot` instruction efficiency in the
key-split softmax, which is below the level these flag-gated reworks reached.

---

## 6. INTEGRATION-READINESS (blocker A / B status)

The new page-128 path (`forward_sparse_decode_p128`, `use_4warp=2`) consumes the
**M3-native 128-token page** cache `[num_pages, 128, Hkv, 128]` with `block_ids`
as logical 128-page ids — exactly the serving layout (block-size 128, RECIPE
blocker 6). This **resolves integration blocker A** (the page-64 hard-check):
the kernel no longer requires the cache to be re-tiled to page-64. Blocker B
(fused 5-D `[nblk,2,128,Hkv,d]` cache): the kernel takes separate K/V `[nblk,128,
Hkv,d]` tensors; consuming the fused 5-D tensor is a trivial pointer/stride
change (K = cache[:,0], V = cache[:,1]) but is NOT yet wired in this binding —
left as a 1-line host change since the microbench drives separate K/V. Net: the
page-128 sub64 path is **bench-ready and integration-ready on layout**, pending
only the fused-cache pointer plumbing and a graph-capture merge for the perf win.

---

## Files

- `sm120_fmha_decode.cu` — page-64 decode kernel (untouched) PLUS the new
  page-128 kernels (`_p128` single-warp, `_p128_4w` full-page 4-warp,
  `_p128_sub64` 64-key-subtile 4-warp = the best) + merge. Bindings:
  `forward_sparse_decode(...)` (page-64) and
  `forward_sparse_decode_p128(q, k_cache[*,128,*,*], v_cache, block_table,
  block_ids, softmax_scale, seq_len_k, split_chunks=0, use_4warp=0)`.
  `use_4warp`: 0=single-warp, 1=full-page 4-warp, **2=sub64 4-warp (ship this)**.
  Env knobs (sub64 path): `DH_SPLIT` (default 1; 2 regresses),
  `MERGE_HD_SPLIT` (default 2), `PV_BF16` (default 0 fp8; 1 = native bf16 HMMA
  PV, neutral/-3% at bs1, more accurate, for future quant-KV), `FUSED_MERGE`
  (default 0; 1 = single-launch last-block in-kernel merge, MEASURED LOSS at
  bs1, retained for the record). The sub64 kernel signature gained
  `pv_bf16, O_final, chunk_done` params; `O_final/chunk_done` are non-null only
  when `FUSED_MERGE=1` (then the 2nd merge kernel is skipped).
- `verify_decode.py` — page-64 correctness vs golden + dense (46 cases, ALL OK).
- `verify_decode_p128.py` — page-128 correctness vs golden page-64 kernel + the
  page-64 decode kernel + dense fp32 (46 cases, ALL OK for use_4warp 0/1/2,
  W4 env). Maps a 128-block to page-64 ids [2b,2b+1] on identical data.
- `bench_decode.py` — page-64 kernel-only median vs Triton.
- `bench_decode_p128.py` — page-128 (and page-64 ref) kernel-only median vs
  Triton `minimax_m3_sparse_attn_decode`, per-kernel breakdown. Defaults to the
  sub64 best (W4=2, chunks 32). Honors `PV_BF16`, `FUSED_MERGE`.
- `bench_batch.py` — batch-scaling kernel-only profiler us, ours vs Triton at
  R=1,2,4,8,16 (the occupancy curve). `CH`, `W4`, `SEQ`, `PV_BF16`,
  `FUSED_MERGE` env knobs.
- `bench_graph.py` — CUDA-graph wall-clock variant (note: graph replay overhead
  ~2us swamps sub-10us kernels; the profiler self-time in bench_decode_p128 /
  bench_batch is the cleaner kernel-only metric and the one quoted above).

## How to reproduce
```
# bench container on GPU 0 alongside prod (prod untouched):
sudo docker run -d --name msa-bench --runtime=nvidia --gpus '"device=0"' \
  --network host --ipc host -e CUDA_VISIBLE_DEVICES=0 \
  -v /home/kacper/msa-120/nvfp4_serving:/work -v /home/kacper/models:/models:ro \
  --entrypoint sleep vllm/vllm-openai:minimax-m3 infinity
sudo docker exec msa-bench bash -lc 'cd /work && W4=2 python3 decode_kernel/verify_decode_p128.py'
sudo docker exec msa-bench bash -lc 'cd /work && W4=2 python3 decode_kernel/bench_decode_p128.py'
```
