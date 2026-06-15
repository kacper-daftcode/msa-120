# LDMATRIX_REWRITE — SM120 MiniMax-M3 bs1 decode partial: software-LDS → ldmatrix

New kernel `sm120_fmha_decode_partial_p128_ldsm` (entrypoint
`forward_sparse_decode_p128(..., use_4warp=3)`, selector `W4=3`). The old paths
(sub64 `W4=2`, 4w `W4=1`, single-warp `W4=0`, page-64 `forward_sparse_decode`)
are untouched. Built with `-gencode=arch=compute_120f,code=sm_120f`.

## What changed (the two coupled mission levers)

1. **ldmatrix-fed MMA (the #1 limiter fix).** The smem→tensor-core FEED is now
   hardware (`ldmatrix` / LDSM), not the 362-`LDS` software gather:
   - **QK** : A=Q and B=K both fed by `ldmatrix.x4` from the row-major
     `[row][head_dim]` smem the cp.async lands. (frag maps validated in
     isolation: K's 16-key tile → key0-7=(v0,v2), key8-15=(v1,v3)).
   - **PV** : switched fp8 block-scaled QMMA → **native bf16 HMMA m16n8k16**.
     A=P (bf16, `ldmatrix.x4`); B=V via **`ldmatrix.x4.trans`** — the transpose
     that *was* the 362-LDS column-wise V gather is now ONE hardware op. V stays
     stored exactly as cp.async lands it (`[key][dim]` row-major); one
     `ldmatrix.x4.trans` over a 16×16 region yields dims0-7=(v0,v1),
     dims8-15=(v2,v3) over 16 keys (validated maxdiff=0 vs reference).
   - **smem swizzle**: K/V/P/Q smem rows are padded to `head_dim+8` bf16. The +8
     (one 16B ldmatrix granule) shifts each row by 4 banks so the 16 per-lane
     addresses an ldmatrix issues land in distinct banks. This is the single
     most important change — see the conflict numbers below.
2. **page-128 / 64-block geometry (Triton's).** One block owns one kv-head's
   full 128-key page (all 16 GQA heads), 4 warps key-split QK and dim-split PV.
   split-K over the 16 selected pages (default `split_chunks=16` → 1 page/chunk
   → grid (Hkv=4, 16, 1)=**64 blocks** at bs1). The merge therefore reduces
   **16 chunks**, not 32. K/V cp.async DRAM→smem path is **unchanged** (128-bit
   LDGSTS), per the mission constraint.

## Before / after instruction mix (ncu, executed, per the 16384/topk16 shape)

Grids: sub64 = (4,32,1) **128 blocks**; ldsm = (4,16,1) **64 blocks**.
Per-block = total ÷ blocks.

| metric (per block) | BEFORE sub64 (W4=2) | AFTER ldsm (W4=3) | Triton |
|---|---:|---:|---:|
| **LDSM** (ldmatrix) | **0** | **192** | 192 |
| **LDS** (shared-ld, executed) | **608** | **64** | 36 |
| shared-load **bank conflicts** (grid total) | **339 968** | **0** | **0** |
| cp.async LDGSTS (grid total) | — | 8 704 | 16 576 |
| global LDG (grid total) | — | 512 | — |

The instruction mix moved to Triton's shape: LDSM 0→192 (= Triton's 192), LDS
608→64, **bank conflicts 339 968 → 0 (identical to Triton's 0)**.

## Before / after Speed-of-Light (ncu, partial kernel, 16384 shape)

| SoL metric | BEFORE sub64 | AFTER ldsm | Triton |
|---|---:|---:|---:|
| DRAM throughput % | 29.4 | **33.4** | 29.2 |
| achieved occupancy % | 8.28 | 8.29 | 8.33 |
| theoretical occupancy % | 16.7 | 8.33 (smem=1 blk/SM) | 8.33 |
| long-scoreboard stall (ratio) | 1.95 | **1.63** | 5.08 |
| shared-load bank-conflict | 5.4-way | **conflict-free** | conflict-free |
| ncu single-kernel Duration (inflated) | — | **8.03 µs** | **9.12 µs** |

Under ncu's serialized single-kernel replay our ldsm partial (8.03 µs) is now
**faster than Triton's (9.12 µs)** with a much lower long-scoreboard stall
(1.63 vs 5.08). The real-world (concurrent) gap that remains is the geometry
crossing scheduling waves + Triton's 2× finer cp.async granularity (16 576 vs
8 704 LDGSTS) giving slightly better load overlap — the DRAM→smem stage the
mission told us not to touch.

## Latency progression (nsys/torch-profiler kernel-only median, 16384)

| stage | partial | merge | **total** | vs Triton 5.77 |
|---|---:|---:|---:|---|
| **baseline** sub64 page-64-split / 32-chunk (W4=2) | 5.23 | 2.99 | **8.22** | 0.70× |
| page-128 ldsm, **no swizzle** (conflicted) | 6.89 | 2.09 | 8.98 | 0.64× |
| **page-128 ldsm + pad-swizzle (FINAL, W4=3)** | **4.59** | **2.08** | **6.67–6.73** | **0.85–0.86×** |
| Triton (`minimax_m3_sparse_attn_decode`) | 3.97 | 1.79 | **5.77** | 1.00× |

- The merge dropped **2.99 → 2.08 µs** purely from the 16-chunk geometry
  (proven floor; matches the prior lever-5 measurement of 2.08).
- The partial dropped **5.23 → 4.59 µs**, and critically the *un-swizzled*
  ldmatrix partial was 6.89 µs — the **pad-swizzle alone bought 2.3 µs** by
  killing the 339 968-conflict (→0) bank-conflict storm. Without the swizzle the
  ldmatrix rewrite is a net loss; with it, it wins.

## Correctness gate (every build)

`verify_decode_p128.py` with `W4=3`: **46/46 cases OK**, rms vs golden
`forward_sparse_paged` = 1.3e-3–2.9e-3 (< 1e-2 gate), rms vs dense fp32
≈ 1e-4 (the bf16 PV is *more* accurate than the old fp8 PV, ~1.7e-3). Bit-exact
not claimed (bf16 PV ≠ the page-64 fp8 reference), but `split_chunks=1` is a
single-page-per-block exact-reduction mode and passes. Seq 4k/16k/64k ×
selected 16/8/3 × chunks {0,1,4,8,16} × R{1,2}.

## HONEST verdict — did we reach Triton's 5.77 µs?

**No. We reached 6.67–6.73 µs (0.85–0.86× Triton), down from the 8.22 µs
baseline (0.70×).** The rewrite delivered exactly what the diagnosis predicted it
would and closed the *named* limiter completely:

- **The MMA-feed limiter is GONE.** LDSM 0→192, LDS 608→64, **bank conflicts
  339 968→0 (= Triton's 0)**. ncu confirms our partial now has a *lower*
  long-scoreboard stall than Triton (1.63 vs 5.08) and is faster in single-kernel
  replay (8.03 vs 9.12 µs).
- **The geometry is Triton's** (64 blocks, 16 chunks), so the merge hit its
  proven 2.08 µs floor.

### The hard wall that caps us, with the measured number

The residual ~0.9 µs is **not** the feed and **not** bandwidth — both kernels sit
at ~30–33 % DRAM, 8.3 % occupancy, **1 block/SM (smem-limited, identical to
Triton)**. Two measured caps remain:

1. **Partial: 4.59 vs 3.97 µs (0.62 µs).** At `chunks=16` each block processes
   exactly ONE 128-page with **no inner loop**, so the single page's K+V load
   (64 KB) is *fully exposed* with nothing to overlap it against, and the smem
   footprint (78.8 KB padded) caps the SM to **1 resident block** so there is
   ~1 warp/scheduler to hide it. V cannot double-buffer (2×64 KB > 99 KB optin),
   and processing 2 pages/block to enable overlap measured **8.54 µs** (worse —
   single-buffered V serializes the two pages). Triton closes this last sliver
   with **2× finer cp.async granularity** (16 576 vs 8 704 LDGSTS/grid) for
   better load MLP — i.e. the one remaining edge is in the DRAM→smem stage the
   mission scoped OUT ("DO NOT touch").
2. **Merge: 2.08 vs 1.79 µs (0.29 µs).** Already at the 16-chunk floor (matches
   the independently-measured lever-5 number); the single-head HD_SPLIT=2 merge
   beat both merge-v2 (2.81) and HD_SPLIT∈{1,4}. It is near-optimal for its
   chunk count; the residual is its 100 %-theoretical-occupancy tail.

**Net: the ldmatrix + geometry rewrite removed the structural page-64↔32-chunk
coupling and the 5.4-way-conflict software V-gather that the Phase-1/2 diagnosis
identified as "the only remaining lever," taking the kernel from 8.22 → 6.7 µs
(1.43×→1.16× of Triton). The last 0.9 µs is load-issue granularity in the
cp.async stage at 1-block/SM occupancy — a DRAM→smem-stage change that was
explicitly out of scope.**

## Reproduce
```
# bench container (GPU0 exclusive), ncu mounted from host:
sudo docker run -d --name msa-bench --runtime=nvidia --gpus '"device=0"' \
  --cap-add SYS_ADMIN --cap-add PERFMON --network host --ipc host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v /home/kacper/msa-120/nvfp4_serving:/work -v /home/kacper/models:/models:ro \
  -v /opt/nvidia/nsight-compute:/opt/nvidia/nsight-compute:ro \
  --entrypoint sleep vllm/vllm-openai:minimax-m3 infinity
sudo docker exec msa-bench bash -lc 'cd /work && W4=3 python3 decode_kernel/verify_decode_p128.py'
sudo docker exec msa-bench bash -lc 'cd /work && W4=3 CHUNKS128=16 python3 decode_kernel/bench_decode_p128.py'
```
