# SM120 hand-written MSA kernels vs vLLM Triton — head-to-head per-op latency

Box: 4x RTX PRO 6000 Blackwell Server (SM120, 188 SMs, 1.7 TB/s). Image
`vllm/vllm-openai:minimax-m3` (torch 2.11.0+cu130, nvcc 13.0). Date 2026-06-14.
Run on `CUDA_VISIBLE_DEVICES=0` in a **separate** `msa-bench` container (bash
entrypoint, small tensors), alongside the live `minimax-m3-nvfp4` marlin prod
container — prod left untouched and serving. Only ~3.8 GB free on GPU 0; all
bench tensors kept small.

Shapes (real MiniMax-M3, from `/models/MiniMax-M3-NVFP4/config.json`): 64 q-heads,
4 kv-heads, head_dim 128, index_n_heads 4, index_head_dim 128, index_block_size
128, index_topk_blocks 16, index_local_blocks 1, init_blocks 0, rope_theta 5e6,
partial_rotary_factor 0.5 (rotary_dim 64), bf16.

Timing: CUDA events, warmup 20 + many iters, median. Two columns reported:
- **kernel-only (us)** — torch-profiler CUDA self-time (on-GPU only, excludes
  host/python). This is the fair kernel-vs-kernel number.
- **wall/batched (us)** — CUDA-event time of the full callable incl. python
  dispatch + (for Triton) per-call autotune/multi-kernel launch overhead. This is
  what an eager (non-cudagraph) caller actually pays per op.

`ratio = triton / ours`  (**>1 = WE ARE FASTER**).

================================================================================

## VERDICT

**MIXED, and on the regime that matters most for interactive serving (bs1
decode) we LOSE the attention op badly.**

- **Op 1 — top-k select: WIN to mixed.** Kernel-only we win in prefill (2.2–2.3x)
  and at long decode context (2.9x @64k), but **lose at short decode context**
  (0.52x @4k, 0.87x @16k) — Triton's lighter 2-kernel path beats our flat-time
  kernel there. On the **wall/dispatch** number we win everywhere (3–5x), because
  Triton's topk is an autotuned 1-kernel (prefill) / 3-kernel (decode) path with
  real per-call overhead. Honest take: our kernel has **flat ~4us** regardless of
  context; Triton scales with nblk, so we win as context grows and on dispatch
  cost, but the raw kernel is not faster at small nblk.

- **Op 2 — indexer block score: NOT a fair head-to-head (entrypoint asymmetry).**
  Our `block_scores` does the **full project + qk-norm + RoPE + score** pipeline
  from hidden states over all N tokens; vLLM's `minimax_m3_index_score` does
  **score-only** from a pre-projected `idx_q` (the projection lives in the model's
  fused qkv GEMM, not in this op). So the numbers below compare our whole indexer
  front-end against Triton's score sub-stage — apples to oranges. We have **no
  score-only entrypoint** to isolate the comparable stage (RESULTS.md blocker D).
  Reported for completeness, flagged NOT-FAIR; do not read it as a kernel loss.

- **Op 3 — sparse paged attend: TIE in prefill, LOSE ~31x in bs1 decode.**
  Prefill is a real tie (0.96–1.01x, both correct). **bs1 decode: 0.03x
  kernel-only (179us vs 5.7us) — our worst result, and it is the
  interactive-critical regime.** Fully profiled below; the limiter is occupancy
  starvation + a prefill-shaped 64-query M-tile with 1 real row at decode (no
  split-K, no GQA-group tiling). This is the concrete SASS/kernel target.

So: our SM120 kernels are **not uniformly faster** than generic Triton on M3
shapes. We win top-k on the wall-cost and at long context; attention prefill
ties; **attention decode — the bs1 path the whole sparse design exists to serve —
is 31x slower** and is the actionable gap.

================================================================================

## HEAD-TO-HEAD TABLE

### Op 1 — top-k block select  (ours `topk_select` vs Triton `minimax_m3_index_topk`)
Set-exact equivalent (proven in `op_equivalence_topk.py`). Same canonical scores,
each fed its native layout.

| regime  | q_len | seq_kv | nblk | ours us (kern / wall) | triton us (kern / wall) | ratio kern | ratio wall |
|---------|------:|-------:|-----:|----------------------:|------------------------:|-----------:|-----------:|
| decode  | 1     | 4096   | 32   | 4.25 / 9.2            | 2.19 / 24.4             | **0.52**   | 2.65       |
| decode  | 1     | 16384  | 128  | 4.10 / 9.4            | 3.58 / 23.5             | **0.87**   | 2.50       |
| decode  | 1     | 65536  | 512  | 4.00 / 9.5            | 11.76 / 32.0            | **2.94**   | 3.37       |
| prefill | 512   | 512    | 4    | 1.72 / 7.9            | 4.03 / 27.6             | **2.34**   | 3.49       |
| prefill | 2048  | 2048   | 16   | 4.98 / 9.6            | 11.05 / 32.0            | **2.22**   | 3.34       |

Our kernel is ~flat (one cub-based selection launch); Triton's time grows with
nblk and its wall cost carries autotune + (decode) 3-kernel split-K/merge.

### Op 2 — indexer block score  (ours `block_scores` vs Triton `minimax_m3_index_score`)  ⚠ NOT FAIR
Ours = full project+norm+rope+score over N tokens. Triton = score-only over q_len
queries. Asymmetric; numbers for completeness only.

| regime  | q_len | seq_kv | N (ours scores) | ours us (wall) | triton us (wall) | fair? |
|---------|------:|-------:|----------------:|---------------:|-----------------:|:-----:|
| decode  | 1     | 4096   | 4096            | 1477.7         | 64.2             | NO (ours re-scores whole ctx) |
| decode  | 1     | 16384  | 16384           | 13059.9        | 244.8            | NO |
| decode  | 1     | 65536  | 65536           | OOM smem*      | 966.9            | NO |
| prefill | 512   | 512    | 512             | 480.7          | 15.9             | partial (ours also projects) |
| prefill | 2048  | 2048   | 2048            | 809.0          | 35.4             | partial (ours also projects) |

\* Our score kernel's dynamic smem is `BLK_M*nblk*4` bytes; at nblk=512 it
exceeds the SM120 102 KB cap (`cudaErrorInvalidValue`). Capped to nblk<=360.
Even the "partial-fair" prefill rows bundle the projection GEMM (hidden→H·d over
all N) into ours, which Triton's op does not do — so the ~25-30x is dominated by
work Triton doesn't perform here, **not** a scoring-kernel deficit. A fair
score-only comparison needs the missing `index_block_scores(idx_q, paged_K, …)`
entrypoint (RESULTS.md blocker D).

### Op 3 — sparse paged attend  (ours `forward_sparse_paged` page-64 vs Triton page-128)
Each fed its NATIVE page layout, SAME logical problem (same seqlen, same selected
block set = last `min(16,nblk)` blocks, same heads). Decode uses Triton's split-K
`minimax_m3_sparse_attn_decode`; prefill uses `minimax_m3_sparse_attn`. Both
outputs independently verified correct (decode rms 2.6e-3, prefill rms 6.9e-3).

| regime  | q_len | seq_kv | sel | ours us (kern / wall) | triton us (kern / wall) | ratio kern |
|---------|------:|-------:|----:|----------------------:|------------------------:|-----------:|
| decode  | 1     | 4096   | 16  | 179.5 / 187.9         | 5.73 / 42.3             | **0.03** ⬅ worst |
| decode  | 1     | 16384  | 16  | 179.2 / 187.5         | 5.72 / 41.8             | **0.03**   |
| decode  | 1     | 65536  | 16  | 180.1 / 188.8         | 5.80 / 41.2             | **0.03**   |
| prefill | 512   | 512    | 4   | 137.5 / 146.8         | 132.7 / 159.1           | **0.96**   |
| prefill | 2048  | 2048   | 16  | 1720.4 / 1677.5       | 1739.4 / 1703.8         | **1.01**   |

Our decode time is FLAT across context (only the 16 selected blocks are touched)
but 31x Triton — see profile.

================================================================================

## NOTES / CAVEATS (honesty)

- **bs1 caveat on serving relevance:** RESULTS.md sec.1 established that at bs1
  decode the marlin prod path captures a FULL cudagraph, so per-op *dispatch*
  overhead (the wall-vs-kernel gap) is amortized in production. The fair number
  for a captured graph is therefore **kernel-only**, on which our decode attend
  is 31x slower and our short-context topk is also slower — the gap is real GPU
  time, not launch overhead.
- **topk prefill correctness vs latency:** our topk benchmark passes a single
  `num_valid=nblk` for all queries (full nblk scan); Triton applies per-query
  causal `valid_blocks` so it scans fewer blocks for early queries. Both do the
  full-nblk work in the worst case; the latency comparison is over the same
  candidate count. Set-equivalence itself is already proven per-query elsewhere.
- **No ncu/nsys in the image** — deep-dive uses torch profiler CUDA self-time +
  occupancy math from launch geometry (see `profile_sparse_attend_decode.txt`).
- Indexer 64k-decode could not run ours (smem cap) — reported as OOM, not faked.

## FILES
- `bench_common.py` — shapes + CUDA-event timing helpers.
- `bench_msa.py` — the 3-op head-to-head (CUDA-event batched + wall).
- `kernel_only.py` — torch-profiler kernel-only times (the fair column).
- `profile_attend_decode.py` + `profile_sparse_attend_decode.txt` — the worst-op
  deep dive (occupancy/limiter analysis).
- `verify_attend.py` — output-correctness sanity for the paged attend.
- `results/*.json` — raw numbers.
