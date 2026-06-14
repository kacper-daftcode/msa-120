# PLAN.md — Closing the ~11% NVFP4 decode gap on MiniMax-M3 (4× RTX PRO 6000, SM120, TP4)

Offline source analysis (read via `sudo docker exec minimax-m3-nvfp4`, no GPU run, container untouched).
Baseline = **90.4 tok/s decode@bs1** (cudagraph/PIECEWISE+FULL, breakable). MXFP4 ref ≈ **102 tok/s** → ~11% gap.

Files read (the ones actually mounted into the live container):
- MoE marlin kernel: `/home/kacper/m3_patch/marlin_moe.py`  (→ `.../fused_moe/experts/marlin_moe.py`)
- NVFP4 quant config: `/home/kacper/m3_patch/modelopt.py`     (→ `.../quantization/modelopt.py`)
- Model / attention:  `/home/kacper/m3_patch_unfused/model.py` (→ `.../models/minimax_m3/nvidia/model.py`)
- In-container: `.../layers/sparse_attn_indexer.py`, `.../compilation/breakable_cudagraph.py`,
  `.../quantization/utils/marlin_utils.py`, `.../fused_moe/oracle/nvfp4.py`
- Model config: `/home/kacper/models/MiniMax-M3-NVFP4/config.json`
- Live server log (current container, started 16:43 UTC).

---

## 0. Corrected mental model (the BASELINE.md/PROFILING.md hypothesis is WRONG for this build)

PROFILING.md and BASELINE.md blame a **per-expert Python `group_gemm` loop + `fp4_quantize` host syncs**.
That is the *un-fused emulation* path. **It is NOT what the live graph server runs.** Confirmed from source + log:

- Log: `Using 'MARLIN' NvFp4 MoE backend` and `compilation_config … cudagraph_mode FULL_AND_PIECEWISE`.
- `marlin_moe.py`: the MoE is **two single fused `ops.moe_wna16_marlin_gemm` calls** (gate_up, then down),
  with one fused swigluoai activation in between. **No Python per-expert loop.** Token→expert grouping is
  done once by `moe_align_block_size`; the GEMM is a single grouped kernel.
- `enforce_eager=False`; FULL decode graph + PIECEWISE graphs captured (51 sizes, 1…512), took 3.22 GiB.

So at bs1 the per-op-dispatch / per-expert-loop story does **not** apply. The real costs are different (below).
Net: the "un-fused B2a MoE" task in RECIPE.md is chasing a cost that the marlin graph path already eliminated.
Do **not** spend the GPU slot re-deriving that; profile to confirm the new hypothesis instead.

### Architecture facts that drive the gap (from config.json + model.py + logs)
- 60 decoder layers. **3 full-attention** layers (config `ignore`s `layers.0/1/2.self_attn*`) + **57 sparse**
  "MSA" layers = a **DeepSeek-V3.2-style sparse (DSA/lightning) indexer** (`sparse_attn_indexer.py`,
  `DeepseekV32IndexerMetadata`), top-k token selection over an FP8/FP4 indexer side-cache.
- 64 q-heads / 4 kv-heads (GQA 16:1), head_dim 128, hidden 6144. MoE: 128 experts, top-4, **1 shared expert**,
  swiglu_limit 7.0, swiglu_alpha 1.702. `scoring_func=sigmoid`, renormalize.
- Quant: checkpoint is genuine W4A4 NVFP4 (`input_activations.dynamic:False, num_bits:4`), but SM120 has **no
  native FP4 compute** (log: `marlin_utils_fp4.py:301 … Weight-only FP4 compression … Marlin`). MoE therefore
  runs **W4A16** (bf16 activations, weight-only FP4). Dense NVFP4 linears →`FlashInferCutlassNvFp4LinearKernel`.
- **No NVLink.** `nvidia-smi topo -m` = all PHB (PCIe). Log: `Custom allreduce is disabled … not supported on
  more than two PCIe-only GPUs`; `SymmMemCommunicator: Device capability 12.0 not supported`. → **every layer's
  TP4 all-reduce goes over PCIe via PYNCCL.** This is a hard latency floor and a real slice of the gap.

### WHERE the ~11% + extra latency most plausibly lives (ranked by likelihood)
1. **Breakable-cudagraph segment boundaries around the 57 sparse indexers + 3 FA layers.**
   `sparse_attn_indexer` and `unified_attention` are `@eager_break_during_capture`
   (`breakable_cudagraph.py:59`): inside capture they **end the graph segment, run the kernel eagerly on the
   capture stream, and start a fresh segment.** At replay the attention kernels are **launched eagerly every
   decode step**, interleaved with replayed GEMM segments → ~60 launch/sync boundaries per token (the
   "sawtooth"). MXFP4 of the *same model* shares this, so it's not the whole 11%, but it is the dominant
   fixed decode latency and the thing most worth measuring first.
2. **PCIe all-reduce (TP4, no NVLink, PYNCCL).** 60 layers × small all-reduces over PCIe. Irreducible given the
   hardware, but message count/overlap is partly tunable (see knobs T7/T8). Likely a few % of TPOT at bs1.
3. **Marlin W4A16 vs MXFP4's path.** Both are weight-only-FP4 on SM120 via marlin, so the GEMM cost is similar;
   the NVFP4 epilogue carries the **swigluoai clamp/alpha/beta** (alpha 1.702, clamp 7.0) which MXFP4 may or may
   not apply identically. This epilogue is cheap (memory-bound elementwise) — a small contributor at most.
   The 11% gap is more likely capture-quality / kernel-selection differences than raw GEMM FLOPs.
4. **Sampling / routing host reads** — minor at bs1, temp=0 argmax; FlashInfer top-p/top-k sampler is used.

**Primary target to confirm on the GPU slot: how much of the 11.1 ms TPOT is GPU-idle gaps at the eager-break
boundaries vs actual kernel time.** If idle ≫ kernel, the fix is graph/segment coalescing, not faster GEMMs.

---

## 1. TUNING MATRIX — low-risk knobs, ranked by (expected gain × safety)

All deltas are edits to `/home/kacper/launch_marlin.sh` (env `-e` lines or the `vllm serve` flags), then a
container relaunch + re-bench. **Change ONE axis at a time**, re-run `bench_client.py`, record in BASELINE.md.
Current effective flags: `--block-size 128 --max-model-len 65536 --gpu-memory-utilization 0.95` (no
`--enforce-eager`), TP4, bf16 KV, `VLLM_TEST_FORCE_FP8_MARLIN=1`, swiglu env forces. `max_num_batched_tokens`
defaulted to 8192; `cudagraph_capture_sizes` defaulted to 51 sizes (1…512).

| #  | Knob (exact delta) | Expected effect | Safety | Why |
|----|--------------------|-----------------|--------|-----|
| **T1** | `--gpu-memory-utilization 0.977` (was 0.95) | **0% decode speed**, +~6% KV cache (451,968 → ~480k tok) → more concurrency headroom. | **Very safe.** | Log explicitly: 0.95 w/ cudagraph profiling = effective 0.9228; 0.977 restores intended KV. No latency effect at bs1 — this is a *capacity* win, take it for free. |
| **T2** | `-e VLLM_MARLIN_INPUT_DTYPE=fp8` | **Potential biggest single decode win.** Switches MoE+dense marlin from **W4A16 → W4A8-FP8** (fp8 activations). `marlin_utils.py:512` explicitly allows fp8 on SM120 (device_capability_family 120). Halves activation bandwidth into the FP4 GEMM. | **MEDIUM — TEST CAREFULLY.** Changes numerics (fp8 act quant) and the GEMM kernel. Output is already gibberish (known numerics gap) so a *correctness* regression is invisible here — **gate this on the eventual numerics fix, or verify token-equivalence vs a known-good ref before trusting it in prod.** Perf-only, it's safe to measure. | W4A8 marlin is the SM120-blessed fast path; W4A16 is the conservative default. This is the single largest lever after CUDA graphs. |
| **T3** | `--max-num-batched-tokens 16384` (or 4096) | Decode@bs1 ~neutral; tunes prefill chunk + decode-batch interplay. 4096 → smaller chunks, lower TTFT jitter under load; 16384 → fewer chunks, higher prefill tput. | **Safe.** | Default 8192 (log). bs1 decode is insensitive; this is a prefill/concurrency knob. Sweep only if you care about TTFT/prefill. |
| **T4** | `--max-num-seqs 16` (was default 256-ish) + matched `--cuda-graph-sizes 1 2 4 8 16` | Frees graph-capture memory (3.22 GiB → <1 GiB) and capture time (19 s → few s); **redirects KV headroom**. **No bs1 decode latency change**, but lets you push T1 further. | **Safe** (if your real max concurrency ≤16, per BASELINE sweep). | 51 captured sizes up to 512 is wasteful for a bs≤16 serving profile. Capturing only the sizes you serve cuts memory + restart time. Does NOT speed a single token. |
| **T5** | `-e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=192` (was 512) | Frees ~320 MB/worker reserved for the indexer logits workspace → more KV. **No latency change.** | **Safe-ish.** Only lower if you never run very long contexts in one batch; the workspace must still fit peak indexer logits. 192 MB is comfortable for 65k ctx @ bs1; validate it doesn't OOM at your max concurrency×len. | Pure capacity reclaim. |
| **T6** | `-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | Alternative to T1: makes 0.95 mean ~0.95 of KV again (disables the cudagraph-mem estimate that shrinks KV). | **Safe.** Mutually exclusive intent with T1 — pick one. T1 (raise util) is cleaner. | Same KV-capacity goal, different mechanism. |
| **T7** | `-e VLLM_ATTENTION_BACKEND=FLASHINFER` | Changes the backend for the **3 dense FA layers only** (sparse layers hardcode `MiniMaxM3SparseBackend`, ignore this env). Possible small TTFT/decode shift on those 3 layers. | **Safe to try, low ceiling.** Currently FLASH_ATTN v2 (log). FlashInfer decode kernels are sometimes faster at bs1. | Only 3/60 layers affected → cap the upside at a couple %. Cheap to measure. |
| **T8** | Leave `disable_custom_all_reduce` as-is (cannot enable) | N/A — **no NVLink, custom AR impossible.** Note: do **not** waste a slot trying to force CUSTOM/SYMM_MEM AR; both are unsupported on SM120 PCIe (log-confirmed). | — | Documents a dead end so you don't chase it. |
| **T9** | `-e VLLM_FLASH_ATTN_VERSION=3` (if FA3 present in image) | FA3 decode kernel for the 3 dense layers. | **Low risk, low ceiling**, may silently no-op if FA3 unsupported on SM120/this build. | Same 3-layer ceiling as T7. Try only after T1/T2. |
| **T10** | **Do NOT set** `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | Would disable breakable graphs → the sparse indexer can't be captured → forces enforce-eager-equivalent → **~19 tok/s (5× slower)**. | **DANGER — known regression.** Listed as an explicit anti-knob. | Breakable cudagraph is load-bearing here; it's what makes 90 tok/s possible. |

### Ranking (expected_gain × safety), highest first
1. **T1** (free KV, zero risk) — take immediately, it's strictly good.
2. **T2** (W4A8-fp8 marlin) — **highest perf ceiling**, the most likely single source of the 11%. Gate on numerics.
3. **T4 + T5 + T6** (memory/capture reclaim) — safe, no latency change but enable larger T1 and faster restarts.
4. **T7 / T9** (dense-attn backend) — cheap, capped at ~2% (only 3 layers).
5. **T3** (batch-token tuning) — prefill/concurrency only; neutral for bs1 decode.

---

## 2. EXACT profiling invocation (run in a free GPU slot; server must be DOWN)

`profile_moe.py` currently builds the offline LLM with **`enforce_eager=True`**, which profiles the WRONG path
(eager, not the breakable-graph prod path). **Two ways to get a correct localization:**

### 2a. Quick op-table (torch profiler) — but make it match prod
Edit `LIVE_ARGS` in `profile_moe.py`: set `enforce_eager=False` and add
`gpu_memory_utilization=0.92` (leave room; the profiler self-aborts >5 GiB used, so confirm GPUs free first):
```bash
cd /home/kacper/msa-120/nvfp4_serving/bench
# confirm GPUs free first:
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
python3 profile_moe.py --tool torch --warmup 8 --decode-steps 1 --decode-tokens 4
# -> top ops by self-CUDA + self-CPU, writes moe_decode_trace.json
```
What to look for:
- **self-CUDA table:** `moe_wna16_marlin_gemm` (the two MoE GEMMs) and the sparse-indexer / flash-attn kernels
  should dominate *kernel* time. If marlin GEMM ≫ attention, the gap is GEMM (→ T2 fp8). If attention/indexer ≫
  GEMM, the gap is the eager-break attention path.
- **self-CPU table:** look for `cudaStreamSynchronize` / `cudaLaunchKernel` totals. With breakable graphs you
  should see **~60 segment-boundary launches** per token — that's the eager-break tax. Large CPU-launch time
  with small CUDA == boundary overhead, not compute.

### 2b. Definitive timeline (nsys) — the one to actually trust for "where do the gaps live"
```bash
cd /home/kacper/msa-120/nvfp4_serving/bench
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o moe_decode_profile \
  python3 profile_moe.py --tool torch --nsys-range --decode-steps 1 --decode-tokens 4
# (ensure profile_moe.py LIVE_ARGS enforce_eager=False as in 2a)

nsys stats --report cuda_api_sum      moe_decode_profile.nsys-rep   # host-side API / sync time
nsys stats --report cuda_gpu_kern_sum moe_decode_profile.nsys-rep   # GPU kernel time
```
In the timeline (ui: open `.nsys-rep` in Nsight Systems), per single decode token, measure:
- **GPU-idle gaps at the eager-break boundaries** (one before/after each sparse-indexer + dense-attn call).
  Sum(idle) / TPOT = the fraction recoverable by reducing boundaries (the #1 hypothesis). If this is large,
  the win is NOT in the GEMM — it's in graph/segment structure.
- **`moe_wna16_marlin_gemm` kernel duration** with W4A16 — note it; re-profile with `VLLM_MARLIN_INPUT_DTYPE=fp8`
  to get the W4A8 delta directly (this *is* the T2 measurement).
- **PYNCCL `all_reduce` / `ncclDevKernel`** total per token — quantifies the PCIe-TP4 floor (T2/T8 context).
- **`cuda_api_sum`:** `cudaStreamSynchronize` + `cudaLaunchKernel` totals = the host-dispatch/boundary tax.

> Reminder baked into `profile_moe.py`: it allocates all 4 GPUs and **self-aborts if >5 GiB is used on any GPU**,
> so it cannot collide with the live server. Stop the server (or use an orchestrator slot) first.

---

## 3. The 3 highest-ROI configs to benchmark SERIALLY (one container relaunch each)

Bench each with (from `/home/kacper/msa-120/nvfp4_serving/bench`):
```bash
python3 bench_client.py --out results/<tag>.json \
  --conditions "<exact flags here>"
# then the full sweep:
./bench_serving.sh
```
Compare decode@bs1 + TPOT against BASELINE.md (90.4 tok/s / 11.1 ms). Roll back if no gain.

### CONFIG 1 — "free wins" (safe baseline-plus). Expect: same speed, more KV, faster restart.
Edit `launch_marlin.sh`:
```
  --gpu-memory-utilization 0.977     # was 0.95  (T1)
  --max-num-seqs 16                  # add       (T4)
  --cuda-graph-sizes 1 2 4 8 16      # add       (T4)  [verify flag name in this build; see note]
```
add env: `-e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=192`  (T5)
Tag: `cfg1_freewins`. **Purpose:** confirm zero decode regression + reclaim memory; this becomes the new base.

### CONFIG 2 — "W4A8 marlin" (the real perf swing). Expect: the bulk of the 11%, IF it holds.
CONFIG 1 + add env:
```
  -e VLLM_MARLIN_INPUT_DTYPE=fp8     # (T2)
```
Tag: `cfg2_w4a8`. **Purpose:** measure W4A16→W4A8 decode delta. ⚠ Numerics change — output already gibberish so
speed is measurable now, but do **not** ship without token-equivalence vs a trusted reference once numerics are
fixed. This is the config the nsys "fp8 delta" in §2b directly predicts.

### CONFIG 3 — "dense-attn backend" (cheap tail). Expect: 0–2% (only 3/60 layers).
Best of CONFIG 1/2 + add env:
```
  -e VLLM_ATTENTION_BACKEND=FLASHINFER   # (T7)   [+ optionally -e VLLM_FLASH_ATTN_VERSION=3 (T9)]
```
Tag: `cfg3_fi_attn`. **Purpose:** squeeze the 3 dense FA layers. Keep only if it beats CONFIG 2.

---

## 4. Correctness / stability flags (read before you sweep)

- **T2 `VLLM_MARLIN_INPUT_DTYPE=fp8` changes numerics** (fp8 activation quant). The model already emits gibberish
  (known NVFP4 numerics gap), so a *new* correctness regression will be **invisible to eyeballing output**.
  Treat T2 as **perf-measurement-only** until the numerics fix lands; then verify token-level equivalence.
- **Never set `VLLM_USE_BREAKABLE_CUDAGRAPH=0`** (T10): collapses to ~19 tok/s. It's the load-bearing mechanism
  that lets the un-capturable sparse indexer coexist with CUDA graphs.
- **`--cuda-graph-sizes` flag name** is build-dependent. In this vLLM the capture set comes from
  `compilation_config.cudagraph_capture_sizes`. If `--cuda-graph-sizes` is rejected at startup, use
  `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}'` instead. Verify against `vllm serve --help`
  in the image before relying on it; a wrong flag will fail fast at launch (safe).
- **T4/T5 memory knobs** must not starve peak workspaces: validate the container reaches "Application startup
  complete" AND serves a non-empty completion (the engine warms ~85 s: weight load + flashinfer autotune +
  graph capture; `/v1/models` returns 200 before tokens flow — wait per BASELINE.md caveats).
- **One axis per relaunch.** The container takes ~85 s to become ready; serialize, don't batch changes, so each
  delta is attributable in BASELINE.md.
- **Do not chase the per-expert-loop / un-fused-MoE optimization** from RECIPE.md "THE FINISH": the live marlin
  graph path is already a single fused grouped GEMM (§0). That cost is already gone; the remaining gap is graph
  segmentation + W4A16-vs-W4A8 + PCIe AR.

---

## 5. One-line summary for the operator

Take **T1 (0.977) for free now**; the **single biggest decode lever is T2 `VLLM_MARLIN_INPUT_DTYPE=fp8`
(W4A16→W4A8 marlin)** — measure it; the rest of the 11% is breakable-cudagraph eager-break boundaries around
the 57 sparse indexers + PCIe TP4 all-reduce (no NVLink), which an **nsys timeline (§2b) will quantify as
GPU-idle gaps**. Bench Config 1 → 2 → 3 serially with `bench_client.py`.
