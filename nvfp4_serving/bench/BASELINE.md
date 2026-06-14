# BASELINE.md — MiniMax-M3-NVFP4 serving baseline (the marathon scoreboard)

Measured against the **live** server with the stdlib client `bench_client.py`
(no GPU, no deps) over `POST /v1/completions`, streaming, `temperature=0`,
`ignore_eos`, `min_tokens==max_tokens` (forced full-length decode). Output is
gibberish (known numerics gap, RECIPE.md) — this is a **performance** baseline only;
speed is unaffected by output correctness.

| Field | Value |
|-------|-------|
| Date | 2026-06-14 (UTC) |
| Model | `/models/MiniMax-M3-NVFP4` |
| Endpoint | `http://localhost:8000/v1/completions` |
| Server build | `vllm-0.1.dev17492+g454b47db8-tp4-c22b5ccc` |
| Hardware | 4× RTX PRO 6000 Blackwell (SM120), TP4 |
| Quant path | NVFP4 weights via **forced Marlin FP4** (no native SM120 FP4: weight-only FP4 → dequant-to-bf16 + per-expert Python MoE loop) |
| Harness | `bench/bench_client.py` · `bench/bench_serving.sh` |
| Raw JSON | `bench/results/marlin_eager.json`, `marlin_graph.json`, `live_cudagraph.json` |

Common conditions: `--tensor-parallel-size 4 --block-size 128 --max-model-len 65536
--gpu-memory-utilization 0.95`, bf16 KV cache, FLASH_ATTN attention backend, KV cache
≈ 19 GiB/GPU (451,968 tokens, ~6.9× max concurrency).

---

## TL;DR — two configs measured; CUDA graphs are a free ~4.8× decode win

| Metric (bs1, 512-tok prompt) | enforce-eager (spec) | **cudagraphs (live now)** | speedup |
|------------------------------|----------------------|---------------------------|---------|
| **Decode throughput** | **18.9 tok/s** | **90.5 tok/s** | **4.8×** |
| TPOT (inter-token) | 52.9 ms/tok | **11.1 ms/tok** | 4.8× |
| TTFT | 194 ms | 22–93 ms | ~2–9× |
| Prefill throughput | ~2.6–5.4k tok/s | ~5.5k tok/s | ~1–2× |

**The single biggest lever found so far is dropping `--enforce-eager`** (enabling
PIECEWISE/breakable CUDA graphs). The eager path is dominated by per-op host dispatch
+ host-sync stalls (per-expert Python `group_gemm` loop + `fp4_quantize`); graph
capture amortizes the dispatch and recovers ~4.8× on decode. This is consistent with
the PROFILING.md hypothesis and is the first scoreboard entry.

> Note: the live container was **swapped during capture** — the original
> `--enforce-eager` container (the spec'd baseline) was replaced at 16:43:38 UTC by a
> new container with `--enforce-eager` REMOVED (logs show "Breakable CUDA graph
> enabled" + fp4_gemm autotune + "Capturing CUDA graphs ... PIECEWISE"). So the
> **current live server runs the cudagraphs config.** Both are recorded below; always
> tag which config a number came from — they are NOT comparable.

---

## CONFIG A — enforce-eager (the spec'd baseline conditions)
`...--enforce-eager...` · captured before the container swap.
Sources: `results/baseline.json` (decode n=5, prefill n=5), `results/marlin_eager.json`.

| Metric | Value |
|--------|-------|
| **Decode @ bs1** | **18.9 tok/s** (median; mean 18.93; n=5×128 out-tok) |
| TTFT (512-tok prompt) | 194 ms median (158 mean, 206 p99) |
| TPOT | **52.9 ms/tok** median (60.1 ms p99 ITL) |
| Prefill (512-tok, max_tokens=1) | **5421 tok/s** median (mean 6179; 515 prompt-tok) |

Concurrency sweep under enforce-eager is **not reliably measured**: the container was
swapped out mid-sweep, so all concurrency=4/16 requests in `baseline.json` got
`ConnectionRefusedError(111)` (a swap artifact, not a load crash). The clean
concurrency=1 partial (3/32) read 19.0 tok/s/req, matching the bs1 number. Re-run the
eager sweep only if eager is restored as a config of interest.

## CONFIG B — cudagraphs (CURRENT LIVE SERVER) ⭐ full clean sweep
`--enforce-eager` REMOVED, PIECEWISE CUDA graphs on. Everything else identical.
Source: `results/live_cudagraph.json` (decode n=5, prefill n=5, sweep 32 prompts/level,
**0 failures at every concurrency**).

| Metric | Value |
|--------|-------|
| **Decode @ bs1** | **90.5 tok/s** (median; mean 86.2; n=5×128 out-tok) |
| TTFT (512-tok prompt) | 22 ms median (44 mean, 118 p99) |
| TPOT | **11.1 ms/tok** median (12.95 ms p99 ITL) |
| Prefill (512-tok, max_tokens=1) | **5529 tok/s** median (mean 9135; 515 prompt-tok) |

### Offered-load sweep (cudagraphs, 32 prompts × 64 out-tok each)
| Concurrency | System out tok/s | req/s | per-req decode tok/s | TTFT ms (med) | TPOT ms (med) | fail |
|-------------|------------------|-------|----------------------|---------------|---------------|------|
| 1 | 81.7 | 1.28 | 91.4 | 92.0 | 10.9 | 0/32 |
| 4 | 244.3 | 3.82 | 62.8 | 43.8 | 15.9 | 0/32 |
| 16 | 642.7 | 10.04 | 41.2 | 61.1 | 24.3 | 0/32 |

System throughput scales cleanly 1→4→16 (≈82 → 244 → 643 out tok/s, ~7.9× at conc16);
per-request decode degrades 91 → 63 → 41 tok/s as batching contends — expected, and the
batched MoE path is the contention point the marathon will attack.

---

## Context vs the MXFP4 reference (~102 tok/s decode @bs1)
| Config | Decode @bs1 | vs MXFP4 (102) |
|--------|-------------|----------------|
| NVFP4 enforce-eager | 18.9 tok/s | **5.4× slower** (~18.5% of MXFP4) |
| NVFP4 cudagraphs | 90.5 tok/s | **~1.13× slower** (~89% of MXFP4) |

The eager NVFP4 path is the "MUCH slower" regression the marathon was called to fix;
**simply enabling CUDA graphs closes most of the gap to MXFP4** (102 → 90.5). The
remaining ~11% and all of the multi-stream contention live in the per-expert
Python `group_gemm` loop + `fp4_quantize` host syncs — quantify with `profile_moe.py`
(needs a free GPU slot) and attack per RECIPE.md §"THE FINISH" (un-fused B2a MoE).

---

## Reproduce
```bash
cd /home/kacper/msa-120/nvfp4_serving/bench

# stdlib client (no GPU, no deps) — full decode/prefill/TTFT/TPOT + 1/4/16 sweep:
./bench_serving.sh
# tag the config explicitly so the JSON meta is honest:
python3 bench_client.py --out results/run.json \
  --conditions "max-model-len 65536, block-size 128, bf16 KV, TP4, CUDA-graphs(piecewise), gpu-mem-util 0.95"

# cross-check with the in-image benchmark:
BACKEND=vllm ./bench_serving.sh        # vllm bench serve, random dataset, per-concurrency
```
Knobs (env or flags): `INPUT_LEN OUTPUT_LEN NUM_PROMPTS CONCURRENCY SWEEP_OUTPUT_LEN
HOST MODEL`. Re-run after each optimization; this file is the scoreboard.

**Operational caveats observed during capture**
- After a container (re)start it needs ~20 s (weight load) + flashinfer `fp4_gemm`
  autotune + CUDA-graph capture before it serves; `/v1/models` returns 200 *before*
  completions actually produce tokens. `bench_client.py` waits on `/v1/models`; if you
  see empty completions, the engine is still warming — wait for "Application startup
  complete" + a non-empty test completion. (`--health-timeout` covers reachability.)
- The original eager `conditions` string is a client default; pass `--conditions` to
  record the real config (the `live_cudagraph.json` meta still says "enforce-eager"
  because it predates that flag — the run itself was against the cudagraphs container).
