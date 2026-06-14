# MiniMax-M3-NVFP4 on SM120 (RTX PRO 6000 Blackwell) — fast, coherent serving

**Status: WORKING.** `brandonmusic/MiniMax-M3-NVFP4` serves coherently on 4× RTX PRO 6000
(compute 12.0) under vLLM at **~90 tok/s decode @ batch-1** (CUDA-graph mode), on the
**correct NVFP4 checkpoint** (not MXFP4). This is the production path.

## TL;DR — what made it work

Two things, in order of importance:

1. **The expert weights were silently loading as ZERO.** The model's `get_expert_mapping`
   passed `ckpt_gate_proj_name="w1"` (etc.) to `fused_moe_make_expert_params_mapping`, which
   builds weight-match substrings like `experts.0.w1.`. The checkpoint keys are
   `experts.0.gate_proj.weight`, so **nothing matched → all 128 experts/rank stayed zero on
   every MoE layer.** Output was word-salad/gibberish. Fix: pass the real checkpoint names
   `gate_proj` / `down_proj` / `up_proj` (the internal shard ids w1/w2/w3 are generated
   inside `make_expert_params_mapping`). See `patches/model.py`.
   **This bug ALSO crippled the marlin path** — the earlier "marlin numerics are broken on
   SM120" conclusion was wrong; it was just zero experts.

2. **Use the native MARLIN fused NVFP4 MoE, not a hand-rolled per-expert loop.** Once experts
   load, vLLM's marlin NVFP4 MoE is both correct AND fast on SM120, and it captures CUDA
   graphs cleanly (the per-expert python loop could not — data-dependent shapes). Forced via
   `VLLM_TEST_FORCE_FP8_MARLIN=1` (read by `fused_moe/oracle/nvfp4.py`). Marlin needs the
   swigluoai clamp limit, which doesn't reach `FusedMoEQuantConfig` on this build, so we
   supply it via env (`patches/marlin_moe.py` has the fallback): `VLLM_FORCE_SWIGLU_CLAMP_LIMIT`,
   `_ALPHA`, `_BETA`.

## Measured (4× RTX PRO 6000, TP4, decode@bs1, 512-tok prompt, temp=0)

| Path | decode @bs1 | prefill | TTFT | TPOT |
|---|---|---|---|---|
| un-fused per-expert loop (reference only) | 2.17 tok/s | 836 | 1842 ms | — |
| marlin, `--enforce-eager` | 18.7 tok/s | 2607 | 167 ms | 53 ms |
| **marlin, CUDA-graph (production)** | **90.4 tok/s** | 5366 | 95 ms | 11 ms |

Concurrency sweep (graph): conc 1/8/32 → **79 / 474 / 962 tok/s** aggregate, 0 failures.
~parity with the MXFP4 variant (~102 tok/s).

Sanity: "capital of Poland" → *Warsaw*; "60 km in 45 min" → *80 km/h*; correct `fib()`.

## Reproduce

Base image: `vllm/vllm-openai:minimax-m3` (vllm 0.1.dev17492+g454b47db8, flashinfer 0.6.12,
CUDA 13). Checkpoint at `/home/kacper/models/MiniMax-M3-NVFP4`.

### 1. config.json edits (one-time, on the checkpoint)
A backup is at `config.json.orig`. Changes:
- remove `text_config.layer_types`
- remove `quantization_config.kv_cache_scheme`  (serve KV in bf16; see note)
- `hidden_act`: `silu` → `swigluoai`

### 2. Launch
```bash
./launch_marlin.sh          # graph mode (default, FAST ~90 tok/s)
./launch_marlin.sh eager    # eager (debug, ~19 tok/s)
```
The script mounts the three overlays in `patches/` over the image and sets the marlin +
swigluoai env. TP4, `--block-size 128 --max-model-len 65536 --gpu-memory-utilization 0.95`,
bf16 KV. Marlin captures ~51 piecewise CUDA graphs at startup ("Breakable CUDA graph enabled").

### 3. Verify
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/MiniMax-M3-NVFP4","messages":[{"role":"user","content":"What is the capital of Poland?"}],"max_tokens":40,"temperature":0}'
```

## The overlays (`patches/`)
- `model.py` → `.../vllm/models/minimax_m3/nvidia/model.py` — **the expert-name fix** (only
  delta vs stock for the marlin path) + the existing hf_to_vllm_mapper weight remap.
- `modelopt.py` → `.../vllm/model_executor/layers/quantization/modelopt.py` — NVFP4 quant
  method with the marlin backend selection + swiglu_limit fallbacks.
- `marlin_moe.py` → `.../vllm/model_executor/layers/fused_moe/experts/marlin_moe.py` — env
  fallback for `gemm1_clamp_limit/alpha/beta` (swigluoai on packed w13).

## Notes / headroom
- KV is **bf16** because FP8 KV earlier tripped "FlashInfer page-128 requires trtllm-gen" on
  SM120. FP8-KV-on-marlin is under evaluation (would ~2× context). See `../` sibling dirs.
- Remaining ~11% vs MXFP4 is marlin-NVFP4 vs marlin-MXFP4 kernel cost.
- Alternatives if marlin ever regresses: the validated bit-exact un-fused path
  (`../patches_unfused/`, `../unfused_moe/`), a CUDA-graph-safe un-fused dispatch
  (`../graphsafe/`), and a flashinfer multi-group SFA-offset bug fix (`../batched_fix/`,
  a real upstream bug in `group_gemm_nvfp4_groupwise_sm120.cuh:74-76`).
- Benchmark harness: `../bench/bench_client.py`.
