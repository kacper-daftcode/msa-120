# NVFP4-on-SM120 optimization marathon — what worked, what's walled off

Starting point: marlin fused NVFP4 MoE + CUDA graphs = **90.4 tok/s decode @bs1** (see README).
We then tried every lever to push further. Summary: **90 tok/s is at/near the practical
ceiling for this checkpoint on this hardware** — the big levers are blocked by concrete
SM120/model constraints, documented below so nobody re-spends the GPU slot on them.

## Applied (kept)

### `--gpu-memory-utilization 0.977` — FREE KV headroom
CUDA-graph memory profiling makes 0.95 effectively ~0.9228. Raising to 0.977:
- GPU KV cache **251k → 512.8k tokens** (~2×), max concurrency **1.92× → 7.82×**.
- Decode speed unchanged (90.8 tok/s), output still coherent. Zero risk. **In the default launch.**

## Tried and REJECTED (with the exact reason — don't retry without a kernel change)

### ngram speculative decoding — NET LOSS (60.8 vs 90.4 tok/s)
- The NVFP4 checkpoint **ships no MTP weights** (the 7 MiniMax-M3 MTP modules were dropped in
  export), and no EAGLE/Medusa head exists for M3 → ngram (prompt-lookup) is the only
  zero-dependency option.
- Measured acceptance on a structured prompt: **7 accepted / 816 drafted = 0.86%**. Reasoning/
  generative output is novel at the token level, so ngram almost never hits.
- Worse, the spec path forces decode cudagraph **FULL→PIECEWISE**, losing the full-graph win.
- Net: warm throughput **60.8 tok/s** vs 90.4 baseline. Only wins when re-emitting given text
  verbatim (long code-edit echo). Not a default. Re-enable per-request via
  `nvfp4_serving_spec/launch_marlin_ngram.sh` only for echo-heavy workloads.
- High-upside path (~2-3×) = quantize + ship the 7 MTP modules as `minimax_m3_mtp` draft.
  That's an offline NVFP4 quant task on the MTP heads — open follow-up.

### W4A8 (`VLLM_MARLIN_INPUT_DTYPE=fp8`) — UNSUPPORTED
- Boot fails hard: `RuntimeError: NVFP4 weight + INT8/FP8 activation is not supported.`
- marlin on SM120 is **weight-only W4A16** (no native FP4 compute; activations stay bf16).
  FP8 activations are explicitly guarded off for NVFP4 weights in this build. Dead end without
  an upstream marlin kernel change.

### FP8 KV cache — BLOCKED by the model's fused kernel
- Backend selection works (`--attention-backend TRITON_ATTN` avoids the FlashInfer page-128 /
  trtllm-gen trap; FLASH_ATTN rejects fp8 on SM120-family-120; auto-select would crash on
  FlashInfer). The 57 sparse-MSA layers already run the Triton sparse impl and handle fp8.
- BUT the 3 dense full-attention layers use a custom fused kernel
  `fused_minimax_m3_qknorm_rope_kv_insert` that **hardcodes bf16 KV**:
  `RuntimeError: ... kv_cache dtype must match qkv (bf16 cache only)`.
- So fp8 KV needs that CUDA kernel patched (`csrc/.../fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`).
  Out of scope for a flag-level win. Until then KV stays bf16. Script kept at
  `nvfp4_serving_fp8kv/` for when the kernel is patched.

## The hard floors (why 90 is the ceiling)
- **No NVLink.** `nvidia-smi topo` = all PHB; custom all-reduce + SYMM_MEM unsupported on SM120
  PCIe → every layer's TP4 all-reduce goes over PCIe via PYNCCL. Fixed cost.
- **Eager-break boundaries.** The 57 MSA-indexer layers + 3 dense layers are
  `@eager_break_during_capture`; at replay the attention kernels run eagerly on the capture
  stream (~60 segment boundaries/token sawtooth) — the dominant fixed decode latency. Structural
  to how vLLM handles the un-capturable sparse indexer; not a flag.
- **W4A16 marlin** (above) — no FP4 tensor-core path on SM120.

## Net result
Production = marlin fused NVFP4 MoE + CUDA graphs + `gpu-mem 0.977`:
**90.8 tok/s decode @bs1, 5.3k tok/s prefill, 512.8k-token KV (7.82× concurrency), coherent.**
~89% of the MXFP4 reference (~102), on the correct NVFP4 checkpoint. The remaining gap and all
three rejected levers are gated on kernel/upstream work, not configuration.
