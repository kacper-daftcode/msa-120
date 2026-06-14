# FP8 KV cache on MiniMax-M3-NVFP4 / Marlin / SM120 — viability + launch plan

Date: 2026-06-14. Image: `vllm/vllm-openai:minimax-m3`. GPUs: 4x RTX PRO 6000
Blackwell, compute_cap **12.0** (family 120). Source inspected live via
`docker exec minimax-m3-nvfp4` (no GPU code run, container untouched).

---

## GO / NO-GO: **GO** (conditional — must force the Triton backend)

FP8 KV is viable on this Marlin + SM120 path, **but only if you force
`--attention-backend TRITON_ATTN`**. With auto-select it will pick FlashInfer,
which crashes on the page-128 / trtllm-gen guard (the exact earlier failure).

Why it works at all:
- The model has 3 full-attention layers (standard `Attention()`) + 57 sparse-MSA
  layers (custom `MiniMaxM3SparseBackend`). **Both** paths support fp8 KV at
  block-size 128 with **no trtllm-gen dependency** — *as long as the full-attn
  layers run on Triton, not FlashInfer*.
- The 57 sparse layers are **already** on the Triton sparse impl today (even with
  bf16 KV). The MSA fast kernel (`sparse_attention_msa`, fmha_sm100) is gated on
  `is_device_capability_family(100)` AND non-quantized KV — SM120 is family 120,
  so `select_main_impl_cls()` already returns `MiniMaxM3SparseTritonImpl`. The
  Triton sparse kernels dequant fp8 before the dots
  (`supported_kv_cache_dtypes = [bfloat16, fp8, fp8_e4m3, fp8_e5m2]`,
  block size fixed at 128). So fp8 KV changes nothing structural on the sparse
  path; it just stores e4m3 instead of bf16.

---

## The SM120 constraint map (load-bearing facts)

Per-backend `supports_kv_cache_dtype(fp8)` and the page-128 behaviour, all read
from this image:

| Backend | fp8 KV on SM120? | block 128? | trtllm-gen / page-128 risk |
|---|---|---|---|
| **FLASH_ATTN** (current prod) | **NO** — requires `fa_version==3 AND is_device_capability_family(90)`. SM120 is family 120 → rejected. Image runs FA2 anyway. | MultipleOf(16) ok | n/a (rejected before that) |
| **FLASHINFER** | YES at validate-time (`fp8/fp8_e4m3`, cc 7.5–12.x) | lists 128 as "supported" so it PASSES validation | **CRASHES at runtime**: `FlashInferMetadataBuilder.__init__` raises for `page_size >= 128` → needs trtllm-gen dynamic kernel (Blackwell + GQA + NVIDIA artifactory cubins). This is the earlier "page-128 requires trtllm-gen" failure. |
| **TRITON_ATTN** | **YES** — `supported_kv_cache_dtypes` includes `fp8`,`fp8_e4m3`,`fp8_e5m2`; `supports_compute_capability()` returns `True` unconditionally | MultipleOf(16) → 128 ok | **NONE** — no trtllm-gen path exists in this backend. Dequants fp8 in-kernel. |
| MINIMAX_M3_SPARSE (the 57 layers) | **YES** — Triton sparse impl, fixed block 128, fp8 dequant-before-dots | fixed 128 | NONE |
| FLEX_ATTENTION / TURBOQUANT | lower priority fallbacks; not needed | — | — |

### Auto-select priority on SM120 (major==12, non-MLA), from `_get_backend_priorities`:
`[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]`

With `--kv-cache-dtype fp8`, validation drops FLASH_ATTN (fp8 unsupported), so the
**first valid backend is FLASHINFER** → selected → runtime page-128 crash.
`validate_configuration()` does NOT catch the page-128 problem because FlashInfer
advertises 128 in `get_supported_kernel_block_sizes()`; the failure only fires in
the metadata builder during/after warmup. **Hence: force Triton.**

Note: the 57 sparse layers register their own `MiniMaxM3SparseBackend` via the
model's `get_attn_backend()` and are **unaffected by `--attention-backend`** — that
flag only steers the 3 standard `Attention()` full-attn layers. So forcing Triton
is purely about keeping those 3 layers off FlashInfer.

---

## KV scales: dynamic / identity — checkpoint `kv_cache_scheme` NOT required

- Active `config.json` `quantization_config` has **no** `kv_cache_scheme` (it was
  removed). `hf_quant_config.json` still declares it (fp8, static, 8-bit) but vLLM
  reads `config.json`, so the model loads with **no static KV scales applied**.
- Full-attn layers: vLLM falls back to default per-tensor scales (1.0) unless
  `k_scale`/`v_scale` tensors are present in the weights; `maybe_remap_kv_scale_name`
  in `model.py` will pick them up *if* the checkpoint ships them, otherwise dynamic.
- Sparse layers: `_insert_kv()` calls `reshape_and_cache_flash(..., scale=ones())`
  — an **identity (1.0) static scale**. fp8 store = plain cast to e4m3, no rescale.
- **Conclusion:** fp8 KV works with dynamic/identity scales; you do NOT need to
  restore `kv_cache_scheme` in config.json for it to run. Restoring it is a
  *possible accuracy lever* later (calibrated per-tensor scales), but it is also a
  *risk*: if vLLM then expects per-layer `k_scale`/`v_scale` weights that the
  checkpoint doesn't actually contain, load can fail. Keep it OUT for the first run.

### Numerical risk to watch (sparse path)
Identity-scale e4m3 has max representable ~448. If any K/V magnitude exceeds that,
sparse-layer values saturate → degraded/garbage output on long context. e4m3 also
loses precision vs bf16. Validate output quality on a real long prompt, not just
"it starts". If quality regresses, the lever is calibrated scales (config route).

---

## Ranked candidate configs (most → least likely to work)

### #1 (BEST — use this first): force Triton + fp8 KV, keep block 128
Deltas vs `launch_marlin.sh`:
```
+ --kv-cache-dtype fp8
+ --attention-backend TRITON_ATTN
```
Everything else identical (TP4, block-size 128, max-model-len 65536, gpu-mem 0.95,
all env + overlays). This is `launch_marlin_fp8kv.sh`.
- Pros: avoids FlashInfer entirely → no page-128/trtllm-gen path; Triton supports
  fp8 + block 128 on SM120; sparse layers already Triton.
- Watch: CUDA-graph capture under Triton fp8 (should capture; Triton attn is
  graph-capturable). Decode tok/s may differ from FA2 baseline (90 tok/s) — Triton
  attn can be a bit slower per-step but fp8 KV halves KV bandwidth; net depends on
  context length. Expect roughly comparable decode at bs1, big win on max context.
- Context win: fp8 KV halves per-token KV bytes → ~2x usable context for the same
  0.95 GPU mem budget. You can optionally bump `--max-model-len` toward ~131072
  after confirming it boots (separate change; not in the script to keep the A/B
  clean).

### #2 (fallback if #1 has a Triton CUDA-graph issue): #1 + eager
```
+ --kv-cache-dtype fp8
+ --attention-backend TRITON_ATTN
+ --enforce-eager
```
Or pass `eager` as `$1` to the script. Slower (~19 tok/s) but isolates whether a
failure is graph-capture vs the fp8 path itself. Diagnostic only.

### #3 (only if you must keep FlashInfer / trtllm-gen becomes available): drop block to 64
```
+ --kv-cache-dtype fp8
+ --attention-backend FLASHINFER
+ --block-size 64
```
NOT recommended: `--block-size 64` changes the sparse backend contract
(`MiniMaxM3SparseBackend.get_supported_kernel_block_sizes()` is **fixed at [128]**,
sparse block == page == 128). A 64 page size will be rejected by the sparse layers
→ won't boot. Listed only to document why block<128 is a dead end here.

### #4 (NO-GO): auto-select (no `--attention-backend`)
```
+ --kv-cache-dtype fp8        # and nothing else
```
Will auto-pick FLASHINFER → page-128/trtllm-gen crash. Do not use.

---

## Exact failure signatures to watch in `docker logs -f minimax-m3-nvfp4`

1. **Page-128 / trtllm-gen (means FlashInfer got selected — wrong path):**
   - `FlashInfer page size 128 requires the trtllm-gen backend ...`
   - `... requires the trtllm-gen backend (Blackwell with NVIDIA artifactory access ...`
   - `... only supported by the trtllm-gen dynamic kernel, which requires GQA/MQA ... not MHA`
   → Fix: ensure `--attention-backend TRITON_ATTN` is actually applied (check the
     `Using TRITON_ATTN backend.` info log at startup).

2. **Backend rejected at startup (forced backend invalid):**
   - `Selected backend AttentionBackendEnum.TRITON_ATTN is not valid for this
     configuration. Reason: [...]` → read the reason; should not happen for fp8+128.

3. **fp8 unsupported (means it tried FlashAttn):**
   - `... kv_cache_dtype not supported ...` for FLASH_ATTN in the
     "No valid attention backend / Reasons:" dump.

4. **Sparse block-size mismatch (if someone changed block-size):**
   - errors referencing sparse block size / page size != 128, or
     `MINIMAX_M3_SPARSE` head/block asserts.

5. **Numerical (boots but bad output):** gibberish / repetition only at long
   context → e4m3 saturation on identity-scale sparse KV. Lever: calibrated scales.

6. **CUDA-graph capture failure under Triton fp8:** capture-time error mentioning
   triton_attn / dynamic shapes → fall back to candidate #2 (`--enforce-eager`).

---

## Verification checklist after launch (orchestrator)
1. `docker logs minimax-m3-nvfp4 | grep -iE "Using .* backend|TRITON_ATTN|fp8|kv.cache"`
   → expect `Using TRITON_ATTN backend.` and an fp8 KV cache line. No FlashInfer.
2. No page-128/trtllm-gen lines anywhere in the log.
3. KV cache size / `# GPU blocks` roughly **2x** the bf16 run (fp8 halves KV bytes).
4. Single short completion returns coherent text.
5. One **long-context** completion (e.g. 30k+ tokens in) returns coherent text
   (guards against e4m3 saturation).
6. Decode tok/s @ bs1 (compare to 90 baseline; some variance expected).
