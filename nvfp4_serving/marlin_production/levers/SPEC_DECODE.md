# Speculative Decoding Plan — MiniMax-M3-NVFP4 (vLLM image `vllm/vllm-openai:minimax-m3`)

Goal: multiply decode throughput at **bs=1** (interactive) on 4x RTX PRO 6000 (SM120), TP4,
without touching the live `minimax-m3-nvfp4` container. Run the candidate script below in a
**serial GPU slot** only.

This analysis was done **offline**, reading source inside the running container via
`sudo docker exec`. Nothing here was executed on GPU.

---

## 0. Build facts (this exact image)

- vLLM version: `0.1.dev17492+g454b47db8` (untagged dev build, v1 engine).
- `--speculative-config` / `-sc` takes a **JSON string**, parsed by `json.loads`
  (`vllm/engine/arg_utils.py:1463`). Format below is verified against this build.
- Supported `SpeculativeMethod` literals in this build:
  `ngram, ngram_gpu, medusa, mlp_speculator, draft_model, suffix, custom_class,
  eagle, eagle3, extract_hidden_states, dflash, mtp` + a long list of model-specific
  MTP variants **including `minimax_m3_mtp`**.
- `vllm/v1/spec_decode/` proposers present: `ngram_proposer.py`, `ngram_proposer_gpu.py`,
  `eagle.py` (EAGLE/EAGLE3 share `SpecDecodeBaseProposer`), `medusa.py`, `draft_model.py`,
  `suffix_decoding.py`, `dflash.py`, `gemma4.py`, `step3p5.py`, `custom_class_proposer.py`,
  `extract_hidden_states.py`. So ngram, EAGLE/EAGLE3, Medusa, draft-model, suffix, and MTP
  are all *mechanically* available.

## 1. Does a draft head / MTP exist for MiniMax-M3-NVFP4? — Honest answer: NO, not in your checkpoint.

This is the key finding, and it cuts both ways:

- **The vLLM build DOES support MiniMax-M3 MTP natively.** There is a real, registered
  drafter:
  - `SpeculativeConfig.hf_config_override` rewrites `minimax_m3_vl` /
    `MiniMaxM3SparseForConditionalGeneration` → `minimax_m3_mtp` /
    `architectures=["MiniMaxM3MTP"]`, reading `num_mtp_modules` for `n_predict`
    (`vllm/config/speculative.py`).
  - `MiniMaxM3MTP` is registered → `vllm.models.minimax_m3:MiniMaxM3MTP`
    (`model_executor/models/registry.py:635`), implemented in
    `vllm/models/minimax_m3/nvidia/mtp.py` (full MTP layer with `enorm/hnorm/eh_proj`
    + a sparse-MSA decoder layer, MoE-aware weight loader). The code even names the
    standalone checkpoint convention `Inferact/MiniMax-M3-MTP` and a bundled
    `language_model.*.mtp.layers.*` layout.
  - The base model `MiniMaxAI/MiniMax-M3` declares `text_config.num_mtp_modules = 7`.

- **BUT your served NVFP4 checkpoint has ZERO MTP weights.**
  - `/home/kacper/models/MiniMax-M3-NVFP4/model.safetensors.index.json`: 89,200 tensors,
    **0** matching `.mtp.layers.*` / `nextn` / `mtp`. Only `model.*` and `lm_head`.
  - The quant `hf_quant_config.json` `ignore` list (123 entries) covers main layers 0–59
    only — no `mtp` entries. The MTP modules were simply **not exported** during NVFP4
    quantization (README "What is quantized" table lists no MTP).
  - The MTP drafter loader (`mtp.py::_map_checkpoint_name`) only consumes `*.mtp.layers.*`
    (+ shared embed/lm_head). With none present it would load **nothing** → cannot run.

- **No EAGLE / EAGLE3 / Medusa head exists for MiniMax-M3** on HF (model card documents none;
  no `*eagle*`/`*medusa*` MiniMax-M3 repo found). Do not invent one.

**Conclusion:** Today, the only **zero-dependency** speculative method for the checkpoint you
serve is **ngram** (prompt-lookup). MTP is the *high-value* path but is **blocked on weights**:
you'd need to NVFP4-quantize the 7 MTP modules from `MiniMaxAI/MiniMax-M3` (they exist upstream)
and add them to the checkpoint, OR obtain/quantize a standalone `*/MiniMax-M3-MTP`. See §5.

---

## 2. Ranked candidates

### #1 — ngram (prompt-lookup) — READY NOW, zero dependency  ← run this first
- No draft model, no weights, no extra VRAM of consequence (small numba CPU buffers).
- Drafts by matching the last `prompt_lookup_min..max` tokens against earlier context and
  copying the continuation. Wins exactly when the model **repeats spans it has already seen**:
  code (re-emitting identifiers, signatures, imports, boilerplate, diffs), structured output
  (JSON/tables), long quotes, and the *reasoning → final-answer* restatement common in
  reasoning models. Near-zero benefit on free-form novel prose.
- **Exact flag (this build):**
  ```
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
  ```
- **Tuning for this reasoning workload:** start `num_speculative_tokens=4`,
  `prompt_lookup_min=2`, `prompt_lookup_max=4`. The reasoning trace is long and self-quoting,
  so a small min-window (2–3) catches more matches; cap speculative tokens at 4–5 so a missed
  guess wastes little. If acceptance looks high (see metrics) push N to 5–6 and `max` to 5;
  if you see decode regress on prose-heavy prompts, drop min to keep matches frequent.
- Alternative: `"method":"ngram_gpu"` also exists (GPU-side matcher, same JSON shape minus the
  numba path). Stick with CPU `ngram` first — it's the well-trodden path and won't contend with
  the GPUs that are already at 0.95 util.

### #2 — minimax_m3_mtp — HIGHEST upside, but BLOCKED on weights
- If/when MTP weights are present, this is the right long-term answer for bs=1: a trained
  1-step (×7 modules) predictor typically lands ~2–3× decode on reasoning/code (vendor claims
  for M-series MTP are in that range), far above ngram on novel text.
- **Flag shape (once weights exist), pointing at the MTP checkpoint:**
  ```
  --speculative-config '{"method":"minimax_m3_mtp","model":"<path-to-MTP-checkpoint>","num_speculative_tokens":1}'
  ```
  (or `num_speculative_tokens` up to the number of usable MTP modules; this build maps
  `n_predict` from `num_mtp_modules`.) **Do not launch this against the current checkpoint —
  it will fail to load any drafter weights.**

### #3 — suffix decoding — READY NOW, zero dependency, worth A/B vs ngram
- `"method":"suffix"` is present (`suffix_decoding.py`) and needs no model. It builds per-prompt
  + global suffix trees and can out-accept ngram on repetitive/agentic traffic, with tunables
  already in `SpeculativeConfig` (`suffix_decoding_max_tree_depth`, `_max_spec_factor`,
  `_min_token_prob`). Lower priority only because ngram is the most battle-tested; keep as a
  fast follow-on experiment in the same serial slot.

### Not viable / not applicable today
- **EAGLE / EAGLE3 / Medusa / mlp_speculator / draft_model** — mechanically supported by the
  image but **no MiniMax-M3 head/draft model exists**. Nothing to load. Skip.

---

## 3. Interactions with CUDA graphs, marlin MoE, MSA, TP4

- **ngram + CUDA graphs: compatible.** ngram only proposes tokens on CPU; the target model
  still runs its normal compiled/graphed forward, now verifying `N+1` tokens per step. v1 is
  spec-decode-aware: `compilation.py` has `adjust_cudagraph_sizes_for_spec_decode(...)` and
  rounds capture sizes to a multiple of `uniform_decode_query_len = num_speculative_tokens+1`.
  You do **not** need `--enforce-eager`; keep graph mode (the prod default).
- **Capture-size note:** with spec-decode the per-step decode query length becomes
  `N+1` (>1). If the MSA/full-attention backend's CUDA-graph support is below
  `UNIFORM_BATCH`, vLLM **auto-downgrades** decode cudagraph FULL→PIECEWISE (or NONE) with a
  warning rather than crashing (`compilation.py:1398-1416`). Expect a possible
  `cudagraph_mode` downgrade log line; that's normal and still fast. PIECEWISE keeps graphs
  around the marlin MoE.
- **marlin fused NVFP4 MoE:** unaffected — ngram doesn't change the MoE path; the target still
  runs the same fused kernel, just over `N+1` query positions per decode step.
- **TP4:** ngram drafting is replicated/cheap; no draft TP to coordinate. The only TP caveat in
  this build is for **sequence parallelism** (`enable_sp`) combined with spec-decode — capture
  sizes must be a common multiple of `N+1` and `tp_size`, else it raises. You are **not** using
  `enable_sp`, so this won't trigger. (`adjust_cudagraph_sizes_for_spec_decode`.)
- **MSA custom attention:** the MTP drafter (#2) reuses the same sparse-MSA decoder layer
  (`mtp.py` builds `MiniMaxM3DecoderLayer(..., force_sparse_attn=True)`), so MTP would inherit
  MSA behavior — relevant only once weights exist. ngram is attention-agnostic.

---

## 4. Expected ballpark + failure signatures (ngram, bs=1, reasoning)

- **Acceptance / speedup (rough):**
  - Code / structured / heavily self-quoting reasoning: mean accepted length ~1.6–2.5
    → ~**1.3–1.8×** decode tok/s (i.e. ~90 → ~120–160 tok/s).
  - Mixed reasoning with novel prose: ~**1.05–1.3×**.
  - Pure novel prose / first-pass creative: ~**1.0×** (ngram finds nothing; spec adds a tiny
    verification overhead — should be within noise).
  ngram is **lossless** (rejection sampling); output distribution is unchanged. So the only
  risk is "no speedup", never "wrong answer".
- **Where to read acceptance:** `vllm/v1/spec_decode/metrics.py` exposes spec-decode metrics;
  look in `docker logs` / `/metrics` for `acceptance` / `num_accepted` / mean accepted length.
  If `num_accepted_tokens_per_pos` ≈ 0, ngram isn't matching → lower `prompt_lookup_min` or
  accept that the workload isn't repetitive.
- **Failure signatures to watch for in the trial:**
  - `cudagraph_mode ... not supported with spec-decode ... setting cudagraph_mode=PIECEWISE/NONE`
    → benign auto-downgrade, keep going.
  - A hard `ValueError` about "multiple of (num_speculative_tokens+1) and tensor_parallel_size"
    → only if SP is on; you're not using it.
  - Decode tok/s **lower** than baseline on prose → expected for non-repetitive text; reduce N
    or just accept ngram is workload-dependent. There is no correctness failure mode.
  - If you ever point ngram-style config at a "model" path it will try draft_model — make sure
    `method` is exactly `"ngram"`.

---

## 5. If you want the big win (MTP, ~2–3×) later — what it takes

1. Get MTP weights for MiniMax-M3: either quantize the **7 MTP modules** that exist in
   `MiniMaxAI/MiniMax-M3` (`num_mtp_modules=7`) to NVFP4 with the same ModelOpt pipeline used
   for the base export, naming them `model.mtp.layers.{0..6}.*` (the loader strips `.mtp.`),
   or obtain a standalone `*/MiniMax-M3-MTP` checkpoint and pass it as `model`.
2. Drop them into the served checkpoint (bundled, `language_model.*.mtp.layers.*`) **or** keep
   separate and pass `"model":"<path>"`.
3. Launch with `{"method":"minimax_m3_mtp","num_speculative_tokens":N}` (N ≤ usable modules).
   Same CUDA-graph caveats as §3 (expect possible FULL→PIECEWISE decode downgrade because the
   per-step query length becomes N+1 over the MSA backend).
This is an offline quantization task, not a launch-flag change — out of scope for the serial
GPU trial but flagged as the highest-leverage follow-up.

---

## 6. What to run

`launch_marlin_ngram.sh` (next to this file) = `launch_marlin.sh` with **only** the ngram
`--speculative-config` delta added. Run it in a serial slot (it will `docker rm -f` and replace
the container — do NOT run while you need the prod container up). Test prompts in
`spec_test_prompts.md`.

### Exact flag delta vs `launch_marlin.sh`
```
+  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}'
```
Everything else (TP4, block-size 128, max-model-len 65536, gpu-mem 0.95, graph mode, parsers,
the three mounted patch files) is unchanged.
