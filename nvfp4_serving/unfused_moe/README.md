# Un-fused swigluoai NVFP4 MoE for SM120 (B2a) — validated

Standalone, numerically-validated implementation of the un-fused NVFP4 MoE that
makes MiniMax-M3-NVFP4 generate coherently on SM120 (RTX PRO 6000). Replaces the
broken forced-marlin path and the fused `b12x_fused_moe` (which only does plain
SiLU — wrong for M3's `swigluoai`).

Method: per routed expert, run a single-group
`flashinfer.gemm.group_gemm_nvfp4_nt_groupwise` for `gate_up`, apply `swigluoai`
(torch, contiguous halves), then a second single-group group_gemm for `down`.

## Validated numbers (REAL checkpoint experts, random tokens)

`validate.py` (layer 10, E=16, T=96):

```
(1) nvfp4-unfused(swigluoai) vs bf16-ref(swigluoai): rel RMS = 0.1345   <- noise floor
(2) nvfp4-unfused(SILU)      vs bf16-ref(swigluoai): rel RMS = 0.3263   <- wrong act
(4) nvfp4-unfused(swigluoai) vs nvfp4-unfused(SILU) : rel RMS = 0.3798  <- swigluoai applied
    bf16-ref(swigluoai)      vs bf16-ref(SILU)      : rel RMS = 0.3593  <- activation gap
```

`validate_multi.py` (robustness across layers/seeds/expert-counts):

```
L=10 E=16 T= 96: (1)=0.1345  (2)=0.3263
L=20 E=32 T=128: (1)=0.1320  (2)=0.3276
L= 5 E= 8 T= 64: (1)=0.1331  (2)=0.3307
L=40 E=64 T=200: (1)=0.1516  (2)=0.3583
```

`(1)` sits at the NVFP4 quant noise floor (swiglu_moe_ref.py reported 0.146),
confirming correctness. `(2)`/`(4)` confirm swigluoai is genuinely applied and
differs from the (wrong) SiLU path the fused kernel uses.

## Verified API / scale-layout contract

`group_gemm_nvfp4_nt_groupwise(a, b, a_scale, b_scale, m_indptr, alpha, ...)`:

- `a` (activation): `flashinfer.fp4_quantize(x, global_scale=a_gs,
  is_sf_swizzled_layout=True)` per expert; `a_scale` = the returned swizzled
  scale flattened; `a_gs = (6*448)/amax(|x|)`. `m` padded to a multiple of 4.
- `b` (weight): on-disk LINEAR E4M3 `weight_scale` `[N, K//16]` ->
  `nvfp4_block_scale_interleave` (m padded to 128) ->
  `convert_sf_to_mma_layout(m=N_pad128, k=K, num_groups=1)`.
- `alpha = (1/a_gs) * weight_scale_2`  (weight_scale_2 = checkpoint fp32 global).
- Run ONE single-group call PER expert (the doc-endorsed B2a per-expert loop).

## Important caveat (multi-group kernel quirk)

The *multi-group* form of `group_gemm_nvfp4_nt_groupwise` (one call, all experts,
real `m_indptr`) was found to miscompute the activation scale-factor (SFA) for
groups index >= 2 in this flashinfer build (0.6.12): groups 0,1 are exact
(~0.095) but later groups diverge regardless of a_scale layout tried (flat
concat / per-group mma / whole-buffer swizzle, segments padded to 4 or 128). The
per-expert single-group loop sidesteps this entirely and is numerically exact. A
fast batched path needs that kernel's grouped-SFA offset fixed (or use the fused
moe_static/dynamic kernels with a swigluoai epilogue — out of scope here).

The `b12x_fused_moe(silu)` comparison in `validate.py` is NOT apples-to-apples:
b12x needs vLLM's `process_weights_after_loading` bake-in (block scales * g_alpha,
alpha->1, fc2_input_scale). Driven on raw checkpoint weights it diverges wildly
(~254); that path is the broken one this work replaces.

## Run (inside the image)

```
sudo docker run --rm --gpus all -v <dir>:/work -v /home/kacper/models:/models:ro \
  --entrypoint python3 vllm/vllm-openai:minimax-m3 /work/validate.py
```

## Files
- `unfused_moe.py` — the standalone `unfused_swigluoai_nvfp4_moe` + helpers.
- `validate.py` — end-to-end validation vs bf16 reference (4 comparisons).
- `validate_multi.py` — robustness sweep over layers/seeds/expert counts.
- `probe_real_w.py` — single real-expert GEMM vs dequant-bf16 (0.095 noise floor).
