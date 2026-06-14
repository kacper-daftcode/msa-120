# NVFP4 MiniMax-M3 on SM120 via vLLM — serving recipe + status

## STATUS: mechanically serving, output is GIBBERISH (MoE numerics gap)
The full pipeline loads + serves on 4x RTX PRO 6000. Generation is incoherent
because the forced marlin-NVFP4 + swigluoai path is numerically wrong. The
verified-correct fix is the un-fused B2a MoE method (docs/B2a_UNFUSED_NVFP4_MOE.md,
python/fmha_sm100/swiglu_moe_ref.py — proves plain/wrong-swigluoai = rel RMS 1.72).

## Blockers solved (all 6) to reach mechanical serving
1. config.json text_config.layer_types ('minimax_m3_sparse' rejected) -> removed.
2. config.json text_config.hidden_act 'silu' -> 'swigluoai' (M3 requires it).
3. weight key remap (patches/model.py hf_to_vllm_mapper): model.language_model.->
   language_model.model., .mlp.experts/shared_experts/gate -> .block_sparse_moe.*,
   model.vision_tower.layers -> vision_tower.vision_model.encoder.layers,
   embeddings.proj -> patch_embedding, multi_modal_projector merge_linear split,
   lm_head -> language_model.lm_head.
4. NVFP4 MoE backend: no native SM120 swigluoai MoE. Forced marlin
   (VLLM_TEST_FORCE_FP8_MARLIN=1) + plumb swiglu clamp/alpha/beta which fail to
   reach the quant config: patches/modelopt.py (self.moe fallback) +
   patches/marlin_moe.py (env fallback) + env VLLM_FORCE_SWIGLU_CLAMP_LIMIT=7.0
   ALPHA=1.702 BETA=1.0.  <-- THIS PATH IS NUMERICALLY BROKEN (gibberish).
5. config.json quantization_config.kv_cache_scheme (fp8) -> removed. fp8 KV made
   the platform prefer FlashInfer; page-128 FlashInfer needs trtllm-gen (SM100).
   bf16 KV -> FLASH_ATTN, page-128 OK.
6. block-size MUST be 128 (indexer requires it; block-64 -> "no common block size").

## Launch (mechanically works, gibberish output)
docker run -d --name minimax-m3-nvfp4 --runtime=nvidia --gpus all --network host \
  --ipc host --shm-size 16g \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_FORCE_SWIGLU_CLAMP_LIMIT=7.0 -e VLLM_FORCE_SWIGLU_ALPHA=1.702 -e VLLM_FORCE_SWIGLU_BETA=1.0 \
  -v /models:/models:ro \
  -v patches/model.py:.../vllm/models/minimax_m3/nvidia/model.py:ro \
  -v patches/modelopt.py:.../vllm/model_executor/layers/quantization/modelopt.py:ro \
  -v patches/marlin_moe.py:.../vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:ro \
  vllm/vllm-openai:minimax-m3 --model /models/MiniMax-M3-NVFP4 --tensor-parallel-size 4 \
  --block-size 128 --max-model-len 131072 --gpu-memory-utilization 0.92 \
  --tool-call-parser minimax_m3 --reasoning-parser minimax_m3 --enable-auto-tool-choice

## NEXT (to fix gibberish): implement B2a un-fused swigluoai NVFP4 MoE
Replace the broken marlin path with a custom ModelOptNvFp4FusedMoE.apply override:
NVFP4 gate_up GEMM (FlashInferB12xNvFp4LinearKernel, SM120-ok) -> swigluoai
(torch, alpha 1.702/limit 7/contiguous) -> NVFP4 down GEMM. Per swiglu_moe_ref.py.

## DIAGNOSIS UPDATE (confirmed findings)
- **Weight remap is CORRECT**: M3 loader (model.py:828-833) expects exactly
  w1=gate, w2=down, w3=up — our gate_proj->w1/up_proj->w3/down_proj->w2 matches.
  Weights load into the right slots. Gibberish is NOT a remap bug.
- **Gibberish = marlin-NVFP4 MoE numerics**: the forced-marlin NVFP4 + swigluoai
  path (VLLM_TEST_FORCE_FP8_MARLIN, a test path) is numerically wrong on this
  checkpoint despite scale handling. Not production-validated.
- **Fused b12x CANNOT do swigluoai**: flashinfer.fused_moe.b12x_fused_moe only
  supports activation in {"silu","relu2"} — no swigluoai. So the fast fused path
  is out for M3 (which needs the clamped GPT-OSS SwiGLU).

## THE FINISH (exact, fully specified): un-fused swigluoai NVFP4 MoE
Subclass vllm .../fused_moe/experts/flashinfer_b12x_moe.py `FlashInferB12xExperts`
(reuse its process_weights_after_loading: w1_sf_mma/w2_sf_mma via
flashinfer_convert_sf_to_mma_layout, g1_alphas/g2_alphas global scales,
fc2_input_scale). Override the forward to REPLACE the single
`flashinfer_b12x_fused_moe(...)` call with un-fused:
  1. route -> token_selected_experts/scales; sort tokens by expert -> m_indptr.
  2. quantize x -> nvfp4 (a, a_scale) with input scale.
  3. flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(a, w1, a_scale, w1_sf_mma,
     m_indptr, alpha=g1_alphas) -> gate_up [tok, 2I].
  4. swigluoai (torch): gate=g[:I], up=g[I:]; (clamp(up,±7)+1)*clamp(gate,max=7)
     *sigmoid(1.702*gate)  (contiguous halves; per swiglu_moe_ref.py).
  5. quantize intermediate -> nvfp4 with fc2_input_scale.
  6. group_gemm_nvfp4_nt_groupwise(.., w2, .., w2_sf_mma, m_indptr,
     alpha=g2_alphas) -> down [tok, H]; scatter + topk-weighted combine.
Then opt into it (select FLASHINFER_B12X + add to NVFP4_BACKENDS_WITH_CLAMP since
it now applies the clamp). Validate output coherence vs the MXFP4 baseline.
The hard part is the token-sort/m_indptr + intermediate nvfp4 quant that
b12x_fused_moe currently does internally; everything else is reused.
