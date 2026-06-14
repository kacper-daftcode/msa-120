#!/usr/bin/env bash
# MARLIN NVFP4 + FP8 KV CACHE for MiniMax-M3-NVFP4 on 4x RTX PRO 6000 (SM120).
#
# Copy of launch_marlin.sh with FP8 KV enabled. Deltas vs launch_marlin.sh:
#   + --kv-cache-dtype fp8         (store K/V as fp8_e4m3 -> ~2x context, less KV BW)
#   + --attention-backend TRITON_ATTN
#
# WHY force Triton (critical on SM120/cc12.0):
#   * Auto-select priority on SM120 is [FLASH_ATTN, FLASHINFER, TRITON_ATTN, ...].
#   * FLASH_ATTN rejects fp8 KV here (needs fa3 + capability-family 90; we're 120).
#   * FLASHINFER would then be picked and CRASH at runtime: block-size 128 forces
#     the trtllm-gen "page>=128" dynamic kernel (Blackwell + GQA + NV artifactory
#     cubins) -> the earlier "FlashInfer page-128 requires trtllm-gen" failure.
#   * TRITON_ATTN supports fp8 KV at block 128 on all compute caps with NO
#     trtllm-gen dependency. The 57 sparse-MSA layers already run the Triton
#     sparse impl on SM120 and natively dequant fp8, so they are unaffected by
#     this flag (it only steers the 3 full-attention layers).
#
# NOTE: the active config.json has NO quantization_config.kv_cache_scheme; fp8 KV
# runs with dynamic / identity (1.0) scales. That is fine to boot. If long-context
# output degrades (e4m3 saturation), the lever is restoring calibrated KV scales.
set -euo pipefail

IMAGE=vllm/vllm-openai:minimax-m3
PATCH_FIXED=/home/kacper/m3_patch_unfused   # fixed model.py lives here
PATCH_MARLIN=/home/kacper/m3_patch          # marlin modelopt.py + marlin_moe.py
MODELS=/home/kacper/models
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

ENFORCE_EAGER="${1:-graph}"   # "graph" (default, FAST CUDA-graph) or "eager" (debug)

sudo docker rm -f minimax-m3-nvfp4 2>/dev/null || true

EAGER_FLAG="--enforce-eager"
[ "$ENFORCE_EAGER" = "graph" ] && EAGER_FLAG=""

sudo docker run -d --name minimax-m3-nvfp4 --runtime=nvidia --gpus all \
  --network host --ipc host --shm-size 16g \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_FORCE_SWIGLU_CLAMP_LIMIT=7.0 \
  -e VLLM_FORCE_SWIGLU_ALPHA=1.702 \
  -e VLLM_FORCE_SWIGLU_BETA=1.0 \
  -v "${MODELS}:/models:ro" \
  -v "${PATCH_FIXED}/model.py:${VLLM}/models/minimax_m3/nvidia/model.py:ro" \
  -v "${PATCH_MARLIN}/modelopt.py:${VLLM}/model_executor/layers/quantization/modelopt.py:ro" \
  -v "${PATCH_MARLIN}/marlin_moe.py:${VLLM}/model_executor/layers/fused_moe/experts/marlin_moe.py:ro" \
  "${IMAGE}" \
  --model /models/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  ${EAGER_FLAG} \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice

echo "marlin+fp8kv launched (mode=${ENFORCE_EAGER}); tail logs with: sudo docker logs -f minimax-m3-nvfp4"
echo "Expect in logs: 'Using TRITON_ATTN backend.' and ~2x GPU KV blocks vs bf16. NO 'page-128 requires trtllm-gen'."
