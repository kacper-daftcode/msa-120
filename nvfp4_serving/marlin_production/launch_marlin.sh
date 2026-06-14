#!/usr/bin/env bash
# MARLIN NVFP4 re-test for MiniMax-M3-NVFP4 on 4x RTX PRO 6000 (SM120).
#
# Hypothesis: the old "marlin numerics broken -> gibberish" diagnosis was WRONG.
# The real cause was the expert-name bug (w1 vs gate_proj) that zeroed all 128
# experts on EVERY path including marlin. With the FIXED model.py, marlin's
# native fused NVFP4 MoE kernel may now be both correct AND fast (and CUDA-graph
# capturable, unlike the un-fused per-expert loop).
#
# Config = the known-working marlin overlay set, with ONLY model.py swapped to
# the fixed version (gate_proj/down_proj/up_proj expert names).
set -euo pipefail

IMAGE=vllm/vllm-openai:minimax-m3
PATCH_FIXED="$(cd "$(dirname "$0")/patches" && pwd)"   # fixed model.py lives here
PATCH_MARLIN="$PATCH_FIXED"          # marlin modelopt.py + marlin_moe.py
MODELS=/home/kacper/models
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

ENFORCE_EAGER="${1:-graph}"   # "graph" (default, FAST: CUDA-graph capture, ~90 tok/s decode@bs1)
                              # or "eager" (debug, ~19 tok/s) — marlin captures cleanly, graph is the prod default

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
  ${EAGER_FLAG} \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice

echo "marlin launched (mode=${ENFORCE_EAGER}); tail logs with: sudo docker logs -f minimax-m3-nvfp4"
