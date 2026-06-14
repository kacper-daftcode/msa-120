#!/usr/bin/env bash
# W4A4 BATCHED-CUTLASS NVFP4 swigluoai MoE — MiniMax-M3-NVFP4 on 4x RTX PRO 6000.
#
# REAL FP4 COMPUTE path: one multi-group flashinfer group_gemm_nvfp4_nt_groupwise
# per projection (gate_up + down) over a STATIC [E, C] routing table, so the MoE
# is CUDA-graph capturable (graph mode by default).  Activations are quantized to
# FP4 and the matmul runs in FP4 (vs marlin, which dequantizes weights to bf16).
#
# vs launch_marlin.sh this DROPS:
#   - VLLM_TEST_FORCE_FP8_MARLIN   (marlin no longer forced)
#   - VLLM_FORCE_SWIGLU_*          (the custom MoE applies swigluoai itself,
#                                   reading layer.swiglu_alpha/limit)
#   - the marlin_moe.py mount
# and ADDS:
#   - modelopt.py (cutlass override) + batched_graphsafe_moe.py + nvfp4_unfused_moe.py
#   - VLLM_M3_CUTLASS_FP4_MOE=1     (gate the batched W4A4 path)
#   - VLLM_M3_CUTLASS_FP4_CAP=<int> (optional static per-expert capacity)
#   - VLLM_M3_UNFUSED_SELFCHECK=1   (optional one-shot real-token numeric check)
set -euo pipefail

IMAGE=vllm/vllm-openai:minimax-m3
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL_PY=/home/kacper/m3_patch_unfused/model.py   # path-agnostic loader (shared)
MODELS=/home/kacper/models
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
SITE=/usr/local/lib/python3.12/dist-packages

ENFORCE_EAGER="${1:-graph}"   # "graph" (default, CUDA-graph capture) or "eager" (debug)
CAP="${VLLM_M3_CUTLASS_FP4_CAP:-}"
SELFCHECK="${VLLM_M3_UNFUSED_SELFCHECK:-0}"
# The batched [E*C, K] activation buffers are captured into per-size cudagraph
# private pools; leave headroom below KV so capture doesn't OOM.  Lower than
# marlin's 0.977 because the static all-E buffers are large.
GPU_MEM="${VLLM_M3_CUTLASS_GPU_MEM:-0.88}"

sudo docker rm -f minimax-m3-nvfp4 2>/dev/null || true

EAGER_FLAG="--enforce-eager"
DYNAMIC="${VLLM_M3_CUTLASS_FP4_DYNAMIC:-0}"
[ "$ENFORCE_EAGER" = "graph" ] && EAGER_FLAG=""
# In eager mode default to the DYNAMIC batched path (no capacity drop -> correct
# for prefill); the static [E,C] path is only needed for graph capture.
[ "$ENFORCE_EAGER" = "eager" ] && DYNAMIC="${VLLM_M3_CUTLASS_FP4_DYNAMIC:-1}"

sudo docker run -d --name minimax-m3-nvfp4 --runtime=nvidia --gpus all \
  --network host --ipc host --shm-size 16g \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_M3_CUTLASS_FP4_MOE=1 \
  -e VLLM_M3_CUTLASS_FP4_DYNAMIC="${DYNAMIC}" \
  -e VLLM_M3_CUTLASS_FP4_CAP="${CAP}" \
  -e VLLM_M3_UNFUSED_SELFCHECK="${SELFCHECK}" \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_M3_CUTLASS_GRAPH_ESTIMATE:-1}" \
  -v "${MODELS}:/models:ro" \
  -v "${MODEL_PY}:${VLLM}/models/minimax_m3/nvidia/model.py:ro" \
  -v "${HERE}/modelopt.py:${VLLM}/model_executor/layers/quantization/modelopt.py:ro" \
  -v "${HERE}/batched_graphsafe_moe.py:${SITE}/batched_graphsafe_moe.py:ro" \
  -v "${HERE}/batched_dynamic_moe.py:${SITE}/batched_dynamic_moe.py:ro" \
  -v "${HERE}/nvfp4_unfused_moe.py:${SITE}/nvfp4_unfused_moe.py:ro" \
  "${IMAGE}" \
  --model /models/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 65536 \
  --gpu-memory-utilization "${GPU_MEM}" \
  ${EAGER_FLAG} \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice

echo "cutlass W4A4 launched (mode=${ENFORCE_EAGER}, cap=${CAP:-auto}); tail logs:"
echo "  sudo docker logs -f minimax-m3-nvfp4"
