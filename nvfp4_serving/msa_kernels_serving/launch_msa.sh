#!/usr/bin/env bash
# Serve MiniMax-M3-NVFP4 with marlin MoE + OUR SM120 MSA kernels (Phase 2).
#
# What runs on OUR code:
#   * DECODE main-attention (block-sparse paged flash-decoding, page-128,
#     ldmatrix W4=3) -- the bs1 interactive hot path -- via
#     forward_sparse_decode_serving (graph-capture safe).
#   * PREFILL indexer TOP-K -- our SM120 topk_select_varlen (per-query num_valid,
#     SET-EXACT vs Triton, 2-3x faster on the many-query prefill rows). Score
#     stays Triton (byte-identical).
# Still Triton (documented gaps -- see SERVING_INTEGRATION.md Phase 2):
#   * DECODE indexer top-k (bs1 launch-bound -> fused Triton is faster there).
#   * Indexer SCORE (needs a score-only paged pybind -- blocker D/E).
#   * PREFILL main-attention (our paged kernel is page-64 / single-seq, not the
#     fused page-128 cu_seqlens-batched cache).
#
# Mechanism: mount our overlay package + a sitecustomize.py startup hook that
# (a) JIT-builds the decode + topk kernels at startup and (b) monkeypatches
# select_main_impl_cls -> MiniMaxM3SparseSm120Impl and select_indexer_impl_cls
# -> MiniMaxM3IndexerSm120Impl on capability-family 120 + bf16 KV + topk16. The
# MoE/quant overlay is identical to launch_marlin.sh so MoE perf is unchanged.
#
# Env knobs (read once at startup):
#   SM120_INDEXER_DECODE=ours|triton  (default triton -- see Phase 2 verdict)
#   SM120_INDEXER_PREFILL=ours|triton (default ours)
#
# To restore the marlin baseline instead:  /home/kacper/launch_marlin.sh graph
set -euo pipefail

IMAGE=vllm/vllm-openai:minimax-m3
MODELS=/home/kacper/models
OVERLAY=/home/kacper/msa-120/nvfp4_serving/msa_kernels_serving
PATCH_FIXED=/home/kacper/m3_patch_unfused
PATCH_MARLIN=/home/kacper/m3_patch
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

ENFORCE_EAGER="${1:-graph}"
EAGER_FLAG=""; [ "$ENFORCE_EAGER" = "eager" ] && EAGER_FLAG="--enforce-eager"

sudo docker rm -f minimax-m3-nvfp4 2>/dev/null || true

sudo docker run -d --name minimax-m3-nvfp4 --runtime=nvidia --gpus all \
  --network host --ipc host --shm-size 16g \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_FORCE_SWIGLU_CLAMP_LIMIT=7.0 \
  -e VLLM_FORCE_SWIGLU_ALPHA=1.702 \
  -e VLLM_FORCE_SWIGLU_BETA=1.0 \
  -e PYTHONPATH=/opt/sm120/msa_kernels_serving:/opt/sm120 \
  -v "${MODELS}:/models:ro" \
  -v "${PATCH_FIXED}/model.py:${VLLM}/models/minimax_m3/nvidia/model.py:ro" \
  -v "${PATCH_MARLIN}/modelopt.py:${VLLM}/model_executor/layers/quantization/modelopt.py:ro" \
  -v "${PATCH_MARLIN}/marlin_moe.py:${VLLM}/model_executor/layers/fused_moe/experts/marlin_moe.py:ro" \
  -v "${OVERLAY}:/opt/sm120/msa_kernels_serving:ro" \
  "${IMAGE}" \
  --model /models/MiniMax-M3-NVFP4 \
  --tensor-parallel-size 4 \
  --block-size 128 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.977 \
  ${EAGER_FLAG} \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice

echo "MSA-decode launched (mode=${ENFORCE_EAGER})."
echo "Watch for '[sm120-msa] ... MiniMaxM3SparseSm120Impl' + a FULL decode"
echo "cudagraph capture with no StreamCaptureInvalidated:"
echo "  sudo docker logs -f minimax-m3-nvfp4"
