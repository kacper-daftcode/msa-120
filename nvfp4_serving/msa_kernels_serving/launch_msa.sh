#!/usr/bin/env bash
# Launch MiniMax-M3-NVFP4 with marlin MoE + the SM120 MSA attention path.
#
# STATUS (2026-06-14): NOT YET RUNNABLE end-to-end. The SM120 attend/indexer
# impl swap is blocked on kernel changes A,B,D,E (see RESULTS.md sec 4). The
# kernels build + run on SM120 and topk is op-equivalence-proven, but the attend
# cannot consume the M3 paged cache (page 64 vs 128, fused vs split) yet. This
# script documents the INTENDED launch once those land; today it would fall back
# to Triton (the selector patch gates on the impl actually being importable).
#
# To restore the marlin production baseline instead, run:
#   /home/kacper/launch_marlin.sh graph        # the box's prod default (~91 tok/s bs1)
#
# Mechanism (see selector_patch.md): mount our overlay package into the image and
# import patches.apply() at startup (via a sitecustomize or an --extra-... hook),
# which monkeypatches select_main_impl_cls / select_indexer_impl_cls to return
# the SM120 impls on device-capability-family 120 + bf16 KV. The kernels JIT-build
# on first import via _loader.py (compute_120f; cusparse/cusolver symlink handled
# by prepare_build_env()).
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

# NOTE: the two `-v` lines marked (MSA) mount our overlay + a startup hook. They
# are commented because the impl swap is not yet runnable; uncomment once
# blockers A,B,D,E land and the impls import cleanly.
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
  `# -v "${OVERLAY}:/opt/sm120_msa:ro"   (MSA overlay)` \
  `# -e PYTHONSTARTUP=/opt/sm120_msa/_startup_apply.py   (MSA selector patch)` \
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

echo "launched (mode=${ENFORCE_EAGER}). Watch the startup log for the eager-break"
echo "count: under FULL_AND_PIECEWISE the decode graph is FULL (no per-token break)."
echo "tail: sudo docker logs -f minimax-m3-nvfp4"
