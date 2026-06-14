#!/usr/bin/env bash
# MARLIN NVFP4 + NGRAM SPECULATIVE DECODING trial for MiniMax-M3-NVFP4
# on 4x RTX PRO 6000 (SM120). SERIAL GPU SLOT ONLY.
#
# This is launch_marlin.sh with ONE delta: a zero-dependency ngram
# (prompt-lookup) --speculative-config. ngram needs no draft model / no extra
# weights and is lossless (rejection sampling). It is the only spec-decode
# method that works against the current NVFP4 checkpoint, because that
# checkpoint ships NO MTP weights (see PLAN.md). It composes with CUDA graphs
# (graph mode is kept); vLLM may auto-downgrade decode cudagraph FULL->PIECEWISE
# for the spec path — that is benign, logged as a warning, still fast.
#
# Uses GPUs exclusively (TP4, all GPUs). The prod `minimax-m3-nvfp4` container
# must be stopped first or this will contend for VRAM. By default this launches
# a SEPARATE container name so it won't silently delete prod; you must free the
# GPUs yourself before running.
set -euo pipefail

IMAGE=vllm/vllm-openai:minimax-m3
PATCH_FIXED=/home/kacper/m3_patch_unfused   # fixed model.py lives here
PATCH_MARLIN=/home/kacper/m3_patch          # marlin modelopt.py + marlin_moe.py
MODELS=/home/kacper/models
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

ENFORCE_EAGER="${1:-graph}"   # "graph" (default, FAST) or "eager" (debug)

# ngram tunables (override via env). Defaults tuned for a reasoning+code workload:
# small min-window catches more self-quoted spans; cap N at 4 so a miss is cheap.
NSPEC="${NSPEC:-4}"            # num_speculative_tokens
PLMIN="${PLMIN:-2}"           # prompt_lookup_min
PLMAX="${PLMAX:-4}"           # prompt_lookup_max
NAME="${NAME:-minimax-m3-nvfp4-ngram}"

SPEC_CONFIG="{\"method\":\"ngram\",\"num_speculative_tokens\":${NSPEC},\"prompt_lookup_max\":${PLMAX},\"prompt_lookup_min\":${PLMIN}}"

# Safety: refuse to run if the production container is still up (would contend
# for all 4 GPUs). Stop it yourself first, then re-run.
if sudo docker ps --format '{{.Names}}' | grep -qx 'minimax-m3-nvfp4'; then
  echo "ERROR: prod container 'minimax-m3-nvfp4' is running and holds all 4 GPUs." >&2
  echo "       Stop it first (serial GPU slot): sudo docker stop minimax-m3-nvfp4" >&2
  exit 1
fi

sudo docker rm -f "${NAME}" 2>/dev/null || true

EAGER_FLAG="--enforce-eager"
[ "$ENFORCE_EAGER" = "graph" ] && EAGER_FLAG=""

echo "Launching ${NAME} with spec-decode: ${SPEC_CONFIG} (mode=${ENFORCE_EAGER})"

sudo docker run -d --name "${NAME}" --runtime=nvidia --gpus all \
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
  --speculative-config "${SPEC_CONFIG}" \
  --tool-call-parser minimax_m3 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice

echo "ngram-spec launched as ${NAME} (mode=${ENFORCE_EAGER})."
echo "Tail logs:    sudo docker logs -f ${NAME}"
echo "Watch for:    a 'cudagraph_mode ... spec-decode ... PIECEWISE' warning (benign),"
echo "              and spec-decode acceptance metrics (mean accepted length > 1 = wins)."
echo "When done, restore prod:  /home/kacper/launch_marlin.sh"
