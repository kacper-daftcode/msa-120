#!/usr/bin/env bash
# ============================================================================
# bench_serving.sh — repeatable serving benchmark for MiniMax-M3-NVFP4 / vLLM
# ----------------------------------------------------------------------------
# Hits the LIVE OpenAI endpoint. Needs NO local GPU. Safe to run while the
# container is serving (this is normal client load; it does NOT touch the GPU
# allocation or restart the container).
#
# Two backends:
#   * client  (default) -> stdlib-only bench_client.py from the host. Always works,
#                           no deps, produces JSON + table covering decode@bs1,
#                           prefill, TTFT/TPOT, and the 1/4/16 concurrency sweep.
#   * vllm                -> runs the in-image `vllm bench serve` (random dataset)
#                           inside the container for each concurrency point.
#                           Preferred for cross-checking; needs the container up.
#
# Usage:
#   ./bench_serving.sh                      # client backend, defaults
#   BACKEND=vllm ./bench_serving.sh         # use vllm bench serve in-container
#   HOST=http://localhost:8000 MODEL=/models/MiniMax-M3-NVFP4 \
#     INPUT_LEN=512 OUTPUT_LEN=128 NUM_PROMPTS=32 CONCURRENCY=1,4,16 \
#     ./bench_serving.sh
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- parameters (env-overridable) ----
HOST="${HOST:-http://localhost:8000}"
MODEL="${MODEL:-/models/MiniMax-M3-NVFP4}"
CONTAINER="${CONTAINER:-minimax-m3-nvfp4}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-32}"
CONCURRENCY="${CONCURRENCY:-1,4,16}"
SWEEP_OUTPUT_LEN="${SWEEP_OUTPUT_LEN:-64}"
BACKEND="${BACKEND:-client}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$HERE/results/baseline_${STAMP}.json}"

mkdir -p "$HERE/results"

echo "[bench_serving] backend=$BACKEND host=$HOST model=$MODEL"
echo "[bench_serving] input_len=$INPUT_LEN output_len=$OUTPUT_LEN num_prompts=$NUM_PROMPTS conc=$CONCURRENCY"

# ---- preflight: endpoint reachable ----
if ! curl -sf --max-time 15 "$HOST/v1/models" >/dev/null; then
  echo "[bench_serving] ERROR: $HOST/v1/models not reachable" >&2
  exit 1
fi

if [[ "$BACKEND" == "vllm" ]]; then
  # --------------------------------------------------------------------------
  # Use vLLM's built-in serving benchmark inside the container (random dataset).
  # One invocation per concurrency point (vllm bench serve uses --max-concurrency
  # to bound in-flight requests, approximating offered load).
  # --------------------------------------------------------------------------
  if ! sudo docker exec "$CONTAINER" which vllm >/dev/null 2>&1; then
    echo "[bench_serving] vllm not found in container; falling back to client backend" >&2
    BACKEND=client
  else
    IFS=',' read -ra CONCS <<< "$CONCURRENCY"
    for C in "${CONCS[@]}"; do
      RESFILE="vllm_serve_conc${C}_${STAMP}.json"
      echo "[bench_serving] vllm bench serve  concurrency=$C ..."
      sudo docker exec "$CONTAINER" vllm bench serve \
        --backend vllm \
        --base-url "http://localhost:8000" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$C" \
        --temperature 0 \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,99 \
        --save-result --result-dir /tmp --result-filename "$RESFILE" \
        || { echo "[bench_serving] vllm bench serve failed at conc=$C" >&2; continue; }
      sudo docker cp "$CONTAINER:/tmp/$RESFILE" "$HERE/results/$RESFILE" 2>/dev/null || true
      echo "[bench_serving] -> results/$RESFILE"
    done
    echo "[bench_serving] vllm backend done. JSON(s) in $HERE/results/"
    exit 0
  fi
fi

# ----------------------------------------------------------------------------
# Default: stdlib-only client (no deps, no GPU).
# ----------------------------------------------------------------------------
python3 "$HERE/bench_client.py" \
  --host "$HOST" \
  --model "$MODEL" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --concurrency "$CONCURRENCY" \
  --sweep-output-len "$SWEEP_OUTPUT_LEN" \
  --out "$OUT"

echo "[bench_serving] done -> $OUT"
