# cutlass_fp4_moe — W4A4 NVFP4 swigluoai MoE for MiniMax-M3 on SM120

A genuine **W4A4 FP4-compute** MoE path for MiniMax-M3-NVFP4 (vs marlin, which
dequantizes NVFP4→bf16 and matmuls in bf16). Activations are quantized to FP4
and fed to flashinfer's cutlass `group_gemm_nvfp4_nt_groupwise` grouped GEMM.

**Status: correct + coherent, but slower than marlin in every regime that
serves. Marlin is restored as prod.** See `RESULTS.md` for the full verdict,
numbers, and the graph-capture OOM analysis.

## Files

| file | role |
|---|---|
| `batched_graphsafe_moe.py` | static `[E,C]` routing + ONE multi-group FP4 GEMM/projection + SFA scatter fix. Graph-capturable in isolation; bit-exact vs the per-expert loop. |
| `batched_dynamic_moe.py` | eager correctness path: grouped FP4 GEMM over only ACTIVE experts (no capacity drop). Serves coherently. |
| `modelopt.py` | `ModelOptNvFp4FusedMoE` override gated by `VLLM_M3_CUTLASS_FP4_MOE=1`. |
| `model.py`, `nvfp4_unfused_moe.py` | shared loader + reference (copied from `m3_patch_unfused`). |
| `launch_cutlass.sh` | launch (graph or eager); drops force-marlin, mounts FP4 modules. |
| `test_batched_graphsafe.py` | numerical proof (vs loop = 0.0, vs bf16 at noise floor). |
| `coherence_check.sh` | live coherence probe (Warsaw / arithmetic / code-gen). |
| `RESULTS.md` | head-to-head verdict + numbers. |

## Run

```bash
# eager (serves correctly; dynamic path, no capacity drop):
VLLM_M3_CUTLASS_GPU_MEM=0.95 VLLM_M3_UNFUSED_SELFCHECK=1 ./launch_cutlass.sh eager
./coherence_check.sh

# graph (OOMs on this box during the 102-graph capture — see RESULTS.md):
VLLM_M3_CUTLASS_FP4_CAP=16 ./launch_cutlass.sh graph
```

## Env flags

* `VLLM_M3_CUTLASS_FP4_MOE=1` — enable this path (else falls back to per-expert
  unfused loop / marlin).
* `VLLM_M3_CUTLASS_FP4_DYNAMIC=1` — use the dynamic (eager-only, no-drop) path;
  auto-on in `eager` launch mode.
* `VLLM_M3_CUTLASS_FP4_CAP=<int>` — static per-expert capacity for the graph path.
* `VLLM_M3_UNFUSED_SELFCHECK=1` — one-shot real-token numeric self-check (rank0).
* `VLLM_M3_CUTLASS_GPU_MEM` / `VLLM_M3_CUTLASS_GRAPH_ESTIMATE` — memory tuning.

## Validate the kernel (GPUs free)

```bash
sudo docker run --rm --runtime=nvidia --gpus all \
  -v /home/kacper/msa-120/nvfp4_serving:/work:ro --entrypoint python3 \
  vllm/vllm-openai:minimax-m3 /work/cutlass_fp4_moe/test_batched_graphsafe.py
# -> vs per-expert loop = 0.0 (bit-exact); vs bf16-dequant at the noise floor.
```
