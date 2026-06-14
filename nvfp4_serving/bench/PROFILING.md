# PROFILING.md — per-component breakdown of the NVFP4 MoE forward pass

Goal: attribute the time of **one decode forward pass** of MiniMax-M3-NVFP4 to its
components so every optimization in the marathon can be measured against a baseline.
The suspected dominant cost is **host-sync stalls** in the per-expert Python
`group_gemm` loop and the `fp4_quantize` of activations (enforce-eager => no CUDA
graph to hide dispatch + sync latency).

> **GPU COORDINATION.** The profiler (`profile_moe.py`) allocates all 4 GPUs and
> therefore **must NOT run while the vLLM server is up** (the server holds
> `gpu-memory-utilization 0.95`). It is delivered **ready-to-run**; run it only in a
> dedicated GPU slot when `nvidia-smi` shows the GPUs free. It self-aborts if it sees
> >5 GiB used on any GPU. The benchmark *client* (`bench_client.py`) needs no GPU and
> is what produced `BASELINE.md`.

---

## 0. The path under the microscope

The NVFP4 MoE decode forward (per MoE layer, per token, batch size 1) is:

| # | Component | What runs | Suspected cost |
|---|-----------|-----------|----------------|
| 1 | Router / gate | small GEMM + top-k expert selection | cheap, but top-k may `.cpu()`/`nonzero` -> **host sync** |
| 2 | Token sort / scatter | sort tokens by expert, build `m_indptr` | index math; potential D2H copies |
| 3 | **fp4_quantize (activations)** | `scaled_fp4_quant` of x -> (fp4, scale) | **host sync** to read global amax / scale (suspected #1 cost) |
| 4 | **per-expert group_gemm (gate_up)** | **Python loop** over experts calling cutlass FP4 mm | **loop dispatch overhead + per-iter sync** (suspected #2 cost) |
| 5 | swigluoai activation | clamp(±7)/sigmoid(1.702·g)/mul elementwise | small, memory-bound |
| 6 | fp4_quantize (intermediate) | quantize swiglu output for down GEMM | another **host sync** |
| 7 | per-expert group_gemm (down) | Python loop, FP4 mm -> hidden | loop dispatch overhead |
| 8 | combine / scatter-add | topk-weighted sum back to token order | cheap |
| 9 | **all-reduce (TP4)** | NCCL across 4 GPUs after down-proj | network-bound; serializes the 4 ranks |
| A | Attention / MSA | flash-attn (bf16 KV, page-128) | usually not the bottleneck at bs1 |
| B | Sampling | argmax (temp=0) | tiny, but `.item()` on output token == **host sync** |

A **host-sync stall** shows up in a trace as: CPU thread busy (or blocked in
`cudaStreamSynchronize`) while the GPU timeline has an **idle gap**. At bs1 with
enforce-eager, each sync is ~tens–hundreds of µs and they multiply by
(num_layers × num_experts-touched × 2 GEMMs), so they dominate.

---

## 1. torch-profiler breakdown (op-level attribution)

Ready-to-run snippet: **`profile_moe.py`** (in this dir). It:

1. refuses to run if a GPU is busy (server up),
2. builds vLLM offline with the **exact live config** (TP4, block-size 128,
   bf16 KV, max-model-len 65536, **enforce-eager**),
3. warms up, then wraps a single bs=1 decode step in `torch.profiler` (CPU+CUDA),
4. prints **top ops by self-CUDA time** and **by self-CPU time**, exports a
   Chrome trace, and flags suspected host-sync/quant ops by name.

Run (free GPU slot only):

```bash
python3 profile_moe.py --tool torch --warmup 8 --decode-steps 1
# -> prints two sorted op tables + a "SUSPECTED HOST-SYNC / QUANT OPS" section
# -> writes moe_decode_trace.json (open in chrome://tracing or ui.perfetto.dev)
```

How to read it:
- **Self-CUDA table** = where GPU time actually goes. Expect `cutlass_scaled_fp4_mm`
  / `group_gemm` (the two per-expert GEMM loops) and attention to top it *if* the
  GPU is the bottleneck.
- **Self-CPU table** = host time. If `cudaStreamSynchronize`, `aten::item`,
  `aten::_local_scalar_dense`, `aten::copy_`/`to_copy` (D2H), or `scaled_fp4_quant`
  have **large CPU time but small CUDA time**, the path is **sync-bound**, not
  compute-bound — that is the marathon's primary target.
- In the Chrome/Perfetto trace, measure **GPU-idle gaps** between kernels. Summed
  idle / total wall = the fraction recoverable by removing the Python loop
  (batch the experts into one grouped GEMM) and the host syncs (keep scales on GPU,
  capture CUDA graphs).

### Per-component timing without the profiler (cheap cross-check)
Wrap regions with NVTX + CUDA events to get wall ms per component, e.g. inside the
MoE `apply()` override:
```python
import torch
ev = lambda: torch.cuda.Event(enable_timing=True)
t = {k: (ev(), ev()) for k in ("quant1","gemm1","swiglu","quant2","gemm2","allreduce")}
t["quant1"][0].record(); a, a_scale = fp4_quantize(x, input_scale); t["quant1"][1].record()
# ... record each region ...
torch.cuda.synchronize()
for k,(s,e) in t.items(): print(k, s.elapsed_time(e), "ms")
```
This gives a clean millisecond split of components 3–9 above and is the quickest
before/after gauge for each optimization.

---

## 2. nsys breakdown (kernel + CUDA-API / host-sync timeline)

For the definitive host-sync picture (which torch self-CPU only approximates),
use Nsight Systems. `profile_moe.py --tool nsys` prints the exact wrapper:

```bash
python3 profile_moe.py --tool nsys
# prints, ready to copy/paste:
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o moe_decode_profile \
  python3 profile_moe.py --tool torch --nsys-range --decode-steps 1
```
(`--nsys-range` marks a `cudaProfilerStart/Stop` window around just the profiled
decode step, so the report is not polluted by engine startup.)

Then mine it:
```bash
nsys stats --report cuda_api_sum      moe_decode_profile.nsys-rep   # host-side API
nsys stats --report cuda_gpu_kern_sum moe_decode_profile.nsys-rep   # GPU kernels
```
- **`cuda_api_sum`**: high total time in `cudaStreamSynchronize`,
  `cudaMemcpyAsync` (DtoH), or `cudaDeviceSynchronize` == the host-sync stalls.
  This is the number to drive to ~0.
- **`cuda_gpu_kern_sum`**: the FP4 GEMM kernels (`group_gemm` / cutlass) and
  attention kernels — the irreducible compute floor once syncs are gone.
- In the timeline view, look for the **sawtooth**: many tiny GEMM kernels (one per
  expert) separated by host gaps == the Python per-expert loop. Collapsing it into
  a single grouped GEMM removes both the launch overhead and the gaps.

---

## 3. What "good" looks like (optimization targets)

| Symptom in profile | Root cause | Fix the marathon will apply |
|--------------------|-----------|------------------------------|
| GPU idle gaps, big `cudaStreamSynchronize` CPU time | per-expert Python loop + per-iter sync | one grouped/batched FP4 GEMM over all experts |
| `scaled_fp4_quant` high CPU, small CUDA | scale/amax read back to host | keep scales on-device; fuse quant into GEMM epilogue |
| `aten::item` / `_local_scalar_dense` calls | `.item()`/`.cpu()` in routing/sampling | vectorize; avoid host reads in the hot loop |
| no CUDA graph, per-op dispatch | `enforce-eager` | drop enforce-eager once numerics are correct -> capture graphs |
| serialized NCCL `all_reduce` | TP4 collective | overlap with compute / reduce TP if memory allows |

Re-run `bench_client.py` after each change; `BASELINE.md` is the scoreboard.
