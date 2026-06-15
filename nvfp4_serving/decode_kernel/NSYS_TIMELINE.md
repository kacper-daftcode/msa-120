# NSYS_TIMELINE — MiniMax-M3 bs1 decode on SM120: where the time goes

Nsight Systems (timeline / CUPTI tracing) diagnosis. **Where do we hang at the
timeline level** for (A) our hand-written decode block-sparse attention kernel vs
vLLM's Triton kernel, and (B) a real marlin serving decode step. Timeline numbers
only — kernel durations, inter-kernel GPU-idle gaps, host launch overhead.

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), 4 GPUs.
Image: `vllm/vllm-openai:minimax-m3`. nsys 2025.3.2 (CUPTI).
Raw `.nsys-rep` + `nsys stats` text outputs in `./nsys_stats/`.

---

## 1. Reproducible nsys setup

nsys is **not** in the `vllm/vllm-openai:minimax-m3` image. It **is** on the host
at `/opt/nvidia/nsight-systems/2025.3.2` (also `/usr/local/cuda/bin/nsys`,
version 2025.3.2.474). nsys uses CUPTI timeline tracing and works as root with
`--cap-add SYS_ADMIN` (no perf-counter perms needed, unlike ncu).

Setup: bind-mount the host Nsight dir into the container and symlink onto PATH.
nsys needs a writable scratch dir → set `TMPDIR` to a mounted path.

```bash
# Bench container (CAPTURE A) — GPU0, SYS_ADMIN, host nsys bind-mounted:
sudo docker run -d --name msa-bench --runtime=nvidia --gpus '"device=0"' \
  --network host --ipc host --cap-add SYS_ADMIN \
  -e CUDA_VISIBLE_DEVICES=0 -e TMPDIR=/work/nsys_tmp \
  -v /home/kacper/msa-120/nvfp4_serving:/work -v /home/kacper/models:/models:ro \
  -v /opt/nvidia/nsight-systems/2025.3.2:/opt/nvidia/nsight-systems/2025.3.2:ro \
  --entrypoint sleep vllm/vllm-openai:minimax-m3 infinity
sudo docker exec msa-bench bash -lc \
  'ln -sf /opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys /usr/local/bin/nsys'
```

Sanity check passed: a trivial CUDA program produced a `.nsys-rep` and
`nsys stats --report cuda_gpu_kern_sum` listed its kernel (50× `add(...)`,
~3.2us each). CUPTI tracing confirmed working inside the container with
SYS_ADMIN + `--capture-range cudaProfilerApi`.

Kernel build env: `forward_sparse_decode_p128(use_4warp=2)` built per the
loader recipe (cusparse/cusolver header symlinks +
`-gencode=arch=compute_120f,code=sm_120f`); driver = `decode_kernel/ncu_driver.py`
which warms up (JIT/autotune) **before** `cudaProfilerStart`, then issues N=30
steady-state launches inside the capture range.

---

## 2. CAPTURE A — ours vs Triton, microbench timeline (bs1, seq_kv=16384, M3 shapes)

64 q-heads / 4 kv-heads / dim 128, 16 topk blocks ×128. Each captures 30
steady-state launches inside `cudaProfilerApi` range (autotune excluded).

Command (per impl):
```bash
WHICH={ours|triton} SEQ_KV=16384 W4=2 CHUNKS=32 WARMUP=60..80 NLAUNCH=30 \
  nsys profile -t cuda,nvtx,osrt -o nsys_out/captureA_$WHICH --force-overwrite true \
  --capture-range cudaProfilerApi --capture-range-end stop \
  python3 decode_kernel/ncu_driver.py
```
Reports: `cuda_gpu_kern_sum` (durations), `cuda_gpu_trace` (per-launch start/end →
gaps), `cuda_api_sum` (host API). Raw text: `nsys_stats/captureA_{ours,triton}.txt`.

### 2a. Per-kernel GPU duration (cuda_gpu_kern_sum, median over 30, ns)

| kernel                                   | OURS dur | TRITON dur |
|------------------------------------------|---------:|-----------:|
| K1 partial  (`..._partial_p128_sub64` / `_gqa_sparse_decode_kernel`) | **5184** | **3968** |
| K2 merge    (`..._merge_bf16` / `_merge_topk_attn_out_kernel`)       | **2976** | **1792** |
| **GPU compute K1+K2**                    | **8160** | **5760**   |

### 2b. Timeline gaps (cuda_gpu_trace, steady-state median, ns)

| metric                          | OURS  | TRITON | OURS − TRITON |
|---------------------------------|------:|-------:|--------------:|
| K1 GPU duration                 | 5184  | 3968   | +1216 |
| K2 GPU duration                 | 2976  | 1792   | +1184 |
| **GPU compute (K1+K2)**         | **8160** | **5760** | **+2400** |
| partial→merge gap (GPU idle)    | 2208  | 9536   | **−7328** (ours smaller) |
| merge→next-partial gap          | 5648  | 30688  | −25040 |
| full step period (launch→launch)| 15840 | 46208  | −30368 |

### 2c. Host-side CUDA API (cuda_api_sum, per-launch)

| metric                                  | OURS   | TRITON |
|-----------------------------------------|-------:|-------:|
| launch API median (ns/call)             | 2425 (`cudaLaunchKernel`) | 2235 (`cuLaunchKernelEx`) |
| launch API as % of API time             | 90.7%  | 92.6%  |

### 2d. Where the +2.3us is

The captured window contains exactly 2 kernels per impl, no memcpy/memset
(the partial→merge handoff is a DRAM round-trip through the partial-output
workspace buffer, not an explicit copy).

- **GPU compute delta = +2.40us**, split almost evenly: partial **+1.22us**,
  merge **+1.18us**.
- partial→merge **gap is NOT our problem**: ours' bubble is **2.21us**, *smaller*
  than Triton's 9.54us (Triton's larger gaps are its heavier per-call Python
  launch path; both are per-`run()` host artifacts, not a kernel-design penalty).
- **Host launch overhead is a wash**: 2.43us vs 2.24us per launch (+0.19us).

> **VERDICT A:** Our +2.40us vs Triton is **entirely on-GPU compute** (partial
> +1.22us, merge +1.18us) — *not* inter-kernel gaps (ours' partial→merge bubble
> is actually 7.3us *smaller* than Triton's) and *not* host launch (+0.19us, a
> wash). We hang in **kernel compute**, evenly across both the partial and merge
> kernels.

---

## 3. CAPTURE B — real serving decode step (marlin + Triton-MSA, TP4, graph mode)

Marlin server (`launch_marlin.sh` config) launched with the vllm entrypoint
wrapped under nsys: `--delay 240 --duration 12 --trace-fork-before-exec=true
--cuda-graph-trace=node`. A single bs1 decode request was streamed continuously
so the 12s window captured **steady-state decode at Running:1 req, ~86–91 tok/s**
(server logs confirmed). Analysis is over **one TP rank** (1 of 4 globalPids);
`.nsys-rep` = `nsys_stats/captureB_serve.nsys-rep`; breakdown text =
`nsys_stats/captureB_breakdown.txt`.

Window: 11.95s, **1029 decode tokens → 11.6ms/token wall (86 tok/s)**.
Per-token structure (kernel invocations / token): **~57 MSA sparse-attend layers,
~114 marlin GEMMs, ~121 NCCL all-reduces**.

### 3a. Timeline breakdown (% of GPU kernel work on one rank)

| category                       | % of GPU work | top kernel |
|--------------------------------|--------------:|-----------|
| DENSE GEMM proj/linear (q/k/v/o + lm/embed) | **39.5%** | cutlass wmma bf16 GEMM + cublas gemvx |
| **ALL_REDUCE (PYNCCL)**        | **17.4%**     | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |
| MoE marlin GEMM                | 15.3%         | `marlin_moe_wna16::Marlin<...>` |
| MoE route/permute              | 7.7%          | `moe_align_block_size`, `moe_sum` |
| **MSA indexer/topk**           | **4.3%**      | `_topk_index_partial_kernel`, `_decode_index_score` |
| **MSA sparse-attend**          | **3.4%**      | `_gqa_sparse_decode_kernel` (the Triton kernel from CAPTURE A) |
| DENSE attn (3 dense layers)    | 3.2%          | flashinfer / `flash_fwd_splitkv_combine` |
| activation (swiglu)            | 2.9%          | `act_and_mul_kernel` |
| elementwise / copy / kv-cache  | 2.7%          | unrolled_elementwise, `reshape_and_cache` |
| MoE quant→fp4                  | 2.3%          | `cvt_fp16_to_fp4` |
| norm/rope                      | 1.0%          | `fusedMiniMaxM3QNormRopeKVInsert` |
| sampling/softmax               | 0.4%          | `TopPSamplingFromProbKernel` |

### 3b. Super-categories

| super-category            | % of GPU work |
|---------------------------|--------------:|
| **Dense attn + projections (GEMM)** | **42.6%** |
| **MoE total** (marlin + route + quant + gating) | **25.3%** |
| **All-reduce (TP4 PYNCCL)** | **17.4%** |
| **MSA total** (indexer/topk + sparse-attend) | **7.7%** |
| **GPU idle (inter-kernel gaps)** | **7.8% of wall** (~0.91ms/token) |

> **VERDICT B:** In a real bs1 decode token, **MSA attend is a small slice
> (~3.4% sparse-attend, 7.7% MSA total)**. The token is dominated by **dense
> linear/attn GEMMs (42.6%) + MoE (25.3%) + the TP4 NCCL all-reduce (17.4%)**.
> The GPU is **92% busy** (only 7.8% inter-kernel gap). We do **not** hang in
> MSA attend at serving scale — all-reduce alone (17.4%) is ~5× the entire
> sparse-attend cost.

---

## 4. Putting CAPTURE A in perspective

The ours-vs-Triton **+2.4us per attend** from CAPTURE A applies to the
`_gqa_sparse_decode` + merge slice, which is **3.4% of a real decode token**.
At ~57 MSA layers/token, even a per-layer attend delta is dwarfed by the 42.6%
dense-GEMM + 17.4% all-reduce + 25.3% MoE that set the 11.6ms/token wall. The
attend microbench gap is a real (compute-bound) +2.4us, but it is **not** the
serving bottleneck; dense GEMM, MoE, and the TP4 all-reduce are.

---

## 5. Files

- `nsys_stats/captureA_ours.nsys-rep`, `captureA_triton.nsys-rep` — CAPTURE A traces
- `nsys_stats/captureA_ours.txt`, `captureA_triton.txt` — kern_sum + gpu_trace + api_sum
- `nsys_stats/captureB_serve.nsys-rep` — CAPTURE B serving trace (12s steady decode, TP4)
- `nsys_stats/captureB_breakdown.txt`, `captureB_per_token.txt` — CAPTURE B category breakdown
- (full CAPTURE B SQLite at `/home/kacper/captureB_serve.sqlite`, 810MB — regenerate via `nsys export`)
