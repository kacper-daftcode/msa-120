#!/usr/bin/env python3
"""
profile_moe.py — per-component time breakdown of ONE decode forward pass of the
NVFP4 MoE path on MiniMax-M3, using the torch profiler.

  *** REQUIRES A DEDICATED GPU SLOT. ***
  Do NOT run while the vLLM server is up: it holds gpu-memory-utilization 0.95
  on all 4 GPUs and there is no room to allocate. Run this ONLY when
  `nvidia-smi` shows the GPUs are free (server stopped / orchestrator grant).

What it does
------------
Spins up vLLM offline (LLM(...)) with the SAME config as the live server
(TP4, block-size 128, bf16 KV, enforce-eager, max-model-len 65536), warms up,
then profiles a single batch-size-1 decode step under torch.profiler with CUDA
activities. It prints the top ops by self-CUDA time and flags the ops that are
the suspected host-sync stalls (fp4_quantize, .item()/.cpu() copies, the
per-expert Python group_gemm loop).

Two collection modes (pick with --tool):
  --tool torch  : torch.profiler (default). Dumps a Chrome trace + a sorted
                  table of top CUDA ops. Good for op-level attribution.
  --tool nsys   : prints the exact `nsys profile` command to wrap this script
                  for a kernel/timeline + CUDA API host-sync view (cudaStreamSync,
                  cudaMemcpyAsync D2H). Run that, then open the .nsys-rep.

Usage (only in a free GPU slot):
  python3 profile_moe.py --tool torch --decode-steps 1 --warmup 8
  python3 profile_moe.py --tool nsys          # prints nsys wrapper command

Notes
-----
* enforce_eager=True matches the live server (no CUDA graphs) so the profile
  reflects the real, un-captured per-op dispatch — which is exactly where the
  per-expert Python loop + host syncs hurt.
* The MoE op names to watch for in the table (vLLM NVFP4 / ModelOpt path):
    - scaled_fp4_quant / fp4_quantize        (NVFP4 quantization of activations)
    - cutlass_scaled_fp4_mm / group_gemm     (per-expert FP4 GEMMs)
    - the swigluoai activation (clamp/sigmoid/mul elementwise kernels)
    - all_reduce / nccl                      (TP4 collective after down-proj)
    - flash_attn / lightning / MSA attention kernels
    - the sampler (argmax/top-k) at the end
  Any op preceded by a long gap with CPU busy but GPU idle == a host sync stall.
"""

import argparse
import os
import sys


LIVE_ARGS = dict(
    model="/models/MiniMax-M3-NVFP4",
    tensor_parallel_size=4,
    block_size=128,
    max_model_len=65536,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
)


def _check_gpu_free():
    """Refuse to run if GPUs look occupied (e.g. server still up)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if used and max(used) > 5000:
            print(f"[profile_moe] REFUSING TO RUN: GPU memory in use {used} MiB.\n"
                  f"  The vLLM server is probably still up (holds 0.95). Stop it\n"
                  f"  or wait for a dedicated GPU slot before profiling.",
                  file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[profile_moe] WARN: could not check GPU state ({e}); continuing.",
              file=sys.stderr)
        return True


def print_nsys_command(args):
    here = os.path.abspath(__file__)
    cmd = (
        "nsys profile \\\n"
        "  --trace=cuda,nvtx,osrt \\\n"
        "  --cuda-memory-usage=true \\\n"
        "  --capture-range=cudaProfilerApi \\\n"   # only the marked decode step
        "  --capture-range-end=stop \\\n"
        "  -o moe_decode_profile \\\n"
        f"  python3 {here} --tool torch --nsys-range --decode-steps 1\n"
    )
    print("# Run this in a FREE GPU slot (server stopped). Produces "
          "moe_decode_profile.nsys-rep\n")
    print(cmd)
    print("# Then inspect host-sync stalls:")
    print("#   nsys stats --report cuda_api_sum moe_decode_profile.nsys-rep")
    print("#   nsys stats --report cuda_gpu_kern_sum moe_decode_profile.nsys-rep")
    print("# Look for cudaStreamSynchronize / cudaMemcpyAsync(DtoH) with high")
    print("# total time and GPU-idle gaps -> those are the fp4_quantize / per-expert")
    print("# loop host syncs that dominate the NVFP4 decode path.")


def run_torch_profile(args):
    if not args.skip_gpu_check and not _check_gpu_free():
        sys.exit(2)

    import torch
    from vllm import LLM, SamplingParams

    print("[profile_moe] building offline LLM (this allocates all 4 GPUs)...",
          file=sys.stderr)
    llm = LLM(**LIVE_ARGS)

    sp = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    # A short prompt so prefill is cheap; we care about the DECODE step.
    warm_prompt = "The quick brown fox jumps over the lazy dog. " * 4

    # Warmup: prime allocator / any lazy init. enforce_eager => no graph capture.
    print(f"[profile_moe] warmup x{args.warmup} ...", file=sys.stderr)
    for _ in range(args.warmup):
        llm.generate([warm_prompt], SamplingParams(
            temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)

    torch.cuda.synchronize()

    # --- profile a single decode step ---
    # Generate 2 tokens: token 1 = prefill+decode, token 2 = a pure decode step.
    # The profiler window captures both; the steady-state decode op mix is what
    # matters. Use max_tokens slightly >1 to ensure a genuine decode iteration.
    from torch.profiler import profile, ProfilerActivity, schedule

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    nsys_range = getattr(args, "nsys_range", False)

    if nsys_range:
        torch.cuda.cudart().cudaProfilerStart()

    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(args.decode_steps):
            llm.generate(
                [warm_prompt],
                SamplingParams(temperature=0.0, max_tokens=args.decode_tokens,
                               ignore_eos=True),
                use_tqdm=False,
            )
        torch.cuda.synchronize()

    if nsys_range:
        torch.cuda.cudart().cudaProfilerStop()

    # --- report ---
    print("\n================ TOP OPS BY SELF CUDA TIME ================")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=args.row_limit))

    print("\n================ TOP OPS BY SELF CPU TIME (host-side; "
          "long CPU w/ little CUDA == host-sync stall) ================")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=args.row_limit))

    trace_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "moe_decode_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\n[profile_moe] Chrome trace -> {trace_path}")
    print("[profile_moe] Open in chrome://tracing or https://ui.perfetto.dev")

    # Heuristic: surface suspected host-sync ops by name.
    SYNC_HINTS = ("fp4_quant", "quantize", "memcpy", "item", "_local_scalar",
                  "to_copy", "copy_", "nonzero", "synchronize", "cudaStreamSync")
    print("\n================ SUSPECTED HOST-SYNC / QUANT OPS ================")
    for row in prof.key_averages():
        name = row.key.lower()
        if any(h in name for h in SYNC_HINTS):
            print(f"  {row.key:<45} "
                  f"cuda={row.self_cuda_time_total/1e3:8.2f}ms "
                  f"cpu={row.self_cpu_time_total/1e3:8.2f}ms "
                  f"calls={row.count}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", choices=["torch", "nsys"], default="torch")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=1,
                    help="profiler outer iterations (each does one generate())")
    ap.add_argument("--decode-tokens", type=int, default=2,
                    help="tokens per generate inside the profiled window "
                         "(>=2 to capture a steady-state decode iteration)")
    ap.add_argument("--row-limit", type=int, default=40)
    ap.add_argument("--nsys-range", action="store_true",
                    help="mark a cudaProfilerApi range (used by the nsys wrapper)")
    ap.add_argument("--skip-gpu-check", action="store_true")
    args = ap.parse_args()

    if args.tool == "nsys":
        print_nsys_command(args)
        return
    run_torch_profile(args)


if __name__ == "__main__":
    main()
