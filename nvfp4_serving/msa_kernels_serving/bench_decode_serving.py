#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel-only latency of the SERVING decode entrypoint:
   partial (sm120_serve_decode_partial_p128_ldsm) + merge, vs Triton decode.
Run inside vllm/vllm-openai:minimax-m3 on CUDA_VISIBLE_DEVICES=0.
   python3 /work/msa_kernels_serving/bench_decode_serving.py
Canonical shape: bs1, seq_kv=16384, Hq=64/Hkv=4, topk=16, chunks=16.
"""
import os, sys, statistics
import torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity

KDIR = "/work/msa_kernels_serving/kernels"
Hq, Hkv, d = 64, 4, 128
g = Hq // Hkv
topk = 16
SCALE = 1.0 / (d ** 0.5)
DTYPE = torch.bfloat16
DEV = "cuda"

def build(name, src, srcdir):
    cu13 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    inc = "/usr/local/cuda/targets/x86_64-linux/include"
    for h in ("cusparse.h", "cusolverDn.h", "cusolver_common.h"):
        dd = os.path.join(inc, h); s = os.path.join(cu13, h)
        if not os.path.exists(dd) and os.path.exists(s):
            try: os.symlink(s, dd)
            except OSError: pass
    return load(name=name, sources=[os.path.join(srcdir, src)],
                extra_include_paths=[srcdir],
                extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f",
                                   "-O3", "-std=c++17", "--expt-relaxed-constexpr"],
                verbose=False)

serve = build("sm120_fmha_decode_serving", "sm120_fmha_decode_serving.cu", KDIR)


def make_inputs(seq_kv=16384, nsel=16, chunks=16, R=1):
    nblk = seq_kv // 128
    k128 = torch.randn(nblk, 128, Hkv, d, device=DEV, dtype=DTYPE)
    v128 = torch.randn(nblk, 128, Hkv, d, device=DEV, dtype=DTYPE)
    kv_cache = torch.stack([k128, v128], dim=1).contiguous()
    q = torch.randn(R, Hq, d, device=DEV, dtype=DTYPE)
    bt = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, -1).expand(R, nblk).contiguous()
    seq_lens = torch.full((R,), seq_kv, device=DEV, dtype=torch.int32)
    ids = torch.full((R, Hkv, topk), -1, device=DEV, dtype=torch.int32)
    base = list(range(nblk - nsel, nblk))
    for r in range(R):
        for h in range(Hkv):
            ids[r, h, :len(base)] = torch.tensor(base, device=DEV, dtype=torch.int32)
    return q, kv_cache, bt, ids, seq_lens, chunks


def fn_serve(args):
    q, kv_cache, bt, ids, seq_lens, chunks = args
    return serve.forward_sparse_decode_serving(
        q, kv_cache, bt, ids, seq_lens, float(SCALE), Hkv, int(chunks))


def kern_breakdown(fn, args, n=200):
    for _ in range(30):
        fn(args)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn(args)
        torch.cuda.synchronize()
    partial_us = merge_us = 0.0
    rows = []
    for e in prof.key_averages():
        nm = e.key
        self_us = e.self_device_time_total / max(1, e.count)
        rows.append((nm, self_us, e.count))
        if "partial_p128_ldsm" in nm:
            partial_us = self_us
        elif "merge_bf16" in nm:
            merge_us = self_us
    return partial_us, merge_us, rows


if __name__ == "__main__":
    args = make_inputs(seq_kv=16384, nsel=16, chunks=16, R=1)
    p, m, rows = kern_breakdown(fn_serve, args, n=300)
    print(f"== serving decode bs1 seq=16384 topk=16 chunks=16 ==")
    print(f"partial = {p:.3f} us   merge = {m:.3f} us   total = {p+m:.3f} us")
    print("--- all CUDA kernels (self us/call, count) ---")
    for nm, us, c in sorted(rows, key=lambda x: -x[1])[:8]:
        print(f"  {us:8.3f}us x{c:6d}  {nm[:80]}")
