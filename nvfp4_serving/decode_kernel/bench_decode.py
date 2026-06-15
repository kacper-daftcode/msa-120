#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""bs1 DECODE head-to-head: our forward_sparse_decode vs vLLM Triton
minimax_m3_sparse_attn_decode.  Kernel-only CUDA-event median (warmup+iters).
Target: <= Triton ~5.7us.  Run on CUDA_VISIBLE_DEVICES=0 inside
vllm/vllm-openai:minimax-m3.
"""
import os, sys, statistics
import torch
sys.path.insert(0, "/work/kernel_profiling")
from bench_common import (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INDEX_TOPK_BLOCKS,
                          SCALE, DTYPE, cuda_time_batched)

DEC_SRC = "/work/decode_kernel"

def build(name, src, srcdir):
    from torch.utils.cpp_extension import load
    cu13="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    inc="/usr/local/cuda/targets/x86_64-linux/include"
    for h in ("cusparse.h","cusolverDn.h","cusolver_common.h"):
        d=os.path.join(inc,h); s=os.path.join(cu13,h)
        if not os.path.exists(d) and os.path.exists(s):
            try: os.symlink(s,d)
            except OSError: pass
    return load(name=name, sources=[os.path.join(srcdir,src)], extra_include_paths=[srcdir],
        extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f","-O3","-std=c++17","--expt-relaxed-constexpr"],
        verbose=False)

decode = build("sm120_fmha_decode","sm120_fmha_decode.cu", DEC_SRC)
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode
from torch.profiler import profile, ProfilerActivity

DEV="cuda"
Hq,Hkv,d = NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM
topk = INDEX_TOPK_BLOCKS
q_len = 1

def kern_us(fn, n=80):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    tot=sum(e.self_device_time_total for e in p.key_averages() if e.self_device_time_total>0)
    return tot/n

CHUNK_GRID = [int(x) for x in os.environ.get("CHUNKS","8,16,24,32").split(",")]

print(f"{'seq_kv':>8} {'triton_us':>10} {'best_ours_us':>12} {'best_chunks':>11} "
      f"{'ratio(T/O)':>10}  per-chunk(us)")
for seq_kv in (4096, 16384, 65536):
    nblk = seq_kv//128
    sel = min(topk, nblk)
    torch.manual_seed(0)
    q = torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)

    # ours page-64 setup
    npage64 = nblk*2
    k64 = torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE)
    v64 = torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE)
    sel_blocks = sorted(range(nblk-sel, nblk))
    ids64=[]
    for b in sel_blocks: ids64 += [2*b,2*b+1]
    bt  = torch.arange(npage64,device=DEV,dtype=torch.int32).view(1,-1)
    ids = torch.tensor(ids64,device=DEV,dtype=torch.int32).view(1,-1)

    # triton page-128 setup
    kv128 = torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
    btt = torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk)
    sel_t = torch.tensor(sel_blocks+[-1]*(topk-len(sel_blocks)),device=DEV,dtype=torch.int32)
    tki = sel_t.view(1,1,topk).expand(Hkv,q_len,topk).contiguous()
    seq_lens = torch.tensor([seq_kv],device=DEV,dtype=torch.int32)
    out = torch.empty(q_len,Hq,d,device=DEV,dtype=DTYPE)
    run_t = lambda: minimax_m3_sparse_attn_decode(q,kv128,tki,btt,seq_lens,num_kv_heads=Hkv,
                                                  sm_scale=float(SCALE),output=out,decode_query_len=q_len)
    t_us = kern_us(run_t)

    per = {}
    for c in CHUNK_GRID:
        run_o = lambda c=c: decode.forward_sparse_decode(q,k64,v64,bt,ids,float(SCALE),int(seq_kv),int(c))
        per[c] = kern_us(run_o)
    bestc = min(per, key=per.get)
    bo = per[bestc]
    chunkstr = " ".join(f"{c}:{per[c]:.2f}" for c in CHUNK_GRID)
    print(f"{seq_kv:>8} {t_us:>10.2f} {bo:>12.2f} {bestc:>11} {t_us/bo:>10.2f}x  {chunkstr}")
