#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""bs1 DECODE head-to-head: our PAGE-128 forward_sparse_decode_p128 (and the
page-64 forward_sparse_decode for reference) vs vLLM Triton
minimax_m3_sparse_attn_decode. Kernel-only CUDA self-time median.
Target: <= Triton ~5.7us. Run on CUDA_VISIBLE_DEVICES=0 inside the bench
container.
"""
import os, sys
import torch
sys.path.insert(0, "/work/kernel_profiling")
from bench_common import (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INDEX_TOPK_BLOCKS,
                          SCALE, DTYPE)

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

def kern_us(fn, n=100, want_breakdown=False):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    evs = [e for e in p.key_averages() if e.self_device_time_total>0]
    tot=sum(e.self_device_time_total for e in evs)
    if want_breakdown:
        bd = {e.key: e.self_device_time_total/n for e in evs}
        return tot/n, bd
    return tot/n

# sub64 page-128 path: split-K at 64-key granularity, chunks up to 2*topk=32
# (=> 128 real-work blocks, the measured occupancy sweet spot at bs1).
CHUNK64  = [int(x) for x in os.environ.get("CHUNKS64","32").split(",")]
CHUNK128 = [int(x) for x in os.environ.get("CHUNKS128","16,24,32").split(",")]

print(f"{'seq_kv':>8} {'triton':>8} {'p64best':>8} {'p128best':>9} {'p128c':>5} "
      f"{'T/p128':>7}  per-chunk-p128(us)")
for seq_kv in (4096, 16384, 65536):
    nblk = seq_kv//128
    sel = min(topk, nblk)
    torch.manual_seed(0)
    q = torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)

    # page-128 native cache
    k128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    v128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    sel_blocks = sorted(range(nblk-sel, nblk))
    ids128 = torch.tensor(sel_blocks,device=DEV,dtype=torch.int32).view(1,-1)
    bt128  = torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,-1)

    # page-64 cache (same data) for reference
    k64 = k128.reshape(nblk,2,64,Hkv,d).reshape(nblk*2,64,Hkv,d).contiguous()
    v64 = v128.reshape(nblk,2,64,Hkv,d).reshape(nblk*2,64,Hkv,d).contiguous()
    ids64=[]
    for b in sel_blocks: ids64 += [2*b,2*b+1]
    bt64  = torch.arange(nblk*2,device=DEV,dtype=torch.int32).view(1,-1)
    ids64t= torch.tensor(ids64,device=DEV,dtype=torch.int32).view(1,-1)

    # triton page-128 fused cache [nblk,2,128,Hkv,d]
    kvT = torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
    btt = torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk)
    sel_t = torch.tensor(sel_blocks+[-1]*(topk-len(sel_blocks)),device=DEV,dtype=torch.int32)
    tki = sel_t.view(1,1,topk).expand(Hkv,q_len,topk).contiguous()
    seq_lens = torch.tensor([seq_kv],device=DEV,dtype=torch.int32)
    out = torch.empty(q_len,Hq,d,device=DEV,dtype=DTYPE)
    run_t = lambda: minimax_m3_sparse_attn_decode(q,kvT,tki,btt,seq_lens,num_kv_heads=Hkv,
                                                  sm_scale=float(SCALE),output=out,decode_query_len=q_len)
    t_us = kern_us(run_t)

    per64 = {}
    for c in CHUNK64:
        r = lambda c=c: decode.forward_sparse_decode(q,k64,v64,bt64,ids64t,float(SCALE),int(seq_kv),int(c))
        per64[c] = kern_us(r)
    p64best = min(per64.values())

    W4 = int(os.environ.get("W4","2"))   # 2 = sub64 4-warp (integration-ready best)
    per128 = {}
    for c in CHUNK128:
        r = lambda c=c: decode.forward_sparse_decode_p128(q,k128,v128,bt128,ids128,float(SCALE),int(seq_kv),int(c),W4)
        per128[c] = kern_us(r)
    bc = min(per128, key=per128.get); bo = per128[bc]
    cs = " ".join(f"{c}:{per128[c]:.2f}" for c in CHUNK128)
    print(f"{seq_kv:>8} {t_us:>8.2f} {p64best:>8.2f} {bo:>9.2f} {bc:>5} {t_us/bo:>7.2f}x  {cs}")

# per-kernel breakdown for p128 at 16384, best chunk
print("\n--- per-kernel breakdown (seq_kv=16384) ---")
seq_kv=16384; nblk=seq_kv//128; sel=min(topk,nblk)
torch.manual_seed(0)
q = torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)
k128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
v128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
sel_blocks=sorted(range(nblk-sel,nblk))
ids128=torch.tensor(sel_blocks,device=DEV,dtype=torch.int32).view(1,-1)
bt128=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,-1)
W4=int(os.environ.get("W4","2"))
for c in CHUNK128:
    r=lambda c=c: decode.forward_sparse_decode_p128(q,k128,v128,bt128,ids128,float(SCALE),int(seq_kv),int(c),W4)
    tot,bd=kern_us(r,want_breakdown=True)
    parts=" ".join(f"{k.split('(')[0][-28:]}:{v:.2f}" for k,v in sorted(bd.items(),key=lambda x:-x[1]))
    print(f"chunks={c:>2} tot={tot:.2f}  {parts}")
