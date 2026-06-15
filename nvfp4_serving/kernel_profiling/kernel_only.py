#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Kernel-only times (sum of CUDA self-time, no Python dispatch) for the fair
head-to-head ops, via torch profiler. Complements bench_msa.py's batched
event timing (which includes Triton's per-call autotune/dispatch overhead)."""
import os, sys, json
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INDEX_N_HEADS,
    INDEX_TOPK_BLOCKS, INDEX_INIT_BLOCKS, INDEX_LOCAL_BLOCKS, SCALE, DTYPE)

CSRC="/work/msa_kernels_serving/kernels"
def build(name,src):
    from torch.utils.cpp_extension import load
    cu13="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"; inc="/usr/local/cuda/targets/x86_64-linux/include"
    for h in ("cusparse.h","cusolverDn.h","cusolver_common.h"):
        d=os.path.join(inc,h); s=os.path.join(cu13,h)
        if not os.path.exists(d) and os.path.exists(s):
            try: os.symlink(s,d)
            except OSError: pass
    return load(name=name,sources=[os.path.join(CSRC,src)],extra_include_paths=[CSRC],
        extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f","-O3","-std=c++17","--expt-relaxed-constexpr"],verbose=False)

DEV="cuda"; SPARSE_BLOCK=128
def kern_us(fn, n=50):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    tot=sum(e.self_device_time_total for e in p.key_averages() if e.self_device_time_total>0)
    return tot/n  # self_device_time_total is in microseconds -> us/call

Hq,Hkv,d=NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM; topk=INDEX_TOPK_BLOCKS
DECODE=[4096,16384,65536]; PREFILL=[512,2048]
regimes=[("decode",1,c) for c in DECODE]+[("prefill",q,q) for q in PREFILL]

paged=build("sm120_fmha_paged","sm120_fmha_paged.cu")
topk_ext=build("sm120_sparse_topk","sm120_sparse_topk.cu")
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode
from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_topk

res=[]
for kind,q_len,seq_kv in regimes:
    nblk=seq_kv//SPARSE_BLOCK if seq_kv%SPARSE_BLOCK==0 else seq_kv//SPARSE_BLOCK+1
    sel=min(topk,nblk); torch.manual_seed(0)
    # ---- attend ----
    q=torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)
    sel_blocks=sorted(range(max(0,nblk-sel),nblk))
    npage64=nblk*2
    k64=torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE); v64=torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE)
    num_m=(q_len+63)//64
    bt=torch.arange(npage64,device=DEV,dtype=torch.int32).view(1,-1).expand(num_m,npage64).contiguous()
    ids64=[]
    for b in sel_blocks: ids64+=[2*b,2*b+1]
    ids=torch.tensor(ids64,device=DEV,dtype=torch.int32).view(1,-1).expand(num_m,len(ids64)).contiguous()
    ours_causal=(kind=="prefill")
    run_o=lambda: paged.forward_sparse_paged(q,k64,v64,bt,ids,float(SCALE),ours_causal,int(seq_kv))
    kv128=torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
    btt=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk)
    sel_t=torch.tensor(sel_blocks+[-1]*(topk-len(sel_blocks)),device=DEV,dtype=torch.int32)
    tki=sel_t.view(1,1,topk).expand(Hkv,q_len,topk).contiguous()
    seq_lens=torch.tensor([seq_kv],device=DEV,dtype=torch.int32); out=torch.empty(q_len,Hq,d,device=DEV,dtype=DTYPE)
    if kind=="decode":
        run_t=lambda: minimax_m3_sparse_attn_decode(q,kv128,tki,btt,seq_lens,num_kv_heads=Hkv,sm_scale=float(SCALE),output=out,decode_query_len=q_len)
    else:
        cu=torch.tensor([0,q_len],device=DEV,dtype=torch.int32); pl=torch.tensor([0],device=DEV,dtype=torch.int32)
        run_t=lambda: minimax_m3_sparse_attn(q,kv128,tki,btt,cu,seq_lens,pl,max_query_len=q_len,num_kv_heads=Hkv,sm_scale=float(SCALE),output=out)
    ao=kern_us(run_o); at=kern_us(run_t)
    # ---- topk ----
    H=INDEX_N_HEADS
    score=torch.randn(H,q_len,nblk,device=DEV,dtype=torch.float32)
    cu=torch.tensor([0,q_len],device=DEV,dtype=torch.int32)
    pl=torch.tensor([seq_kv-q_len if kind=="decode" else 0],device=DEV,dtype=torch.int32)
    run_tk_t=lambda: minimax_m3_index_topk(score,cu,pl,max_query_len=q_len,topk=topk,init_blocks=INDEX_INIT_BLOCKS,local_blocks=INDEX_LOCAL_BLOCKS)
    so=score.permute(0,2,1).contiguous()
    run_tk_o=lambda: topk_ext.topk_select(so,int(nblk),int(INDEX_INIT_BLOCKS),int(INDEX_LOCAL_BLOCKS))
    to=kern_us(run_tk_o); tt=kern_us(run_tk_t)
    res.append(dict(regime=kind,q_len=q_len,seq_kv=seq_kv,
                    attend_our=ao,attend_tri=at,attend_ratio=at/ao,
                    topk_our=to,topk_tri=tt,topk_ratio=tt/to))
    print(f"{kind:7s} q={q_len:<5d} kv={seq_kv:<6d} | ATTEND our={ao:7.2f} tri={at:6.2f} r={at/ao:5.2f}x"
          f" | TOPK our={to:6.2f} tri={tt:6.2f} r={tt/to:5.2f}x  (kernel-only us)")

json.dump(res,open("/work/kernel_profiling/results/kernel_only.json","w"),indent=2)
print("wrote kernel_only.json")
