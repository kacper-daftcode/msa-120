#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CUDA-GRAPH bs1 decode: ours (2-kernel split-K+merge) vs Triton, wall-clock
of the captured graph (the real serving path; launch overhead amortized).
Also reports the batch curve R=1,2,4,8."""
import os, sys
import torch
sys.path.insert(0, "/work/kernel_profiling")
from bench_common import (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INDEX_TOPK_BLOCKS,
                          SCALE, DTYPE)
DEC_SRC="/work/decode_kernel"
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
        extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f","-O3","-std=c++17","--expt-relaxed-constexpr"],verbose=False)
decode=build("sm120_fmha_decode","sm120_fmha_decode.cu",DEC_SRC)
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode
DEV="cuda"; Hq,Hkv,d=NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM; topk=INDEX_TOPK_BLOCKS

def graph_us(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph(); s=torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize()
    for _ in range(10): g.replay()
    torch.cuda.synchronize()
    e0=torch.cuda.Event(True); e1=torch.cuda.Event(True)
    e0.record()
    for _ in range(n): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1)/n*1000.0

CH=int(os.environ.get("CH","32")); W4=int(os.environ.get("W4","2"))
PV=int(os.environ.get("PV_BF16","0")); FM=int(os.environ.get("FUSED_MERGE","0"))
os.environ["PV_BF16"]=str(PV); os.environ["FUSED_MERGE"]=str(FM)
print(f"# CH={CH} W4={W4} PV_BF16={PV} FUSED_MERGE={FM}  (CUDA-graph wall us)")
print(f"{'seq_kv':>8} {'R':>3} {'triton':>8} {'ours':>8} {'T/ours':>7}")
for seq_kv in (4096,16384,65536):
    nblk=seq_kv//128; sel=min(topk,nblk)
    sel_blocks=sorted(range(nblk-sel,nblk))
    for R in (1,2,4,8):
        torch.manual_seed(0)
        q=torch.randn(R,Hq,d,device=DEV,dtype=DTYPE)
        k128=torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
        v128=torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
        ids128=torch.tensor(sel_blocks,device=DEV,dtype=torch.int32).view(1,-1).expand(R,sel).contiguous()
        bt128=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,-1).expand(R,nblk).contiguous()
        run_o=lambda: decode.forward_sparse_decode_p128(q,k128,v128,bt128,ids128,float(SCALE),int(seq_kv),CH,W4)
        o_us=graph_us(run_o)
        # triton
        kvT=torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
        btt=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk).expand(R,nblk).contiguous()
        sel_t=torch.tensor(sel_blocks+[-1]*(topk-len(sel_blocks)),device=DEV,dtype=torch.int32)
        tki=sel_t.view(1,1,topk).expand(Hkv,R,topk).contiguous()
        seq_lens=torch.tensor([seq_kv]*R,device=DEV,dtype=torch.int32)
        out=torch.empty(R,Hq,d,device=DEV,dtype=DTYPE)
        try:
            run_t=lambda: minimax_m3_sparse_attn_decode(q,kvT,tki,btt,seq_lens,num_kv_heads=Hkv,sm_scale=float(SCALE),output=out,decode_query_len=1)
            t_us=graph_us(run_t)
            ts=f"{t_us:>8.2f}"; rat=f"{t_us/o_us:>6.2f}x"
        except Exception as ex:
            ts=f"{'ERR':>8}"; rat=f"{'-':>7}"
        print(f"{seq_kv:>8} {R:>3} {ts} {o_us:>8.2f} {rat}")
