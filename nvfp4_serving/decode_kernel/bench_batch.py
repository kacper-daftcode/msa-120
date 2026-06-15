#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Batch-scaling: kernel-only profiler self-time (us) for ours vs Triton at
R=1,2,4,8,16. seq_kv=16384. Shows where occupancy lets us cross Triton."""
import os, sys
import torch
sys.path.insert(0,"/work/kernel_profiling")
from bench_common import NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM,INDEX_TOPK_BLOCKS,SCALE,DTYPE
DEC_SRC="/work/decode_kernel"
def build(name,src,srcdir):
    from torch.utils.cpp_extension import load
    cu13="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    inc="/usr/local/cuda/targets/x86_64-linux/include"
    for h in ("cusparse.h","cusolverDn.h","cusolver_common.h"):
        dd=os.path.join(inc,h); s=os.path.join(cu13,h)
        if not os.path.exists(dd) and os.path.exists(s):
            try: os.symlink(s,dd)
            except OSError: pass
    return load(name=name,sources=[os.path.join(srcdir,src)],extra_include_paths=[srcdir],
        extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f","-O3","-std=c++17","--expt-relaxed-constexpr"],verbose=False)
decode=build("sm120_fmha_decode","sm120_fmha_decode.cu",DEC_SRC)
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode
from torch.profiler import profile,ProfilerActivity
DEV="cuda"; Hq,Hkv,d=NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM; topk=INDEX_TOPK_BLOCKS
def kern_us(fn,n=100):
    for _ in range(30): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n): fn()
        torch.cuda.synchronize()
    return sum(e.self_device_time_total for e in p.key_averages() if e.self_device_time_total>0)/n
CH=int(os.environ.get("CH","32")); W4=int(os.environ.get("W4","2"))
seq_kv=int(os.environ.get("SEQ","16384")); nblk=seq_kv//128; sel=min(topk,nblk)
sel_blocks=sorted(range(nblk-sel,nblk))
print(f"# seq_kv={seq_kv} CH={CH}  kernel-only profiler us")
print(f"{'R':>3} {'triton':>8} {'t/req':>7} {'ours':>8} {'o/req':>7} {'T/ours':>7}")
for R in (1,2,4,8,16):
    torch.manual_seed(0)
    q=torch.randn(R,Hq,d,device=DEV,dtype=DTYPE)
    k128=torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    v128=torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    ids=torch.tensor(sel_blocks,device=DEV,dtype=torch.int32).view(1,-1).expand(R,sel).contiguous()
    bt=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,-1).expand(R,nblk).contiguous()
    o_us=kern_us(lambda: decode.forward_sparse_decode_p128(q,k128,v128,bt,ids,float(SCALE),int(seq_kv),CH,W4))
    kvT=torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
    btt=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk).expand(R,nblk).contiguous()
    sel_t=torch.tensor(sel_blocks+[-1]*(topk-len(sel_blocks)),device=DEV,dtype=torch.int32)
    tki=sel_t.view(1,1,topk).expand(Hkv,R,topk).contiguous()
    seq_lens=torch.tensor([seq_kv]*R,device=DEV,dtype=torch.int32)
    out=torch.empty(R,Hq,d,device=DEV,dtype=DTYPE)
    try:
        t_us=kern_us(lambda: minimax_m3_sparse_attn_decode(q,kvT,tki,btt,seq_lens,num_kv_heads=Hkv,sm_scale=float(SCALE),output=out,decode_query_len=1))
        print(f"{R:>3} {t_us:>8.2f} {t_us/R:>7.2f} {o_us:>8.2f} {o_us/R:>7.2f} {t_us/o_us:>6.2f}x")
    except Exception as ex:
        print(f"{R:>3} {'ERR':>8} {'-':>7} {o_us:>8.2f} {o_us/R:>7.2f} {'-':>7}  ({str(ex)[:40]})")
