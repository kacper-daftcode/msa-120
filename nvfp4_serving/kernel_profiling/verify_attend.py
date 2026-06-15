#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sanity: our forward_sparse_paged (page-64) vs a dense reference on a small
decode-shaped problem, so the timed 184us is confirmed real correct work."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, SCALE, DTYPE

CSRC = "/work/msa_kernels_serving/kernels"
def build(name, src):
    from torch.utils.cpp_extension import load
    cu13="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"; inc="/usr/local/cuda/targets/x86_64-linux/include"
    for h in ("cusparse.h","cusolverDn.h","cusolver_common.h"):
        d=os.path.join(inc,h); s=os.path.join(cu13,h)
        if not os.path.exists(d) and os.path.exists(s):
            try: os.symlink(s,d)
            except OSError: pass
    return load(name=name,sources=[os.path.join(CSRC,src)],extra_include_paths=[CSRC],
                extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f","-O3","-std=c++17","--expt-relaxed-constexpr"],verbose=False)

paged = build("sm120_fmha_paged","sm120_fmha_paged.cu")
DEV="cuda"
Hq,Hkv,d = NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM
q_len=1; seq_kv=512  # 4 blocks of 128 -> 8 pages of 64
nblk = seq_kv//128
sel = nblk  # select all blocks (dense within selected)
torch.manual_seed(0)
q = torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)
# page-64 cache
npage64 = nblk*2
k_cache = torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE)
v_cache = torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE)
num_m=1
bt = torch.arange(npage64,device=DEV,dtype=torch.int32).view(1,-1)
ids = torch.arange(npage64,device=DEV,dtype=torch.int32).view(1,-1)  # all 64-pages selected
CAUSAL = os.environ.get("VA_CAUSAL","1") == "1"
o,lse = paged.forward_sparse_paged(q,k_cache,v_cache,bt,ids,float(SCALE),CAUSAL,int(seq_kv))

# reference: gather full K/V [seq_kv, Hkv, d] from the 64-pages in order
K = k_cache.reshape(npage64*64,Hkv,d)[:seq_kv]   # [seq_kv,Hkv,d]
V = v_cache.reshape(npage64*64,Hkv,d)[:seq_kv]
# GQA: each q head maps to kv head h//(Hq//Hkv)
g = Hq//Hkv
ref = torch.empty_like(o)
for h in range(Hq):
    kh = h//g
    qh = q[0,h].float()              # [d]
    Kh = K[:,kh].float()             # [S,d]
    Vh = V[:,kh].float()
    s = (Kh @ qh) * SCALE            # [S]
    p = torch.softmax(s,dim=0)
    ref[0,h] = (p[:,None]*Vh).sum(0).to(DTYPE)
err = (o.float()-ref.float()).abs().max().item()
rms = (o.float()-ref.float()).pow(2).mean().sqrt().item()
print(f"decode-shape attend: max_abs_err={err:.4e} rms={rms:.4e}  {'OK' if rms<5e-2 else 'FAIL'}")
