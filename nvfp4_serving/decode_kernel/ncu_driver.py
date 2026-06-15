#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Single-shape ncu driver for the bs1 decode block-sparse attention kernels.
Warms up, then issues a fixed number of steady-state launches that ncu captures.
Select impl via WHICH={ours,triton}; shape via SEQ_KV (default 16384).

Kernel names emitted (for ncu --kernel-name regex):
  ours partial : sm120_fmha_decode_partial_p128_sub64
  ours merge   : sm120_fmha_decode_merge_bf16
  triton partial: _gqa_sparse_decode_kernel
  triton merge  : _merge_topk_attn_out_kernel
"""
import os, sys
import torch
sys.path.insert(0, "/work/kernel_profiling")
from bench_common import (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INDEX_TOPK_BLOCKS,
                          SCALE, DTYPE)

DEC_SRC = "/work/decode_kernel"
WHICH = os.environ.get("WHICH", "ours")
SEQ_KV = int(os.environ.get("SEQ_KV", "16384"))
NLAUNCH = int(os.environ.get("NLAUNCH", "3"))
WARMUP = int(os.environ.get("WARMUP", "50"))
W4 = int(os.environ.get("W4", "2"))
CHUNKS = int(os.environ.get("CHUNKS", "32"))

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

DEV="cuda"
Hq,Hkv,d = NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM
topk = INDEX_TOPK_BLOCKS
q_len = 1
nblk = SEQ_KV//128
sel = min(topk, nblk)
torch.manual_seed(0)
q = torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)
sel_blocks = sorted(range(nblk-sel, nblk))

if WHICH == "ours":
    decode = build("sm120_fmha_decode","sm120_fmha_decode.cu", DEC_SRC)
    k128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    v128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    ids128 = torch.tensor(sel_blocks,device=DEV,dtype=torch.int32).view(1,-1)
    bt128  = torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,-1)
    def run():
        return decode.forward_sparse_decode_p128(q,k128,v128,bt128,ids128,float(SCALE),int(SEQ_KV),int(CHUNKS),W4)
else:
    from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode
    kvT = torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
    btt = torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk)
    sel_t = torch.tensor(sel_blocks+[-1]*(topk-len(sel_blocks)),device=DEV,dtype=torch.int32)
    tki = sel_t.view(1,1,topk).expand(Hkv,q_len,topk).contiguous()
    seq_lens = torch.tensor([SEQ_KV],device=DEV,dtype=torch.int32)
    out = torch.empty(q_len,Hq,d,device=DEV,dtype=DTYPE)
    def run():
        minimax_m3_sparse_attn_decode(q,kvT,tki,btt,seq_lens,num_kv_heads=Hkv,
                                      sm_scale=float(SCALE),output=out,decode_query_len=q_len)
        return out

for _ in range(WARMUP):
    run()
torch.cuda.synchronize()
print(f"[{WHICH}] warmed up seq_kv={SEQ_KV} chunks={CHUNKS} w4={W4}; issuing {NLAUNCH} profiled launches", flush=True)
torch.cuda.cudart().cudaProfilerStart()
for _ in range(NLAUNCH):
    run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print(f"[{WHICH}] done", flush=True)
