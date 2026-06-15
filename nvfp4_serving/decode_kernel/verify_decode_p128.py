#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Correctness of the PAGE-128 decode kernel (forward_sparse_decode_p128) vs:
  (a) the GOLDEN forward_sparse_paged (page-64, causal=False at decode), rms<1e-2,
  (b) the page-64 forward_sparse_decode (same data relaid), and
  (c) a dense fp32 softmax reference.
The page-128 cache is the native M3 layout [num_pages,128,Hkv,128]; a 128-block b
maps to page-64 ids [2b,2b+1]. Run inside vllm/vllm-openai:minimax-m3 on
CUDA_VISIBLE_DEVICES=0.
"""
import os, sys
import torch
sys.path.insert(0, "/work/kernel_profiling")
from bench_common import NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INDEX_TOPK_BLOCKS, SCALE, DTYPE

PAGED_SRC = "/work/msa_kernels_serving/kernels"
DEC_SRC   = "/work/decode_kernel"

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

paged  = build("sm120_fmha_paged","sm120_fmha_paged.cu", PAGED_SRC)
decode = build("sm120_fmha_decode","sm120_fmha_decode.cu", DEC_SRC)

DEV="cuda"
Hq,Hkv,d = NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM
topk = INDEX_TOPK_BLOCKS
g = Hq//Hkv

def run_case(seq_kv, nsel, split_chunks, seed=0, R=1):
    torch.manual_seed(seed)
    nblk = seq_kv//128                      # number of 128-blocks of real KV
    nsel = min(nsel, nblk)
    # ---- native page-128 cache [nblk,128,Hkv,128] ----
    k128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    v128 = torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
    # page-64 view of the SAME data: page-64 id 2b+s == k128[b, s*64:(s+1)*64]
    k64 = k128.reshape(nblk,2,64,Hkv,d).reshape(nblk*2,64,Hkv,d).contiguous()
    v64 = v128.reshape(nblk,2,64,Hkv,d).reshape(nblk*2,64,Hkv,d).contiguous()
    npage64 = nblk*2

    sel_blocks = sorted(range(nblk-nsel, nblk))
    # page-128 selection: logical 128-page ids
    ids128 = torch.tensor(sel_blocks,device=DEV,dtype=torch.int32).view(1,-1).expand(R,nsel).contiguous()
    bt128  = torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,-1).expand(R,nblk).contiguous()
    # page-64 selection (for golden + page-64 decode)
    ids64=[]
    for b in sel_blocks: ids64 += [2*b, 2*b+1]
    bt64  = torch.arange(npage64,device=DEV,dtype=torch.int32).view(1,-1).expand(R,npage64).contiguous()
    ids64t= torch.tensor(ids64,device=DEV,dtype=torch.int32).view(1,-1).expand(R,len(ids64)).contiguous()
    q = torch.randn(R,Hq,d,device=DEV,dtype=DTYPE)

    # GOLDEN (prefill page-64, causal=False) per request
    o_gold = torch.empty(R,Hq,d,device=DEV,dtype=DTYPE)
    for r in range(R):
        og,_ = paged.forward_sparse_paged(q[r:r+1], k64, v64, bt64[r:r+1], ids64t[r:r+1],
                                          float(SCALE), False, int(seq_kv))
        o_gold[r] = og[0]

    W4 = int(os.environ.get("W4", "0"))
    # PAGE-128 DECODE kernel (W4=1 -> 4-warp variant)
    o_p128,_ = decode.forward_sparse_decode_p128(q, k128, v128, bt128, ids128,
                                                 float(SCALE), int(seq_kv), int(split_chunks), W4)
    # PAGE-64 DECODE kernel (same data)
    o_p64,_  = decode.forward_sparse_decode(q, k64, v64, bt64, ids64t,
                                            float(SCALE), int(seq_kv), int(split_chunks))

    # dense fp32 reference
    Kf = k128.reshape(nblk*128,Hkv,d).float()
    Vf = v128.reshape(nblk*128,Hkv,d).float()
    sel_rows=[]
    for b in sel_blocks: sel_rows += list(range(b*128,(b+1)*128))
    sel_rows = torch.tensor(sel_rows,device=DEV)
    o_ref = torch.empty(R,Hq,d,device=DEV,dtype=torch.float32)
    for r in range(R):
        for h in range(Hq):
            kh=h//g
            qh=q[r,h].float()
            Kh=Kf[sel_rows,kh]; Vh=Vf[sel_rows,kh]
            s=(Kh@qh)*SCALE
            p=torch.softmax(s,0)
            o_ref[r,h]=(p[:,None]*Vh).sum(0)

    def stats(a,b):
        a=a.float(); b=b.float()
        return (a-b).abs().max().item(), (a-b).pow(2).mean().sqrt().item()
    e_g,rms_g   = stats(o_p128,o_gold)
    e_64,rms_64 = stats(o_p128,o_p64)
    e_r,rms_r   = stats(o_p128,o_ref)
    ok = rms_g < 1e-2 and rms_r < 5e-2
    print(f"seq_kv={seq_kv:6d} nsel={nsel:2d} chunks={split_chunks:2d} R={R} | "
          f"vs GOLD rms={rms_g:.3e} max={e_g:.3e} | vs p64 rms={rms_64:.3e} | "
          f"vs DENSE rms={rms_r:.3e} | {'OK' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    allok = True
    for seq_kv in (4096, 16384, 65536):
        for nsel in (16, 8, 3):
            for chunks in (0, 1, 4, 8, 16):
                allok &= run_case(seq_kv, nsel, chunks)
    allok &= run_case(16384, 16, 8, R=2)
    print("ALL OK" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
