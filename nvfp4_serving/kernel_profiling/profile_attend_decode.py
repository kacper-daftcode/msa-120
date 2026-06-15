#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deep-dive profile of the WORST fair head-to-head: sparse_attend decode bs1.

No ncu/nsys in the image -> torch profiler (CUDA kernel durations) + occupancy
math from the launch geometry. Goal: name the concrete limiter for our
forward_sparse_paged in the bs1-decode regime (where we are 5x slower than the
Triton split-K decode kernel).
"""
import os, sys
import torch
from torch.profiler import profile, ProfilerActivity
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INDEX_TOPK_BLOCKS, SCALE, DTYPE

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

DEV="cuda"
Hq,Hkv,d = NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM
topk = INDEX_TOPK_BLOCKS
q_len=1; seq_kv=16384
nblk = seq_kv//128; sel=min(topk,nblk)

dev_props = torch.cuda.get_device_properties(0)
print(f"=== device: {dev_props.name} ===")
print(f"SMs={dev_props.multi_processor_count}  "
      f"max_threads_per_sm={dev_props.max_threads_per_multi_processor}  "
      f"regs_per_sm={getattr(dev_props,'regs_per_multiprocessor','?')}  "
      f"smem_per_sm={getattr(dev_props,'shared_memory_per_multiprocessor','?')}")
SM = dev_props.multi_processor_count

paged = build("sm120_fmha_paged","sm120_fmha_paged.cu")
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode

torch.manual_seed(0)
q = torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)

# ---- ours (page-64) ----
npage64 = nblk*2
k_cache = torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE)
v_cache = torch.randn(npage64,64,Hkv,d,device=DEV,dtype=DTYPE)
num_m=1
bt = torch.arange(npage64,device=DEV,dtype=torch.int32).view(1,-1).expand(num_m,npage64).contiguous()
sel_blocks=sorted(range(nblk-sel,nblk))
ids64=[]
for b in sel_blocks: ids64+=[2*b,2*b+1]
ids = torch.tensor(ids64,device=DEV,dtype=torch.int32).view(1,-1).expand(num_m,len(ids64)).contiguous()
def run_ours():
    return paged.forward_sparse_paged(q,k_cache,v_cache,bt,ids,float(SCALE),False,int(seq_kv))

# ---- triton (page-128) ----
kv_cache = torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
btt = torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk)
sel_t = torch.tensor(sorted(range(nblk-sel,nblk))+[-1]*(topk-sel),device=DEV,dtype=torch.int32)
topk_idx = sel_t.view(1,1,topk).expand(Hkv,q_len,topk).contiguous()
seq_lens = torch.tensor([seq_kv],device=DEV,dtype=torch.int32)
out = torch.empty(q_len,Hq,d,device=DEV,dtype=DTYPE)
def run_triton():
    minimax_m3_sparse_attn_decode(q,kv_cache,topk_idx,btt,seq_lens,num_kv_heads=Hkv,
                                  sm_scale=float(SCALE),output=out,decode_query_len=q_len)

for _ in range(30): run_ours(); run_triton()
torch.cuda.synchronize()

def prof(fn, tag):
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as p:
        for _ in range(50): fn()
        torch.cuda.synchronize()
    print(f"\n===== {tag} =====")
    ka = p.key_averages()
    rows=[]
    for e in ka:
        if e.device_type.name=="CUDA" or e.self_device_time_total>0:
            if e.self_device_time_total>0:
                rows.append((e.key, e.self_device_time_total/1000.0, e.count))
    rows.sort(key=lambda r:-r[1])
    tot=sum(r[1] for r in rows)
    for k,t,c in rows[:8]:
        print(f"  {t/50*1000:8.2f}us/call  ({100*t/tot:5.1f}%)  x{c:<4d}  {k[:70]}")
    print(f"  -- total CUDA {tot/50*1000:.2f}us/call --")
    return rows

ro = prof(run_ours, "OURS forward_sparse_paged (page-64)")
rt = prof(run_triton, "TRITON minimax_m3_sparse_attn_decode (page-128, split-K)")

# ---- occupancy math ----
print("\n===== OCCUPANCY / LAUNCH GEOMETRY =====")
# ours: grid=(num_m_blocks=1, num_heads_q=64) -> 64 blocks, 128 threads each
ours_blocks = num_m * Hq
print(f"OURS:   grid=(num_m={num_m}, Hq={Hq}) = {ours_blocks} thread-blocks, "
      f"128 thr/blk. SMs={SM}.")
print(f"        -> {ours_blocks} blocks across {SM} SMs = "
      f"{ours_blocks/SM:.2f} blocks/SM (ideal >=1-2 for full occupancy).")
print(f"        Each block: 64x128 M-tile but only q_len={q_len} real query row "
      f"-> {q_len}/64 = {100*q_len/64:.1f}% useful rows. {sel} blocks scanned "
      f"sequentially per block.")
print(f"        Work/block ~ {sel} K-blocks x (64x128 QK + softmax + 64x128 PV) "
      f"with 63/64 query rows masked-dead.")
print(f"TRITON: split-K decode -> grid=(total_q*num_topk_chunks, num_kv_heads). "
      f"parallelizes the {sel} selected blocks across chunks + a merge kernel, "
      f"so the bs1 query's work is spread over many more thread-blocks.")
print(f"        Triton block does only the {Hq//Hkv}-head GQA group per kv-head "
      f"(BLOCK_SIZE_H={Hq//Hkv}) -- no 64-row M-tile waste.")
