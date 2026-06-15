import os,sys,torch
sys.path.insert(0,"/work/kernel_profiling")
from bench_common import *
from torch.utils.cpp_extension import load
cu13="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"; inc="/usr/local/cuda/targets/x86_64-linux/include"
for h in ("cusparse.h","cusolverDn.h","cusolver_common.h"):
    dd=os.path.join(inc,h); s=os.path.join(cu13,h)
    if not os.path.exists(dd) and os.path.exists(s):
        try: os.symlink(s,dd)
        except OSError: pass
decode=load(name="sm120_fmha_decode",sources=["/work/decode_kernel/sm120_fmha_decode.cu"],extra_include_paths=["/work/decode_kernel"],extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f","-O3","-std=c++17","--expt-relaxed-constexpr"],verbose=False)
from torch.profiler import profile,ProfilerActivity
DEV="cuda";Hq,Hkv,d=NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM;topk=INDEX_TOPK_BLOCKS
seq_kv=16384;nblk=seq_kv//128;sel=min(topk,nblk)
torch.manual_seed(0);q=torch.randn(1,Hq,d,device=DEV,dtype=DTYPE)
k=torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE);v=torch.randn(nblk,128,Hkv,d,device=DEV,dtype=DTYPE)
sb=sorted(range(nblk-sel,nblk));ids=torch.tensor(sb,device=DEV,dtype=torch.int32).view(1,-1);bt=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,-1)
W4=int(os.environ.get("W","2"));CH=int(os.environ.get("CH","32"))
def r(): return decode.forward_sparse_decode_p128(q,k,v,bt,ids,float(SCALE),seq_kv,CH,W4)
for _ in range(30): r()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
  for _ in range(200): r()
  torch.cuda.synchronize()
bd={}
for e in p.key_averages():
  if e.self_device_time_total<=0: continue
  k2="part" if "partial" in e.key else ("merge" if "merge" in e.key else e.key[:18])
  bd[k2]=bd.get(k2,0)+e.self_device_time_total/200
print("W4=%d CH=%d: part=%.2f merge=%.2f tot=%.2f"%(W4,CH,bd.get("part",0),bd.get("merge",0),sum(bd.values())))
