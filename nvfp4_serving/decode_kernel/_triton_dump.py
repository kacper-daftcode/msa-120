import os
os.environ["TRITON_CACHE_DIR"]="/work/decode_kernel/nsys_stats/triton_cache"
os.environ["TRITON_KERNEL_DUMP"]="1"
os.environ["TRITON_DUMP_DIR"]="/work/decode_kernel/nsys_stats/triton_dump"
import torch, sys
sys.path.insert(0,"/work/kernel_profiling")
from bench_common import (NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM,INDEX_TOPK_BLOCKS,SCALE,DTYPE)
from vllm.models.minimax_m3.common.ops.sparse_attn import minimax_m3_sparse_attn_decode
DEV="cuda"; Hq,Hkv,d=NUM_Q_HEADS,NUM_KV_HEADS,HEAD_DIM; topk=INDEX_TOPK_BLOCKS; q_len=1
seq_kv=16384; nblk=seq_kv//128; sel=min(topk,nblk)
torch.manual_seed(0)
q=torch.randn(q_len,Hq,d,device=DEV,dtype=DTYPE)
kvT=torch.randn(nblk,2,128,Hkv,d,device=DEV,dtype=DTYPE)
btt=torch.arange(nblk,device=DEV,dtype=torch.int32).view(1,nblk)
sel_blocks=sorted(range(nblk-sel,nblk))
sel_t=torch.tensor(sel_blocks+[-1]*(topk-len(sel_blocks)),device=DEV,dtype=torch.int32)
tki=sel_t.view(1,1,topk).expand(Hkv,q_len,topk).contiguous()
seq_lens=torch.tensor([seq_kv],device=DEV,dtype=torch.int32)
out=torch.empty(q_len,Hq,d,device=DEV,dtype=DTYPE)
for _ in range(3):
    minimax_m3_sparse_attn_decode(q,kvT,tki,btt,seq_lens,num_kv_heads=Hkv,sm_scale=float(SCALE),output=out,decode_query_len=q_len)
torch.cuda.synchronize()
print("triton ran ok")
