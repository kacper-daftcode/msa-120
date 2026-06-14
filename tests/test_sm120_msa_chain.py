"""End-to-end MSA sparse-attention chain on SM120 (M6):
   block-score -> top-k select -> block-sparse forward, validated vs torch.
"""
import os, math, torch, numpy as np
from torch.utils.cpp_extension import load

C = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")
FLAGS = ["-gencode=arch=compute_120f,code=sm_120f", "-O3", "-std=c++17", "--expt-relaxed-constexpr"]
print("Building the 3 MSA-stage extensions...")
score = load(name="sm120_block_score", sources=[os.path.join(C, "sm120_block_score.cu")],
             extra_cuda_cflags=FLAGS, verbose=False)
topk = load(name="sm120_sparse_topk", sources=[os.path.join(C, "sm120_sparse_topk.cu")],
            extra_include_paths=[C], extra_cuda_cflags=FLAGS, verbose=False)
fwd = load(name="sm120_fmha_sparse_v1", sources=[os.path.join(C, "sm120_fmha_sparse.cu")],
           extra_cuda_cflags=FLAGS, verbose=False)
print("Built.\n")
BS = 64       # KV block size (matches forward BLK_N)
TOPK = 16

def torch_scores(q, k, scale, h=0):
    return (q[:, h, :].float() @ k[:, h, :].float().T) * scale   # [S, Sk]

def run(name, S, Sk, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, 1, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Sk, 1, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Sk, 1, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(128)
    nblk = (Sk + BS - 1) // BS

    # --- STAGE 1: block-score (kernel) ---
    msc = score.block_max_score(q, k, scale, BS)              # [1, nblk, S]
    # --- STAGE 2: top-k select (kernel) ---
    idx = topk.topk_select(msc.contiguous(), nblk, 0, 0)      # [S, 1, 16]
    block_ids = idx[:, 0, :].contiguous()                     # [S, 16] per-query
    # --- STAGE 3: block-sparse forward (kernel), per-query, non-causal ---
    o, lse = fwd.forward_sparse(q, k, v, block_ids, scale, False)

    # --- torch reference: independent score->top16->masked attention ---
    sc = torch_scores(q, k, scale)                            # [S, Sk] fp32
    # block max-pool reference
    bmax = torch.stack([sc[:, b*BS:min((b+1)*BS, Sk)].max(dim=1).values for b in range(nblk)], dim=1)  # [S, nblk]
    ref_idx = torch.argsort(-bmax, dim=1)[:, :TOPK]           # [S, 16]
    # (a) indexer composition: do kernel-selected blocks match torch top16?
    sel_match = all(sorted(block_ids[i].tolist()) == sorted(ref_idx[i].tolist()) for i in range(S))
    # (b) forward composition: masked attention over kernel-selected blocks
    mask = torch.zeros(S, Sk, dtype=torch.bool, device="cuda")
    for i in range(S):
        for b in block_ids[i].tolist():
            if b < 0: continue
            mask[i, b*BS:min((b+1)*BS, Sk)] = True
    sc_m = sc.masked_fill(~mask, float("-inf"))
    attn = torch.nan_to_num(torch.softmax(sc_m, dim=-1), 0.0)
    ref_o = (attn @ v[:, 0, :].float())
    rms = ((o[:, 0, :].float() - ref_o).pow(2).mean().sqrt() / ref_o.pow(2).mean().sqrt()).item()
    ok = sel_match and rms < 0.05
    print(f"  {'✓' if ok else '✗'} {name:20} S={S} Sk={Sk} blocks={nblk} top{TOPK}  "
          f"select_match={sel_match}  fwd_rms={rms:.4f}")
    return ok

print(f"GPU: {torch.cuda.get_device_name(0)}\n")
print("="*64); print("M6: full MSA chain  score -> topk -> sparse-forward"); print("="*64)
res = []
res.append(run("32 blk, top16", S=128, Sk=2048, seed=1))
res.append(run("48 blk, top16", S=256, Sk=3072, seed=2))
res.append(run("64 blk, top16", S=192, Sk=4096, seed=3))
print("\n" + "="*64)
print(">>> CHAIN OK <<<" if all(res) else ">>> CHAIN FAILURES <<<")
