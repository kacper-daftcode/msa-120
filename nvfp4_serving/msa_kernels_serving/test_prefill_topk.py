# SPDX-License-Identifier: MIT
"""Validate the PREFILL indexer topk: minimax_m3_index_score + our varlen topk
vs the Triton ground truth (minimax_m3_index_score + minimax_m3_index_topk).

Mirrors the metadata a prefill batch produces (cu_seqlens_q rebased, context_lens
= cached tokens). Each query token has its own causal kv_len = context + j + 1.
"""
from __future__ import annotations
import os, sys
import torch

CSRC = os.path.join(os.path.dirname(__file__), "kernels")


def build_topk():
    from torch.utils.cpp_extension import load
    return load(name="sm120_sparse_topk",
                sources=[os.path.join(CSRC, "sm120_sparse_topk.cu")],
                extra_include_paths=[CSRC],
                extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f", "-O3",
                                   "-std=c++17", "--expt-relaxed-constexpr"], verbose=False)


def sets_from(idx_ht):
    out = {}
    H, Q, T = idx_ht.shape
    for h in range(H):
        for q in range(Q):
            out[(h, q)] = {int(b) for b in idx_ht[h, q].tolist() if b >= 0}
    return out


def num_valid(cu_q, ctx, total_q, BS):
    dev = cu_q.device
    t = torch.arange(total_q, device=dev, dtype=torch.int32)
    req = torch.searchsorted(cu_q.to(torch.int32), t, right=True) - 1
    req = req.clamp_min(0)
    local_j = t - cu_q.to(torch.int32)[req]
    kv_len = ctx.to(torch.int32)[req] + local_j + 1
    return ((kv_len.clamp_min(0) + BS - 1) // BS).to(torch.int32)


def run(tk, reqs, seed, INIT, LOCAL):
    # reqs: list of (context_len, query_len)
    torch.manual_seed(seed)
    dev = "cuda"
    from vllm.models.minimax_m3.common.ops.index_topk import (
        SPARSE_BLOCK_SIZE, minimax_m3_index_score, minimax_m3_index_topk,
    )
    BS = SPARSE_BLOCK_SIZE
    H, D, TOPK = 1, 128, 16
    SCALE = 1.0 / (D ** 0.5)

    qlens = [q for _, q in reqs]
    ctxs = [c for c, _ in reqs]
    seqs = [c + q for c, q in reqs]
    total_q = sum(qlens)
    R = len(reqs)
    cu_q = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()),
                        device=dev, dtype=torch.int32)
    ctx = torch.tensor(ctxs, device=dev, dtype=torch.int32)
    seq_lens = torch.tensor(seqs, device=dev, dtype=torch.int32)
    max_seq = max(seqs)
    max_query_len = max(qlens)
    max_block = (max_seq + BS - 1) // BS
    total_blocks = sum((s + BS - 1) // BS for s in seqs) + 8
    kcache = torch.randn(total_blocks, BS, D, device=dev, dtype=torch.bfloat16) * 0.1
    bt = torch.zeros(R, max_block, device=dev, dtype=torch.int32)
    nxt = 0
    for r, s in enumerate(seqs):
        nb = (s + BS - 1) // BS
        bt[r, :nb] = torch.arange(nxt, nxt + nb, device=dev, dtype=torch.int32)
        nxt += nb
    idx_q = torch.randn(total_q, H, D, device=dev, dtype=torch.bfloat16) * 0.5

    score = minimax_m3_index_score(
        idx_q, kcache, bt, cu_q, seq_lens, ctx, max_query_len, max_seq, H, SCALE)
    # Triton ground-truth topk
    ref = minimax_m3_index_topk(score.clone(), cu_q, ctx, max_query_len,
                                TOPK, INIT, LOCAL)  # [H, total_q, topk]

    # ours: mask out-of-range -inf + varlen topk
    nv = num_valid(cu_q, ctx, total_q, BS)
    Kdim = score.shape[2]
    kidx = torch.arange(Kdim, device=dev).view(1, 1, -1)
    smask = kidx >= nv.view(1, -1, 1)
    score_m = score.masked_fill(smask, float("-inf"))
    max_score = score_m.permute(0, 2, 1).contiguous()
    ours = tk.topk_select_varlen(max_score, nv, int(INIT), int(LOCAL))
    ours_hqt = ours.permute(1, 0, 2).contiguous()

    rs, os_ = sets_from(ref.cpu()), sets_from(ours_hqt.cpu())
    bad = [(k, rs[k] ^ os_[k]) for k in rs if rs[k] != os_[k]]
    return len(rs), len(bad), bad


def main():
    tk = build_topk()
    grids = [
        ("chunk-prefill", [(0, 512), (0, 1024)], 0, 1),
        ("ctx+chunk", [(2000, 256), (8000, 512), (100, 700)], 0, 1),
        ("single-long", [(0, 4096)], 0, 1),
        ("mixed-ctx", [(517, 200), (4096, 64), (16000, 128), (30, 900)], 0, 1),
        ("local2", [(2000, 300), (8000, 200)], 0, 2),
        ("init1local1", [(2000, 300), (8000, 200), (100, 600)], 1, 1),
    ]
    all_ok = True
    for name, reqs, ini, loc in grids:
        for seed in range(3):
            ntot, nbad, bad = run(tk, reqs, seed, ini, loc)
            ok = nbad == 0
            all_ok = all_ok and ok
            print(f"  {name:16s} seed={seed} init={ini} local={loc} rows={ntot} -> "
                  f"{'OK' if ok else f'FAIL({nbad})'}")
            if not ok:
                for k, diff in bad[:4]:
                    print(f"      (h,q)={k} symdiff={sorted(diff)[:6]}")
    print("PREFILL VERDICT:", "ALL SET-EXACT" if all_ok else "FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
