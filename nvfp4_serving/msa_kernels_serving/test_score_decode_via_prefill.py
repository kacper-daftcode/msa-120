# SPDX-License-Identifier: MIT
"""Check: does the PUBLIC minimax_m3_index_score (prefill score op) produce the
same per-block max-scores as the fused decode path, when fed decode-shaped
inputs (qlen=1, cu_seqlens_q=arange, prefix_lens=seq_lens-1)? If so, we can use
ONE public score op for both decode+prefill and run our varlen topk on it,
without touching the private decode-score kernel.

Compares the resulting topk SET (our varlen topk on the prefill-score) vs the
fused Triton decode ground truth, on a mixed batch.
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


def main():
    torch.manual_seed(0)
    dev = "cuda"
    from vllm.models.minimax_m3.common.ops.index_topk import (
        SPARSE_BLOCK_SIZE, minimax_m3_index_decode, minimax_m3_index_score,
    )
    BS = SPARSE_BLOCK_SIZE
    H, D = 1, 128
    TOPK, INIT, LOCAL = 16, 0, 1
    SCALE = 1.0 / (D ** 0.5)
    tk = build_topk()

    seq_lens_list = [517, 1031, 4096, 60001, 128, 16000, 2049, 33333]
    R = len(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, device=dev, dtype=torch.int32)
    max_seq = int(max(seq_lens_list))
    max_block = (max_seq + BS - 1) // BS
    total_blocks = sum((s + BS - 1) // BS for s in seq_lens_list) + 8
    kcache = torch.randn(total_blocks, BS, D, device=dev, dtype=torch.bfloat16) * 0.1
    bt = torch.zeros(R, max_block, device=dev, dtype=torch.int32)
    nxt = 0
    for r, s in enumerate(seq_lens_list):
        nb = (s + BS - 1) // BS
        bt[r, :nb] = torch.arange(nxt, nxt + nb, device=dev, dtype=torch.int32)
        nxt += nb
    idx_q = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16) * 0.5

    # Ground truth: fused decode score+topk.
    ref = minimax_m3_index_decode(
        idx_q, kcache, bt, seq_lens, max_seq, TOPK, INIT, LOCAL, H, SCALE, 1)

    # Decode shaped as "prefill" with 1 query/request:
    #   cu_seqlens_q = [0,1,2,...,R]; prefix_lens = seq_lens - 1 (KV context before
    #   the single query); seq_lens (KV total) = seq_lens. The score op's
    #   valid_blocks for query 0 in each request = ceil(seq_len/128).
    cu_q = torch.arange(R + 1, device=dev, dtype=torch.int32)
    prefix = (seq_lens - 1).to(torch.int32)
    score = minimax_m3_index_score(
        idx_q, kcache, bt, cu_q, seq_lens, prefix,
        1,  # max_query_len
        max_seq, H, SCALE,
    )  # [H, total_q=R, score_block_stride]
    # score op leaves out-of-range blocks UNWRITTEN (torch.empty) -> fill -inf
    # for slots >= each query's valid_blocks so our varlen topk sees -inf there.
    Kdim = score.shape[2]
    kidx = torch.arange(Kdim, device=dev).view(1, 1, -1)
    nv = ((seq_lens.to(torch.int64) + BS - 1) // BS).view(1, R, 1)
    mask = kidx >= nv  # [1,R,Kdim] out-of-range
    score = score.masked_fill(mask, float("-inf"))

    max_score = score.permute(0, 2, 1).contiguous()  # [H,K,R]
    nv_t = ((seq_lens + BS - 1) // BS).to(torch.int32)
    ours = tk.topk_select_varlen(max_score, nv_t, int(INIT), int(LOCAL))
    ours_hqt = ours.permute(1, 0, 2).contiguous()

    ref_sets = sets_from(ref.cpu())
    our_sets = sets_from(ours_hqt.cpu())
    nbad = 0
    for k in ref_sets:
        if ref_sets[k] != our_sets[k]:
            nbad += 1
            print(f"  MISMATCH (h,q)={k} seq={seq_lens_list[k[1]]} "
                  f"ref-only={sorted(ref_sets[k]-our_sets[k])[:4]} "
                  f"our-only={sorted(our_sets[k]-ref_sets[k])[:4]}")
    print(f"prefill-score + varlen-topk vs fused-decode: {len(ref_sets)-nbad}/{len(ref_sets)}")
    print("VERDICT:", "SET-EXACT" if nbad == 0 else "DIVERGES")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
