# SPDX-License-Identifier: MIT
"""Empirical: does our topk_select match Triton minimax_m3_index_decode on a
MIXED-seq-length decode batch (the real serving case)?

Builds an index-K paged cache + idx_q, runs the REAL Triton fused decode
score+topk as ground truth, then runs the Triton SCORE alone + our topk_select
and compares the selected block SET per (head, query). M3 config: topk=16,
init=0, local=1, num_index_heads(per worker)=1, head_dim=128, block=128.
"""
from __future__ import annotations
import os, sys
import torch

CSRC = os.path.join(os.path.dirname(__file__), "kernels")


def build_topk():
    from torch.utils.cpp_extension import load
    return load(
        name="sm120_sparse_topk",
        sources=[os.path.join(CSRC, "sm120_sparse_topk.cu")],
        extra_include_paths=[CSRC],
        extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f", "-O3",
                           "-std=c++17", "--expt-relaxed-constexpr"],
        verbose=False,
    )


def sets_from(idx_ht):  # [H, Q, topk] -> list per (h,q)
    out = {}
    H, Q, T = idx_ht.shape
    for h in range(H):
        for q in range(Q):
            out[(h, q)] = {int(b) for b in idx_ht[h, q].tolist() if b >= 0}
    return out


def run_case(tk, seq_lens_list, H, seed, INIT, LOCAL):
    torch.manual_seed(seed)
    dev = "cuda"
    from vllm.models.minimax_m3.common.ops.index_topk import (
        SPARSE_BLOCK_SIZE, minimax_m3_index_decode, _decode_index_score_kernel,
    )
    BS = SPARSE_BLOCK_SIZE
    D = 128
    TOPK = 16
    SCALE = 1.0 / (D ** 0.5)
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
    ref = minimax_m3_index_decode(
        idx_q, kcache, bt, seq_lens, max_seq, TOPK, INIT, LOCAL, H, SCALE, 1)
    score_block_stride = ((max_block + 15) // 16) * 16
    score = torch.full((H, R, score_block_stride), float("-inf"),
                       device=dev, dtype=torch.float32)
    TARGET_GRID, MAXC = 4096, 256
    target = max(1, min(MAXC, TARGET_GRID // max(1, R * H)))
    nchunks = 1 << (target.bit_length() - 1)
    from vllm.platforms import current_platform
    use_pdl = current_platform.is_arch_support_pdl()
    pdl = {"launch_pdl": True} if use_pdl else {}
    _decode_index_score_kernel[(R, nchunks)](
        idx_q, kcache, score, bt, seq_lens, H, D, INIT, LOCAL, SCALE, 1,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        kcache.stride(0), kcache.stride(1), kcache.stride(2),
        score.stride(0), score.stride(1), score.stride(2),
        bt.stride(0), BLOCK_SIZE_K=BS, num_kv_chunks=nchunks, USE_PDL=use_pdl, **pdl)
    max_score = score.permute(0, 2, 1).contiguous()  # [H,K,R]
    nv = torch.tensor([(s + BS - 1) // BS for s in seq_lens_list],
                      device=dev, dtype=torch.int32)
    ours = tk.topk_select_varlen(max_score, nv, int(INIT), int(LOCAL))
    ours_hqt = ours.permute(1, 0, 2).contiguous()
    ref_sets = sets_from(ref.cpu())
    our_sets = sets_from(ours_hqt.cpu())
    bad = []
    for k in ref_sets:
        if ref_sets[k] != our_sets[k]:
            bad.append((k, seq_lens_list[k[1]],
                        sorted(ref_sets[k] - our_sets[k])[:4],
                        sorted(our_sets[k] - ref_sets[k])[:4]))
    return len(ref_sets), len(bad), bad


def stress():
    tk = build_topk()
    import random
    grids = [
        ("orig8", [517, 1031, 4096, 60001, 128, 16000, 2049, 33333], 1, 0, 1),
        ("single", [4097], 1, 0, 1),
        ("uniform4", [8192, 8192, 8192, 8192], 1, 0, 1),
        ("trivial(<=16blk)", [128, 256, 384, 512, 2048], 1, 0, 1),
        ("local2", [517, 4096, 60001, 16000], 1, 0, 2),
        ("init1local1", [517, 4096, 60001, 16000, 2049], 1, 1, 1),
        ("H4heads", [517, 4096, 60001, 16000], 4, 0, 1),
        ("tinymix", [129, 130, 200, 1500], 1, 0, 1),
    ]
    all_ok = True
    for name, sl, H, ini, loc in grids:
        for seed in range(3):
            ntot, nbad, bad = run_case(tk, sl, H, seed, ini, loc)
            ok = (nbad == 0)
            all_ok = all_ok and ok
            tag = "OK" if ok else f"FAIL({nbad}/{ntot})"
            print(f"  {name:18s} seed={seed} H={H} init={ini} local={loc} rows={ntot} -> {tag}")
            if not ok:
                for b in bad[:4]:
                    print(f"      {b}")
    print("STRESS VERDICT:", "ALL SET-EXACT" if all_ok else "FAILURES FOUND")
    return 0 if all_ok else 1


def main():
    torch.manual_seed(0)
    dev = "cuda"
    from vllm.models.minimax_m3.common.ops.index_topk import (
        SPARSE_BLOCK_SIZE, minimax_m3_index_decode, _decode_index_score_kernel,
    )
    import triton
    BS = SPARSE_BLOCK_SIZE  # 128
    H = 1                   # per-worker index heads (TP4: 4//4)
    D = 128
    TOPK, INIT, LOCAL = 16, 0, 1
    SCALE = 1.0 / (D ** 0.5)
    tk = build_topk()

    # Mixed decode batch: requests with DIFFERENT seq lens.
    seq_lens_list = [517, 1031, 4096, 60001, 128, 16000, 2049, 33333]
    R = len(seq_lens_list)
    seq_lens = torch.tensor(seq_lens_list, device=dev, dtype=torch.int32)
    max_seq = int(max(seq_lens_list))
    max_block = (max_seq + BS - 1) // BS
    # paged index-K cache
    total_blocks = sum((s + BS - 1) // BS for s in seq_lens_list) + 8
    kcache = torch.randn(total_blocks, BS, D, device=dev, dtype=torch.bfloat16) * 0.1
    # block_table: contiguous pages per request
    max_logical = max_block
    bt = torch.zeros(R, max_logical, device=dev, dtype=torch.int32)
    nxt = 0
    for r, s in enumerate(seq_lens_list):
        nb = (s + BS - 1) // BS
        bt[r, :nb] = torch.arange(nxt, nxt + nb, device=dev, dtype=torch.int32)
        nxt += nb
    idx_q = torch.randn(R, H, D, device=dev, dtype=torch.bfloat16) * 0.5

    # ---- GROUND TRUTH: fused Triton decode score+topk ----
    ref = minimax_m3_index_decode(
        idx_q, kcache, bt, seq_lens, max_seq, TOPK, INIT, LOCAL, H, SCALE, 1,
    )  # [H, R, topk]

    # ---- Triton SCORE alone (reuse the decode score kernel) ----
    score_block_stride = ((max_block + 15) // 16) * 16
    score = torch.full((H, R, score_block_stride), float("-inf"),
                       device=dev, dtype=torch.float32)
    TARGET_GRID, MAXC = 4096, 256
    target = max(1, min(MAXC, TARGET_GRID // max(1, R * H)))
    nchunks = 1 << (target.bit_length() - 1)
    from vllm.platforms import current_platform
    use_pdl = current_platform.is_arch_support_pdl()
    pdl = {"launch_pdl": True} if use_pdl else {}
    _decode_index_score_kernel[(R, nchunks)](
        idx_q, kcache, score, bt, seq_lens, H, D, INIT, LOCAL, SCALE, 1,
        idx_q.stride(0), idx_q.stride(1), idx_q.stride(2),
        kcache.stride(0), kcache.stride(1), kcache.stride(2),
        score.stride(0), score.stride(1), score.stride(2),
        bt.stride(0), BLOCK_SIZE_K=BS, num_kv_chunks=nchunks, USE_PDL=use_pdl, **pdl,
    )

    # our topk wants [H, K, Q]; score is [H, Q, K] -> transpose
    max_score = score.permute(0, 2, 1).contiguous()  # [H, K, R]

    def compare(ours_hqt, label):
        ref_sets = sets_from(ref.cpu())
        our_sets = sets_from(ours_hqt.cpu())
        nmatch = ntot = 0
        mism = []
        for k in ref_sets:
            ntot += 1
            if ref_sets[k] == our_sets[k]:
                nmatch += 1
            else:
                mism.append((k, seq_lens_list[k[1]], len(ref_sets[k]), len(our_sets[k]),
                             sorted(ref_sets[k] - our_sets[k])[:5],
                             sorted(our_sets[k] - ref_sets[k])[:5]))
        print(f"[{label}] MIXED-batch set match: {nmatch}/{ntot}")
        for k, s, nr, no, miss, extra in mism[:12]:
            print(f"    (h,q)={k} seq={s} nblk={(s+BS-1)//BS} ref|={nr} our|={no} "
                  f"ref-only={miss} our-only={extra}")
        return nmatch == ntot

    # --- scalar (Phase-1, known to diverge) ---
    ours_scalar = tk.topk_select(max_score, int(max_block), int(INIT), int(LOCAL))
    ok_scalar = compare(ours_scalar.permute(1, 0, 2).contiguous(), "SCALAR")

    # --- VARLEN: per-query num_valid = ceil(seq/128) ---
    nv = torch.tensor([(s + BS - 1) // BS for s in seq_lens_list],
                      device=dev, dtype=torch.int32)  # [R] (== total_q, qlen=1)
    ours_var = tk.topk_select_varlen(max_score, nv, int(INIT), int(LOCAL))
    ok_var = compare(ours_var.permute(1, 0, 2).contiguous(), "VARLEN")

    print("VERDICT scalar:", "SET-EXACT" if ok_scalar else "DIVERGES (expected)")
    print("VERDICT varlen:", "SET-EXACT" if ok_var else "DIVERGES")
    return 0 if ok_var else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stress":
        sys.exit(stress())
    sys.exit(main())
