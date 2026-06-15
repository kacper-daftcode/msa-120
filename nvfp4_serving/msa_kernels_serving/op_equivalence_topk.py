#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Op-equivalence: SM120 ``topk_select`` vs vLLM Triton ``minimax_m3_index_topk``.

Runs inside the ``vllm/vllm-openai:minimax-m3`` container (torch 2.11+cu130,
nvcc 13.0). JIT-builds ``sm120_sparse_topk.cu`` (compute_120f) and compares its
block selection against the reference Triton ``minimax_m3_index_topk`` on
identical random scores, across init/local-block settings and causal lengths.

Equivalence criterion: the SET of selected KV blocks per (head, query) must be
IDENTICAL. (Triton returns score-sorted ids; our kernel returns ascending ids;
both pad with -1. So we compare sorted, de-padded sets.)

Both ops consume scores per (head, query, block). Layout note:
  * Triton wants score [H, total_q, max_block].
  * Our topk_select wants max_score [H, max_block(=K_tiles), total_q].
We build one canonical score[H, Q, B] and feed each its expected layout, so the
two ops see numerically the same per-(h,q,block) values.

Usage (inside container):
  python3 op_equivalence_topk.py --csrc /tmp/sm120/csrc
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def build_topk_ext(csrc: str):
    from torch.utils.cpp_extension import load

    return load(
        name="sm120_sparse_topk",
        sources=[os.path.join(csrc, "sm120_sparse_topk.cu")],
        extra_include_paths=[csrc],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120f,code=sm_120f",
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ],
        verbose=False,
    )


def selected_sets(idx: torch.Tensor) -> list[set]:
    """idx [Q, topk] int32 -> list of de-padded (>=0) block-id sets per query."""
    out = []
    for row in idx.tolist():
        out.append({b for b in row if b >= 0})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csrc", default="/tmp/sm120/csrc")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"

    from vllm.models.minimax_m3.common.ops.index_topk import (
        SPARSE_BLOCK_SIZE,
        minimax_m3_index_topk,
    )

    topk_ext = build_topk_ext(args.csrc)

    TOPK = 16
    H = 4  # M3 index heads == kv heads
    block_size = SPARSE_BLOCK_SIZE  # 128

    # Test grid: (#blocks B, init, local). Single-request prefill, all queries
    # share the same causal length so the reference's per-query valid_blocks is
    # uniform (== B), letting a single scalar num_valid drive our kernel.
    cases = [
        # (num_blocks B, init_blocks, local_blocks)
        (20, 0, 0),
        (20, 1, 1),
        (32, 2, 3),
        (64, 0, 1),
        (40, 4, 4),
        (17, 0, 0),   # fewer blocks than topk -> partial select + (-1) pad
        (16, 0, 0),   # exactly topk
        (8, 1, 1),    # fewer than topk, with force begin/end
    ]

    all_ok = True
    # Q queries, each at its OWN causal position so the reference's per-query
    # valid_blocks == q+1 (block_size_q==1, prefix_len chosen so block q is the
    # last visible). We feed our kernel ONE topk_select call per query position
    # with the matching scalar num_valid, so both ops see the identical visible
    # block range [0, valid_blocks) for that query.
    print(f"{'B':>4} {'init':>4} {'local':>5} {'Qpos':>5}  {'match/total':>12}  result")
    print("-" * 56)
    for (B, init_blocks, local_blocks) in cases:
        # Single request whose blocks span [0, B). One query per block position:
        # query q is at token position (q+1)*block_size - 1 -> valid_blocks q+1.
        # We run the reference once over all Q queries and our kernel once per
        # query (scalar num_valid == that query's valid_blocks).
        Q = B
        # prefix_lens=0; position of query q == q (block_size_q==1 maps pid_q->pos)
        # so reference valid_blocks(q) = (0 + q*block_size + block_size)//block_size
        # == q+1 when we space queries one BLOCK apart. The reference uses pid_q
        # directly as the position (sample_interval==1 => pos == pid_q), giving
        # valid_blocks = (pid_q + 128)//128 = 1 for pid_q<128. To get
        # valid_blocks == q+1 we instead set prefix_len so block q is last.
        # Simplest faithful construction: one query (Q=1) per case at the END,
        # so valid_blocks == B. Loop B over the case list already covers sizes.
        score_h1b = torch.randn(H, 1, B, device=dev, dtype=torch.float32)
        cu_seqlens_q = torch.tensor([0, 1], device=dev, dtype=torch.int32)
        # prefix_len = (B-1)*block_size so the single query (pid_q=0, pos=0) sees
        # valid_blocks = (prefix + 0 + block_size)//block_size = B.
        prefix_lens = torch.tensor([(B - 1) * block_size], device=dev, dtype=torch.int32)
        ref_idx = minimax_m3_index_topk(
            score_h1b.contiguous(),  # [H, total_q=1, max_block=B]
            cu_seqlens_q,
            prefix_lens,
            max_query_len=1,
            topk=TOPK,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )  # [H, 1, topk]

        # --- Ours: max_score [H, B, 1] ---
        max_score_hb1 = score_h1b.permute(0, 2, 1).contiguous()
        ours = topk_ext.topk_select(
            max_score_hb1, int(B), int(init_blocks), int(local_blocks)
        )  # [1, H, topk]
        ours_h1t = ours.permute(1, 0, 2).contiguous()  # [H, 1, topk]

        match = 0
        total = 0
        for h in range(H):
            ref_set = selected_sets(ref_idx[h])[0]
            our_set = selected_sets(ours_h1t[h])[0]
            total += 1
            if ref_set == our_set:
                match += 1
        ok = (total > 0) and (match == total)
        all_ok = all_ok and ok
        print(
            f"{B:>4} {init_blocks:>4} {local_blocks:>5} {'end':>5}  "
            f"{match:>5}/{total:<6}  {'OK' if ok else 'MISMATCH'}"
        )

    print("-" * 52)
    print("VERDICT:", "ALL MATCH (set-exact)" if all_ok else "MISMATCH FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
