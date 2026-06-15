#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Head-to-head per-op latency: SM120 hand-written MSA kernels vs vLLM Triton.

Runs inside vllm/vllm-openai:minimax-m3 on CUDA_VISIBLE_DEVICES=0, alongside the
marlin prod container (small tensors only). Three ops, M3 shapes, decode +
prefill regimes. Emits JSON to results/ and prints a markdown table.

Ops:
  1. top-k block select : ours topk_select       vs Triton minimax_m3_index_topk
  2. indexer block score: ours block_scores      vs Triton minimax_m3_index_score
     (entrypoint asymmetry: ours = project+norm+rope+score from hidden states;
      Triton = score-only from pre-projected idx_q. Timed end-to-end, asymmetry
      noted in the report.)
  3. sparse paged attend: ours forward_sparse_paged vs Triton minimax_m3_sparse_attn
     / minimax_m3_sparse_attn_decode (each fed its NATIVE page layout: ours=64,
      Triton=128; same logical problem).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import (  # noqa: E402
    DTYPE, HEAD_DIM, INDEX_BLOCK_SIZE, INDEX_HEAD_DIM, INDEX_INIT_BLOCKS,
    INDEX_LOCAL_BLOCKS, INDEX_N_HEADS, INDEX_SCALE, INDEX_TOPK_BLOCKS,
    NUM_KV_HEADS, NUM_Q_HEADS, RMS_EPS, ROPE_THETA, ROTARY_DIM, SCALE,
    cuda_time, cuda_time_batched,
)

CSRC = os.environ.get("SM120_MSA_CSRC", "/work/msa_kernels_serving/kernels")
DEV = "cuda"
SPARSE_BLOCK = INDEX_BLOCK_SIZE  # 128


# --------------------------------------------------------------------------
# Build our kernels (JIT) with the validated recipe.
# --------------------------------------------------------------------------
def prepare_build_env():
    cu13 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    inc = "/usr/local/cuda/targets/x86_64-linux/include"
    if os.path.isdir(cu13) and os.path.isdir(inc):
        for h in ("cusparse.h", "cusolverDn.h", "cusolver_common.h"):
            dst, src = os.path.join(inc, h), os.path.join(cu13, h)
            if not os.path.exists(dst) and os.path.exists(src):
                try:
                    os.symlink(src, dst)
                except OSError:
                    pass


_FLAGS = ["-gencode=arch=compute_120f,code=sm_120f", "-O3", "-std=c++17",
          "--expt-relaxed-constexpr"]


def build(name, src):
    from torch.utils.cpp_extension import load
    prepare_build_env()
    return load(name=name, sources=[os.path.join(CSRC, src)],
                extra_include_paths=[CSRC], extra_cuda_cflags=_FLAGS, verbose=False)


# --------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------
DECODE_CTX = [4096, 16384, 65536]   # seq_len_kv, q_len=1
PREFILL_QLEN = [512, 2048]


def regimes():
    out = []
    for ctx in DECODE_CTX:
        out.append(("decode", 1, ctx))
    for ql in PREFILL_QLEN:
        out.append(("prefill", ql, ql))   # full causal; kv == q
    return out


# ==========================================================================
# OP 1: top-k block select
# ==========================================================================
def bench_topk(topk_ext, results):
    from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_topk
    H = INDEX_N_HEADS
    for kind, q_len, seq_kv in regimes():
        nblk = (seq_kv + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        total_q = q_len
        # canonical scores [H, total_q, nblk]
        torch.manual_seed(0)
        score = torch.randn(H, total_q, nblk, device=DEV, dtype=torch.float32)

        # ---- Triton ----
        cu_seqlens_q = torch.tensor([0, total_q], device=DEV, dtype=torch.int32)
        if kind == "decode":
            prefix_lens = torch.tensor([seq_kv - q_len], device=DEV, dtype=torch.int32)
        else:
            prefix_lens = torch.tensor([0], device=DEV, dtype=torch.int32)
        score_t = score.contiguous()

        def run_triton():
            return minimax_m3_index_topk(
                score_t, cu_seqlens_q, prefix_lens, max_query_len=total_q,
                topk=INDEX_TOPK_BLOCKS, init_blocks=INDEX_INIT_BLOCKS,
                local_blocks=INDEX_LOCAL_BLOCKS)

        # ---- Ours: max_score [H, nblk, total_q] ----
        score_ours = score.permute(0, 2, 1).contiguous()
        num_valid = nblk

        def run_ours():
            return topk_ext.topk_select(
                score_ours, int(num_valid), int(INDEX_INIT_BLOCKS),
                int(INDEX_LOCAL_BLOCKS))

        run_triton(); run_ours()
        t_med, *_ = cuda_time(run_triton)
        o_med, *_ = cuda_time(run_ours)
        t_b = cuda_time_batched(run_triton)
        o_b = cuda_time_batched(run_ours)
        results.append(dict(op="topk_select", regime=kind, q_len=q_len,
                            seq_kv=seq_kv, nblk=nblk,
                            our_us=o_b * 1e3, triton_us=t_b * 1e3,
                            our_wall_us=o_med * 1e3, triton_wall_us=t_med * 1e3,
                            ratio=(t_b / o_b) if o_b > 0 else 0.0))
        print(f"[topk]   {kind:7s} q={q_len:<5d} kv={seq_kv:<6d} nblk={nblk:<4d} "
              f"ours={o_b*1e3:8.2f}us triton={t_b*1e3:8.2f}us  ratio={t_b/o_b:.2f}x")


# ==========================================================================
# OP 2: indexer block score (end-to-end, asymmetric -- noted)
# ==========================================================================
def bench_indexer(idx_ext, results):
    from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_score
    H = INDEX_N_HEADS
    d = INDEX_HEAD_DIM
    hidden = 4096  # representative hidden size for the projection GEMM
    # ENTRYPOINT ASYMMETRY (see GOAL note): our block_scores does the FULL
    # project+norm+rope+score over ALL N tokens (q against k over the whole
    # context). It has no score-only-from-pre-projected-idx_q entrypoint. So it
    # is only fairly comparable to Triton's minimax_m3_index_score in PREFILL,
    # where q_len == N == seq_kv (both process every query x every visible
    # block). In DECODE (q_len=1 over a long context) ours must re-score the
    # entire context while Triton scores 1 query -> NOT a fair head-to-head;
    # we record it but flag it 'fair=False'.
    # Our score kernel's smem is BLK_M*nblk*4 bytes; cap N so it stays < ~99KB.
    SMEM_MAX_NBLK = 360  # 64*360*4 = ~92KB < SM120 dynamic smem cap
    for kind, q_len, seq_kv in regimes():
        N = seq_kv          # ours scores a dense [N,128] index-K over all N
        nblk = (N + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        ours_runnable = nblk <= SMEM_MAX_NBLK
        fair = (kind == "prefill")
        torch.manual_seed(0)

        # ---- Ours: full project+norm+rope+score from hidden states ----
        # q,k are hidden states [N, hidden]; q_proj [H*d, hidden]; k_proj [d,hidden]
        run_ours = None
        if ours_runnable:
            q_hs = torch.randn(N, hidden, device=DEV, dtype=DTYPE)
            k_hs = torch.randn(N, hidden, device=DEV, dtype=DTYPE)
            q_proj = torch.randn(H * d, hidden, device=DEV, dtype=DTYPE) * 0.02
            k_proj = torch.randn(d, hidden, device=DEV, dtype=DTYPE) * 0.02
            q_norm = torch.ones(d, device=DEV, dtype=DTYPE)
            k_norm = torch.ones(d, device=DEV, dtype=DTYPE)
            positions = torch.arange(N, device=DEV, dtype=torch.int64)

            def run_ours():
                return idx_ext.block_scores(
                    q_hs, k_hs, q_proj, k_proj, q_norm, k_norm, positions,
                    INDEX_BLOCK_SIZE, H, d, float(INDEX_SCALE), ROTARY_DIM,
                    float(ROPE_THETA), float(RMS_EPS), True, True)

        # ---- Triton: score-only from pre-projected idx_q + paged index-K ----
        idx_q = torch.randn(N, H, d, device=DEV, dtype=DTYPE)
        index_kv_cache = torch.randn(nblk, SPARSE_BLOCK, d, device=DEV, dtype=DTYPE)
        block_table = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, nblk)
        cu_seqlens_q = torch.tensor([0, q_len], device=DEV, dtype=torch.int32)
        seq_lens = torch.tensor([seq_kv], device=DEV, dtype=torch.int32)
        if kind == "decode":
            prefix_lens = torch.tensor([seq_kv - q_len], device=DEV, dtype=torch.int32)
            idx_q = torch.randn(q_len, H, d, device=DEV, dtype=DTYPE)
        else:
            prefix_lens = torch.tensor([0], device=DEV, dtype=torch.int32)
            idx_q = torch.randn(q_len, H, d, device=DEV, dtype=DTYPE)

        def run_triton():
            return minimax_m3_index_score(
                idx_q, index_kv_cache, block_table, cu_seqlens_q, seq_lens,
                prefix_lens, max_query_len=q_len, max_seq_len=seq_kv,
                num_kv_heads=H, sm_scale=float(INDEX_SCALE))

        run_triton()
        t_med, *_ = cuda_time(run_triton)
        t_b = cuda_time_batched(run_triton)
        if ours_runnable:
            run_ours()
            o_med, *_ = cuda_time(run_ours)
            o_b = cuda_time_batched(run_ours)
        else:
            o_med = o_b = float("nan")
        ratio = (t_b / o_b) if (o_b and o_b == o_b and o_b > 0) else float("nan")
        results.append(dict(op="block_score", regime=kind, q_len=q_len,
                            seq_kv=seq_kv, nblk=nblk, asymmetric=True, fair=fair,
                            our_us=o_b * 1e3, triton_us=t_b * 1e3,
                            our_wall_us=o_med * 1e3, triton_wall_us=t_med * 1e3,
                            ratio=ratio,
                            note=("FAIR (prefill: q_len==N, both score all "
                                  "queries; ours also does projection)" if fair
                                  else "NOT FAIR (decode: ours re-scores whole "
                                  "context, triton scores 1 query; ours lacks a "
                                  "score-only entrypoint)")))
        o_str = f"{o_b*1e3:8.2f}us" if o_b == o_b else "   OOM/smem"
        print(f"[score]  {kind:7s} q={q_len:<5d} kv={seq_kv:<6d} N={N:<6d} "
              f"ours={o_str} triton={t_b*1e3:8.2f}us  ratio={ratio:.2f}x "
              f"({'FAIR' if fair else 'asymmetric'})")


# ==========================================================================
# OP 3: sparse paged attend
# ==========================================================================
def bench_attend(paged_ext, results):
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn, minimax_m3_sparse_attn_decode)
    Hq, Hkv, d = NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM
    topk = INDEX_TOPK_BLOCKS
    for kind, q_len, seq_kv in regimes():
        nblk = (seq_kv + SPARSE_BLOCK - 1) // SPARSE_BLOCK
        sel = min(topk, nblk)
        torch.manual_seed(0)

        # Same logical selected-block set for both impls. Selected logical
        # 128-blocks: the last `sel` blocks (covers the local + recent context).
        sel_blocks_128 = list(range(max(0, nblk - sel), nblk))  # ascending ids
        sel_blocks_128 = sorted(sel_blocks_128)
        while len(sel_blocks_128) < topk:
            sel_blocks_128.append(-1)  # pad
        sel_t = torch.tensor(sel_blocks_128, device=DEV, dtype=torch.int32)

        q = torch.randn(q_len, Hq, d, device=DEV, dtype=DTYPE)

        # ---------- Triton (page-128 fused cache) ----------
        # cache [num_blocks, 2, 128, Hkv, d]
        kv_cache = torch.randn(nblk, 2, SPARSE_BLOCK, Hkv, d, device=DEV, dtype=DTYPE)
        block_table_t = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, nblk)
        # topk_idx [Hkv, total_q, topk]: same selected set for every (head,query)
        topk_idx = sel_t.view(1, 1, topk).expand(Hkv, q_len, topk).contiguous()
        seq_lens = torch.tensor([seq_kv], device=DEV, dtype=torch.int32)
        out = torch.empty(q_len, Hq, d, device=DEV, dtype=DTYPE)

        if kind == "decode":
            def run_triton():
                minimax_m3_sparse_attn_decode(
                    q, kv_cache, topk_idx, block_table_t, seq_lens,
                    num_kv_heads=Hkv, sm_scale=float(SCALE), output=out,
                    decode_query_len=q_len)
        else:
            cu_seqlens_q = torch.tensor([0, q_len], device=DEV, dtype=torch.int32)
            prefix_lens = torch.tensor([0], device=DEV, dtype=torch.int32)

            def run_triton():
                minimax_m3_sparse_attn(
                    q, kv_cache, topk_idx, block_table_t, cu_seqlens_q, seq_lens,
                    prefix_lens, max_query_len=q_len, num_kv_heads=Hkv,
                    sm_scale=float(SCALE), output=out)

        # ---------- Ours (page-64 split cache) ----------
        # Each 128-block -> two 64-subtiles. block_ids/block_table in 64-units.
        npage64 = nblk * 2
        k_cache = torch.randn(npage64, 64, Hkv, d, device=DEV, dtype=DTYPE)
        v_cache = torch.randn(npage64, 64, Hkv, d, device=DEV, dtype=DTYPE)
        num_m_blocks = (q_len + 64 - 1) // 64
        # block_table: identity over the 64-pages, rows == num_m_blocks
        bt_row = torch.arange(npage64, device=DEV, dtype=torch.int32)
        block_table_o = bt_row.view(1, npage64).expand(num_m_blocks, npage64).contiguous()
        # block_ids in 64-block units: each selected 128-block -> two 64-blocks.
        ids64 = []
        for b in sel_blocks_128:
            if b < 0:
                ids64 += [-1, -1]
            else:
                ids64 += [2 * b, 2 * b + 1]
        ids64_t = torch.tensor(ids64, device=DEV, dtype=torch.int32)
        block_ids_o = ids64_t.view(1, -1).expand(num_m_blocks, len(ids64)).contiguous()

        # causal=True only valid when the query M-tile starts at absolute pos 0
        # (prefill). Our paged kernel has no decode query-position offset, so a
        # decode token (large abs pos) MUST run causal=False -- the selected
        # blocks are the visible set; the partial last block is masked by seq_k.
        ours_causal = (kind == "prefill")

        def run_ours():
            return paged_ext.forward_sparse_paged(
                q, k_cache, v_cache, block_table_o, block_ids_o,
                float(SCALE), ours_causal, int(seq_kv))

        run_triton(); run_ours()
        o_med, *_ = cuda_time(run_ours)
        t_med, *_ = cuda_time(run_triton)
        o_b = cuda_time_batched(run_ours)
        t_b = cuda_time_batched(run_triton)
        results.append(dict(op="sparse_attend", regime=kind, q_len=q_len,
                            seq_kv=seq_kv, nblk=nblk, sel=sel,
                            our_us=o_b * 1e3, triton_us=t_b * 1e3,
                            our_wall_us=o_med * 1e3, triton_wall_us=t_med * 1e3,
                            ratio=(t_b / o_b) if o_b > 0 else 0.0,
                            note="ours page-64, triton page-128, same logical "
                                 "seqlen/selected-blocks/heads"))
        print(f"[attend] {kind:7s} q={q_len:<5d} kv={seq_kv:<6d} sel={sel:<3d} "
              f"ours={o_b*1e3:8.2f}us triton={t_b*1e3:8.2f}us  ratio={t_b/o_b:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="topk,score,attend")
    ap.add_argument("--out", default="/work/kernel_profiling/results/bench.json")
    args = ap.parse_args()

    print("=== building SM120 kernels (JIT) ===", flush=True)
    ops = args.ops.split(",")
    results = []
    if "topk" in ops:
        topk_ext = build("sm120_sparse_topk", "sm120_sparse_topk.cu")
        print("topk built", flush=True)
        bench_topk(topk_ext, results)
    if "score" in ops:
        idx_ext = build("sm120_indexer", "sm120_indexer.cu")
        print("indexer built", flush=True)
        bench_indexer(idx_ext, results)
    if "attend" in ops:
        paged_ext = build("sm120_fmha_paged", "sm120_fmha_paged.cu")
        print("paged built", flush=True)
        bench_attend(paged_ext, results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
