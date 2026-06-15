#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Correctness of the SERVING decode entrypoint forward_sparse_decode_serving:
  - block_ids [R, Hkv, topk] per-kv-head, seq_lens [R] device tensor.
Validated vs:
  (a) the GOLDEN forward_sparse_paged (page-64, causal=False), rms<1e-2,
  (b) a dense fp32 softmax reference (rms<5e-2).
Per-kv-head selection is exercised by giving each kv-head a DIFFERENT block set.
Run inside vllm/vllm-openai:minimax-m3 on CUDA_VISIBLE_DEVICES=0.

Mount: repo root at /work. Usage:
  python3 /work/msa_kernels_serving/verify_decode_serving.py
"""
import os, sys
import torch
from torch.utils.cpp_extension import load

KDIR = "/work/msa_kernels_serving/kernels"
PAGED_SRC = KDIR

Hq, Hkv, d = 64, 4, 128
g = Hq // Hkv
topk = 16
SCALE = 1.0 / (d ** 0.5)
DTYPE = torch.bfloat16
DEV = "cuda"

def build(name, src, srcdir):
    cu13 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
    inc = "/usr/local/cuda/targets/x86_64-linux/include"
    for h in ("cusparse.h", "cusolverDn.h", "cusolver_common.h"):
        dd = os.path.join(inc, h); s = os.path.join(cu13, h)
        if not os.path.exists(dd) and os.path.exists(s):
            try: os.symlink(s, dd)
            except OSError: pass
    return load(name=name, sources=[os.path.join(srcdir, src)],
                extra_include_paths=[srcdir],
                extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f",
                                   "-O3", "-std=c++17", "--expt-relaxed-constexpr"],
                verbose=False)

paged = build("sm120_fmha_paged", "sm120_fmha_paged.cu", PAGED_SRC)
serve = build("sm120_fmha_decode_serving", "sm120_fmha_decode_serving.cu", KDIR)


def run_case(seq_kv, nsel, split_chunks, R=1, per_head_diff=True, seed=0, layout="NHD"):
    torch.manual_seed(seed)
    nblk = seq_kv // 128
    nsel = min(nsel, nblk)
    k128 = torch.randn(nblk, 128, Hkv, d, device=DEV, dtype=DTYPE)
    v128 = torch.randn(nblk, 128, Hkv, d, device=DEV, dtype=DTYPE)
    # page-64 view of same data (for golden)
    k64 = k128.reshape(nblk, 2, 64, Hkv, d).reshape(nblk * 2, 64, Hkv, d).contiguous()
    v64 = v128.reshape(nblk, 2, 64, Hkv, d).reshape(nblk * 2, 64, Hkv, d).contiguous()
    npage64 = nblk * 2

    # Build the FUSED M3 cache [nblk, 2, 128, Hkv, d] with K=[:,0], V=[:,1].
    # NHD: contiguous as-is. HND: physical [nblk,2,Hkv,128,d] -> a non-contiguous
    # view back to logical [nblk,2,128,Hkv,d] (pos/head strides swapped), exactly
    # what vLLM passes under the HND cache layout.
    if layout == "NHD":
        kv_cache = torch.stack([k128, v128], dim=1).contiguous()  # [nblk,2,128,Hkv,d]
    else:  # HND
        kphys = k128.permute(0, 2, 1, 3).contiguous()  # [nblk,Hkv,128,d]
        vphys = v128.permute(0, 2, 1, 3).contiguous()
        fused_phys = torch.stack([kphys, vphys], dim=1).contiguous()  # [nblk,2,Hkv,128,d]
        kv_cache = fused_phys.permute(0, 1, 3, 2, 4)  # logical [nblk,2,128,Hkv,d], non-contig

    q = torch.randn(R, Hq, d, device=DEV, dtype=DTYPE)
    bt128 = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, -1).expand(R, nblk).contiguous()
    seq_lens = torch.full((R,), seq_kv, device=DEV, dtype=torch.int32)

    # Per-(req,kvh) selection. To exercise the per-kv-head index path, give each
    # kv-head a DISTINCT (shifted) block set when per_head_diff.
    ids128 = torch.full((R, Hkv, topk), -1, device=DEV, dtype=torch.int32)
    sel_per_head = {}
    for r in range(R):
        for h in range(Hkv):
            shift = (h * 1 + r) if per_head_diff else 0
            base = sorted(range(max(0, nblk - nsel - shift), nblk - shift))[:nsel]
            if not base:
                base = list(range(min(nsel, nblk)))
            sel_per_head[(r, h)] = base
            ids128[r, h, :len(base)] = torch.tensor(base, device=DEV, dtype=torch.int32)

    o_serve, _ = serve.forward_sparse_decode_serving(
        q, kv_cache, bt128, ids128, seq_lens, float(SCALE), Hkv, int(split_chunks))

    # GOLDEN per (req, q-head): page-64, causal=False over that head's kv-head set.
    o_gold = torch.empty(R, Hq, d, device=DEV, dtype=DTYPE)
    bt64 = torch.arange(npage64, device=DEV, dtype=torch.int32).view(1, -1).expand(1, npage64).contiguous()
    for r in range(R):
        for h in range(Hkv):
            base = sel_per_head[(r, h)]
            ids64 = []
            for b in base:
                ids64 += [2 * b, 2 * b + 1]
            ids64t = torch.tensor(ids64, device=DEV, dtype=torch.int32).view(1, -1).contiguous()
            # run golden for the full GQA group of this kv head, isolating heads
            qg = q[r:r+1].clone()  # [1,Hq,d]
            og, _ = paged.forward_sparse_paged(qg, k64, v64, bt64, ids64t,
                                               float(SCALE), False, int(seq_kv))
            for hh in range(h * g, (h + 1) * g):
                o_gold[r, hh] = og[0, hh]

    # dense fp32 ref
    Kf = k128.reshape(nblk * 128, Hkv, d).float()
    Vf = v128.reshape(nblk * 128, Hkv, d).float()
    o_ref = torch.empty(R, Hq, d, device=DEV, dtype=torch.float32)
    for r in range(R):
        for h in range(Hq):
            kh = h // g
            base = sel_per_head[(r, kh)]
            rows = []
            for b in base:
                rows += list(range(b * 128, (b + 1) * 128))
            rows = torch.tensor(rows, device=DEV)
            qh = q[r, h].float()
            Kh = Kf[rows, kh]; Vh = Vf[rows, kh]
            s = (Kh @ qh) * SCALE
            p = torch.softmax(s, 0)
            o_ref[r, h] = (p[:, None] * Vh).sum(0)

    def stats(a, b):
        a = a.float(); b = b.float()
        return (a - b).abs().max().item(), (a - b).pow(2).mean().sqrt().item()
    e_g, rms_g = stats(o_serve, o_gold)
    e_r, rms_r = stats(o_serve, o_ref)
    ok = rms_g < 1e-2 and rms_r < 5e-2
    print(f"seq_kv={seq_kv:6d} nsel={nsel:2d} chunks={split_chunks:2d} R={R} phd={int(per_head_diff)} "
          f"{layout} | vs GOLD rms={rms_g:.3e} max={e_g:.3e} | vs DENSE rms={rms_r:.3e} | {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    allok = True
    for seq_kv in (4096, 16384, 65536):
        for nsel in (16, 8, 3):
            for chunks in (0, 1, 4, 16):
                allok &= run_case(seq_kv, nsel, chunks, R=1, per_head_diff=True)
    # batched, per-head-different
    allok &= run_case(16384, 16, 0, R=2, per_head_diff=True)
    allok &= run_case(16384, 12, 4, R=3, per_head_diff=True)
    # HND cache layout (stride-swapped) -- exercises the real-stride path
    for chunks in (0, 16):
        allok &= run_case(16384, 16, chunks, R=1, per_head_diff=True, layout="HND")
        allok &= run_case(16384, 8, chunks, R=2, per_head_diff=True, layout="HND")
    print("ALL OK" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)
