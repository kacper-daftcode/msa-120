#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Minimal driver: build serving kernel, run forward_sparse_decode_serving in a
loop so ncu can profile sm120_serve_decode_partial_p128_ldsm. bs1 seq=16384.
"""
import os, torch
from torch.utils.cpp_extension import load
KDIR = "/work/msa_kernels_serving/kernels"
Hq, Hkv, d = 64, 4, 128
topk = 16
SCALE = 1.0 / (d ** 0.5)
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

serve = build("sm120_fmha_decode_serving", "sm120_fmha_decode_serving.cu", KDIR)
seq_kv, nsel, chunks = 16384, 16, 16
nblk = seq_kv // 128
k128 = torch.randn(nblk, 128, Hkv, d, device=DEV, dtype=torch.bfloat16)
v128 = torch.randn(nblk, 128, Hkv, d, device=DEV, dtype=torch.bfloat16)
kv_cache = torch.stack([k128, v128], dim=1).contiguous()
q = torch.randn(1, Hq, d, device=DEV, dtype=torch.bfloat16)
bt = torch.arange(nblk, device=DEV, dtype=torch.int32).view(1, -1).contiguous()
seq_lens = torch.full((1,), seq_kv, device=DEV, dtype=torch.int32)
ids = torch.full((1, Hkv, topk), -1, device=DEV, dtype=torch.int32)
base = list(range(nblk - nsel, nblk))
for h in range(Hkv):
    ids[0, h, :len(base)] = torch.tensor(base, device=DEV, dtype=torch.int32)

for _ in range(50):
    serve.forward_sparse_decode_serving(q, kv_cache, bt, ids, seq_lens, float(SCALE), Hkv, int(chunks))
torch.cuda.synchronize()
