#!/usr/bin/env python3
"""Build the decode kernel with ptxas -v + lineinfo and keep intermediates."""
import os
os.environ["TORCH_CUDA_ARCH_LIST"]=""  # we pass gencode explicitly
SRC="/work/decode_kernel/sm120_fmha_decode.cu"
OUT="/work/decode_kernel/nsys_stats/sass_build"
os.makedirs(OUT, exist_ok=True)
# header symlinks (validated recipe)
cu13="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
inc="/usr/local/cuda/targets/x86_64-linux/include"
for h in ("cusparse.h","cusolverDn.h","cusolver_common.h"):
    d=os.path.join(inc,h); s=os.path.join(cu13,h)
    if not os.path.exists(d) and os.path.exists(s):
        try: os.symlink(s,d)
        except OSError: pass
import torch
from torch.utils.cpp_extension import include_paths
incs = include_paths()
inc_flags = " ".join(f"-I{p}" for p in incs) + " -I/usr/include/python3.12"
# compile to cubin directly with verbose ptxas + lineinfo, keep .ptx
import subprocess
cmd = (f"nvcc -gencode=arch=compute_120f,code=sm_120f -O3 -std=c++17 "
       f"--expt-relaxed-constexpr -lineinfo "
       f"-Xptxas -v --ptxas-options=-v "
       f"-I/work/decode_kernel {inc_flags} "
       f"--keep --keep-dir {OUT} "
       f"-cubin {SRC} -o {OUT}/sm120_fmha_decode.cubin")
print("CMD:", cmd, flush=True)
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("=== STDOUT ==="); print(r.stdout)
print("=== STDERR (ptxas -v) ==="); print(r.stderr)
print("rc=", r.returncode)
