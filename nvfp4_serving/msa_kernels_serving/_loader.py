# SPDX-License-Identifier: MIT
"""Lazy JIT loader for the SM120 MSA kernel extensions.

VALIDATED on 2026-06-14 inside ``vllm/vllm-openai:minimax-m3``
(torch 2.11.0+cu130, nvcc 13.0, RTX PRO 6000 / SM120). All four .cu files
JIT-compile and run; the paged attend + topk pass their standalone golden tests
and topk passes op-equivalence vs the vLLM Triton reference (see
op_equivalence_topk.py and RESULTS.md).

Two image-specific gotchas, both handled by ``prepare_build_env()`` below:

  1. The base image ships the CUDA *compiler* (/usr/local/cuda-13.0) but a few
     math-lib headers torch's ATen transitively includes (cusparse.h,
     cusolverDn.h, cusolver_common.h) live only in the pip wheel
     ``nvidia/cu13/include``. We symlink just those into the toolkit include dir
     so nvcc's own runtime/crt headers still win (mixing the *whole* cu13 include
     in front of /usr/local/cuda breaks __cudaLaunch host-stub codegen).

  2. The arch MUST be the SM120 *family* target ``compute_120f``/``sm_120f``,
     NOT plain ``compute_120`` -- the attend kernels use the consumer-Blackwell
     block-scaled MMA (``mma...kind::mxf8f6f4.block_scale.scale_vec::1X``) for the
     FP8-PV GEMM, which ptxas rejects on ``sm_120`` but accepts on ``sm_120f``.
"""

from __future__ import annotations

import functools
import os

_CSRC = os.environ.get(
    "SM120_MSA_CSRC",
    os.path.join(os.path.dirname(__file__), "kernels"),
)

# compute_120f / sm_120f -- see module docstring (block-scaled MMA needs the
# family target).
_CUDA_FLAGS = [
    "-gencode=arch=compute_120f,code=sm_120f",
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
]

# Math-lib headers missing from /usr/local/cuda in the base image but present in
# the cu13 pip wheel. Symlinked (not -I'd) so nvcc's own crt/runtime headers win.
_CU13_INCLUDE = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
_CUDA_TOOLKIT_INCLUDE = "/usr/local/cuda/targets/x86_64-linux/include"
_MISSING_HEADERS = ("cusparse.h", "cusolverDn.h", "cusolver_common.h")


def prepare_build_env() -> None:
    """Symlink the cu13 math-lib headers into the CUDA toolkit include dir.

    Idempotent; safe to call before every build. No-op if the headers already
    resolve (e.g. a future image that ships the full toolkit) or the cu13 wheel
    is absent.
    """
    if not os.path.isdir(_CU13_INCLUDE) or not os.path.isdir(_CUDA_TOOLKIT_INCLUDE):
        return
    for h in _MISSING_HEADERS:
        dst = os.path.join(_CUDA_TOOLKIT_INCLUDE, h)
        src = os.path.join(_CU13_INCLUDE, h)
        if not os.path.exists(dst) and os.path.exists(src):
            try:
                os.symlink(src, dst)
            except OSError:
                pass  # read-only fs / race -> let the build surface the real error


@functools.lru_cache(maxsize=None)
def _load(module_name: str, source: str):
    """JIT-load (or import an AOT-built) extension for one .cu file."""
    try:  # prefer an AOT-built module if a wheel shipped one
        import importlib

        return importlib.import_module(f"sm120_msa_kernels.{module_name}")
    except (ImportError, ModuleNotFoundError):
        pass
    prepare_build_env()
    from torch.utils.cpp_extension import load  # heavy: local import

    return load(
        name=module_name,
        sources=[os.path.join(_CSRC, source)],
        extra_include_paths=[_CSRC],
        extra_cuda_cflags=_CUDA_FLAGS,
        verbose=False,
    )


def indexer_ext():
    """sm120_indexer.cu -> .block_scores(...)  (full project+score pipeline)."""
    return _load("sm120_indexer", "sm120_indexer.cu")


def topk_ext():
    """sm120_sparse_topk.cu -> .topk_select(max_score, num_valid, fbeg, fend)."""
    return _load("sm120_sparse_topk", "sm120_sparse_topk.cu")


def perhead_ext():
    """sm120_fmha_perhead.cu -> .forward_sparse / .forward_sparse_perhead(...)."""
    return _load("sm120_fmha_perhead", "sm120_fmha_perhead.cu")


def paged_ext():
    """sm120_fmha_paged.cu -> .forward_sparse_paged(...)  (PAGE_SIZE==64 v1)."""
    return _load("sm120_fmha_paged", "sm120_fmha_paged.cu")


def decode_serving_ext():
    """sm120_fmha_decode_serving.cu -> .forward_sparse_decode_serving(...).

    Graph-capture-safe page-128 flash-decoding entrypoint for the LIVE serving
    path: block_ids [R,Hkv,topk] per-kv-head, seq_lens DEVICE int32 [R], M3 fused
    cache [num_blocks,2,128,Hkv,128] consumed via real strides (NHD/HND), no
    host .item() / .contiguous() on the cache. Derived from the validated W4=3
    _ldsm partial in decode_kernel/sm120_fmha_decode.cu (see that file's banner).
    """
    return _load("sm120_fmha_decode_serving", "sm120_fmha_decode_serving.cu")
