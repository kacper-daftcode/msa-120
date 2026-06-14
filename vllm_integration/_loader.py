# SPDX-License-Identifier: MIT
"""Lazy JIT/AOT loader for the SM120 MSA kernel extensions.

DRAFT — this is the *stub* import layer the two adapter impls call into. In a
real image build the four ``.cu`` files (``sm120_indexer.cu``,
``sm120_sparse_topk.cu``, ``sm120_fmha_perhead.cu``, ``sm120_fmha_paged.cu``)
are compiled either:

  * AOT into a wheel during the image build (preferred — the official
    ``vllm/vllm-openai:minimax-m3`` image may not ship ``nvcc``; verify before
    relying on first-import JIT), or
  * JIT via ``torch.utils.cpp_extension.load`` on first import (needs nvcc +
    CUDA toolkit present in the container).

The gencode flags mirror ``tests/test_sm120_sparse_topk.py``:
``-gencode=arch=compute_120f,code=sm_120f``.

Each ``.cu`` is its own ``PYBIND11_MODULE(TORCH_EXTENSION_NAME, ...)``, so they
load as four separate extension modules. We memoize per module name.
"""

from __future__ import annotations

import functools
import os

# Path to python/fmha_sm100/csrc inside whatever wheel/overlay ships the kernels.
# In the image overlay this resolves under dist-packages (see selector_patch.md).
_CSRC = os.environ.get(
    "SM120_MSA_CSRC",
    os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc"),
)

_CUDA_FLAGS = [
    "-gencode=arch=compute_120f,code=sm_120f",
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
]


@functools.lru_cache(maxsize=None)
def _load(module_name: str, source: str):
    """JIT-load (or import the AOT-built) extension for one .cu file."""
    # Prefer an AOT-built module if present (set by the wheel build).
    try:  # pragma: no cover - depends on image build
        import importlib

        return importlib.import_module(f"sm120_msa_kernels.{module_name}")
    except (ImportError, ModuleNotFoundError):
        pass
    from torch.utils.cpp_extension import load  # local import: heavy

    return load(
        name=module_name,
        sources=[os.path.join(_CSRC, source)],
        extra_include_paths=[_CSRC],
        extra_cuda_cflags=_CUDA_FLAGS,
        verbose=False,
    )


def indexer_ext():
    """sm120_indexer.cu -> .block_scores(...)"""
    return _load("sm120_indexer", "sm120_indexer.cu")


def topk_ext():
    """sm120_sparse_topk.cu -> .topk_select(max_score, num_valid, fbeg, fend)"""
    return _load("sm120_sparse_topk", "sm120_sparse_topk.cu")


def perhead_ext():
    """sm120_fmha_perhead.cu -> .forward_sparse_perhead(...)"""
    return _load("sm120_fmha_perhead", "sm120_fmha_perhead.cu")


def paged_ext():
    """sm120_fmha_paged.cu -> .forward_sparse_paged(...)"""
    return _load("sm120_fmha_paged", "sm120_fmha_paged.cu")
